"""The rank node: enrich with citation metrics, score, then diversify.

All code. Ranking decides what the screener sees, which is the largest cost in a run,
so it has to be reproducible: the same corpus and charter must yield the same pool.

Both selections are computed and both are kept. The pure relevance top-N is not used
for anything downstream — it exists so `selection_diff` can say what MMR and the
per-topic quota actually changed on this corpus, which `--dry-run` prints. A
diversification pass nobody can see the effect of is indistinguishable from one that
silently does nothing.

iCite is a network call, not a model call, so it runs on a dry run too. It is one
request per five hundred PMIDs and it is what makes the citation component real rather
than a prior.
"""

from __future__ import annotations

from typing import Any

from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.ranking import quota_select, score_all, selection_diff
from okf_loremaster.schemas import Candidate

__all__ = ["rank_node"]

NODE = "rank"


async def rank_node(state: RunState, deps: Deps) -> dict[str, Any]:
    candidates: list[Candidate] = list(state.get("unique") or [])
    charter = state.get("charter")
    query_topic: dict[str, str] = dict(state.get("query_topic") or {})
    warnings = list(state.get("warnings") or [])

    with span(deps, NODE) as report:
        enriched = await _enrich(deps, candidates, warnings)

        scored = score_all(
            enriched,
            now_year=deps.now_year,
            min_year=charter.min_year if charter is not None else None,
        )
        pure = scored[: deps.pool_size]
        pool = quota_select(
            scored,
            query_topic=query_topic,
            pool_size=deps.pool_size,
            lambda_=deps.mmr_lambda,
        )
        comparison = selection_diff(pure, pool, query_topic=query_topic)

        deps.progress(NODE, comparison.summary())
        report["summary"] = (
            f"{len(pool)} of {len(scored)} pooled; {comparison.changed} owed to "
            "diversification"
        )

    return {"scored": scored, "pool": pool, "comparison": comparison, "warnings": warnings}


async def _enrich(
    deps: Deps, candidates: list[Candidate], warnings: list[str]
) -> list[Candidate]:
    """Attach iCite metrics. A failure costs a signal, never the run.

    Ranking has five other components, and losing the run over a third-party service
    being down would be a poor trade for one of six.
    """
    if not candidates:
        return []
    deps.progress(NODE, f"citation metrics for {len(candidates)} papers")
    try:
        metrics = await deps.clients.icite.metrics([c.pmid for c in candidates], node=NODE)
    except Exception as exc:  # any client failure degrades the same way
        note = f"iCite unavailable ({type(exc).__name__}); ranking without citation metrics"
        warnings.append(note)
        deps.warn(NODE, note)
        return candidates

    missing = 0
    enriched: list[Candidate] = []
    for candidate in candidates:
        found = metrics.get(candidate.pmid)
        if found is None:
            missing += 1
            enriched.append(candidate)
        else:
            enriched.append(candidate.with_metrics(found))

    if missing:
        deps.progress(NODE, f"{missing} paper(s) had no iCite record")
    return enriched
