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
than a prior. When it cannot be reached the node asks E-utilities for PubMed's own
cited-by counts instead — a worse signal, and still a signal.
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
    """Attach iCite metrics, or E-utilities counts, or neither. Never fails the run.

    Ranking has five other components, and losing the run over a third-party service
    being down would be a poor trade for one of six.

    iCite first because its RCR is normalized against a field baseline, and that is the
    difference between "well cited" and "well cited for this literature" — a corpus
    spanning a large field and a small one ranks on field size without it. The fallback
    recovers a raw count only, which is why it is a fallback and not the primary.
    """
    if not candidates:
        return []
    deps.progress(NODE, f"citation metrics for {len(candidates)} papers")
    try:
        metrics = await deps.clients.icite.metrics([c.pmid for c in candidates], node=NODE)
    except Exception as exc:  # any client failure degrades the same way
        # The message and not only the type. A TLS trust failure arrives here carrying the
        # one sentence in the run that says what to do about it, and `HttpError` on its own
        # throws that away — the degrade is silent about a cause the user can actually fix.
        # Already redacted: an `HttpError` never carries a credential.
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return await _cited_by_fallback(deps, candidates, warnings, detail)

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


async def _cited_by_fallback(
    deps: Deps,
    candidates: list[Candidate],
    warnings: list[str],
    detail: str,
) -> list[Candidate]:
    """PubMed's own cited-by links, when iCite could not be reached.

    Worth a second call because the alternative is not a weaker signal, it is no signal:
    an unmeasured citation component scores every paper in the corpus at the same neutral
    value, and a constant added to every score changes no ordering at all. A count that
    runs low still ranks a decade-old paper above one nobody has read.

    On E-utilities, which the run has already been talking to successfully for every
    search and fetch — so the host that just failed is not the host being asked. Both
    warnings are kept when this fails too: the second says the signal is gone, and only
    the first says what to do about it.
    """
    try:
        counts = await deps.clients.eutils.cited_by_counts(
            [c.pmid for c in candidates], node=NODE
        )
    except Exception as exc:
        counts = {}
        second = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        # Into `warnings` and not only onto the bus. A live run showed this exact
        # failure and the bundle recorded nothing about it: `log.md` is written from
        # `warnings`, so a bus-only warning is one nobody reading the bundle can see.
        note = f"cited-by counts also unavailable ({second})"
        warnings.append(note)
        deps.warn(NODE, note)

    if not counts:
        # Not marked measured. Zero for every paper is the floor applied corpus-wide,
        # which is an inverted signal rather than an absent one.
        note = f"iCite unavailable ({detail}); ranking without citation metrics"
        warnings.append(note)
        deps.warn(NODE, note)
        return candidates

    note = (
        f"iCite unavailable ({detail}); ranking on PubMed cited-by counts for "
        f"{len(counts)} of {len(candidates)} paper(s) instead, which are not normalized "
        f"by field and run lower than iCite's"
    )
    warnings.append(note)
    deps.warn(NODE, note)
    return [
        c.with_citation_count(counts[c.pmid]) if c.pmid in counts else c
        for c in candidates
    ]
