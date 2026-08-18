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

**`--basis` is enforced here, before the screener, and that placement is the whole
design.** Whether a paper is actually in the open-access subset is only knowable from a
BioC response, and BioC is called in `fulltext` — which sits *after* `curate`, downstream
of the only edge that can go back and search again. A shortfall discovered there is
terminal, and it would have been paid for twice over: once in screening budget spent on
papers that could never qualify, and once in a corpus that quietly comes up short of the
charter's topic floors. Checking availability at the tail of this node costs at most
`pool_size` requests, they are cached so `fulltext` re-reads them for free, and a
shortfall is still somewhere the re-query edge can do something about.
"""

from __future__ import annotations

import asyncio
from typing import Any

from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.ranking import quota_select, score_all, selection_diff
from okf_loremaster.schemas import Candidate, ScoredCandidate, TextBasisPolicy

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
        scored = await _apply_basis(deps, scored, warnings)
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
            f"{len(pool)} of {len(scored)} pooled; {comparison.changed} owed to diversification"
        )

    return {"scored": scored, "pool": pool, "comparison": comparison, "warnings": warnings}


async def _apply_basis(
    deps: Deps, scored: list[ScoredCandidate], warnings: list[str]
) -> list[ScoredCandidate]:
    """Narrow the ranked corpus to what `--basis` says the run is willing to read.

    A no-op on the default, which is why the default costs nothing.

    Under `abstract`, one filter and no network: a paper PubMed serves no abstract for
    cannot be read abstract-only, and extracting it anyway produces a document whose
    every field failed verification against the string `(no abstract available)`.

    Under `full-text`, two passes. A PMC id is necessary and nowhere near sufficient —
    most PMC-linked papers are not in the open-access subset — so the free filter runs
    first and BioC is asked only about what survives it, top-ranked first and never more
    than `pool_size` of them. Availability is read off the body, never off the status
    code: BioC answers "not open access" with HTTP 200 and a plain-text `[Error]`, which
    `fetch` already turns into `None`.

    A request that *fails* is not an answer, so the paper is kept. Losing a genuinely
    open-access paper to one timeout would be a silent, permanent hole in the corpus,
    where keeping it costs only a paper that falls back to its abstract in `fulltext` —
    and `text_basis` in the bundle says so, per document.
    """
    if deps.basis is TextBasisPolicy.ANY or not scored:
        return scored

    if deps.basis is TextBasisPolicy.ABSTRACT:
        kept = [entry for entry in scored if entry.candidate.has_abstract]
        _warn_shortfall(deps, warnings, kept=len(kept), of=len(scored), what="an abstract")
        return kept

    linked = [entry for entry in scored if entry.candidate.may_have_full_text]
    checked, unchecked = linked[: deps.pool_size], linked[deps.pool_size :]
    if unchecked:
        # Said out loud rather than left implicit. A bounded check that reads as an
        # exhaustive one is how a corpus comes up short for a reason nobody can find.
        note = (
            f"--basis full-text checked open-access availability for the top "
            f"{len(checked)} PMC-linked paper(s) of {len(linked)}; the remaining "
            f"{len(unchecked)} were ranked too low to reach the pool and were dropped "
            f"unchecked"
        )
        warnings.append(note)
        deps.warn(NODE, note)

    deps.progress(NODE, f"open-access check for {len(checked)} papers")
    available = await _open_access(deps, checked)
    _warn_shortfall(
        deps, warnings, kept=len(available), of=len(scored), what="open-access full text"
    )
    return available


async def _open_access(deps: Deps, scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """The subset BioC will actually serve full text for, in ranked order.

    One request per paper, all at once — the shared NCBI limiter decides how many are in
    flight, and it is the same limiter and the same client `fulltext` uses, so the
    responses are already in the HTTP cache when that node asks for them again.
    """
    if not scored:
        return []

    async def available(entry: ScoredCandidate) -> bool:
        try:
            return await deps.clients.bioc.fetch(entry.candidate.pmcid or "", node=NODE) is not None
        except Exception:
            return True

    verdicts = await asyncio.gather(*(available(entry) for entry in scored))
    return [entry for entry, ok in zip(scored, verdicts, strict=True) if ok]


def _warn_shortfall(deps: Deps, warnings: list[str], *, kept: int, of: int, what: str) -> None:
    """Say what the policy cost, always, even when it cost nothing worth noticing.

    A restricted basis is the user's decision and this does not second-guess it. What it
    refuses to do is let the decision be invisible: a corpus of 60 papers where 200 were
    asked for looks like a thin literature unless something says it was a filter.
    """
    if kept == of:
        return
    note = (
        f"--basis {deps.basis.value} kept {kept} of {of} ranked paper(s) — the rest have "
        f"no {what}. Topic floors and the paper target are measured against what is left, "
        f"so both may come up short of a default run on the same searches"
    )
    warnings.append(note)
    deps.warn(NODE, note)


async def _enrich(deps: Deps, candidates: list[Candidate], warnings: list[str]) -> list[Candidate]:
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
        counts = await deps.clients.eutils.cited_by_counts([c.pmid for c in candidates], node=NODE)
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
    return [c.with_citation_count(counts[c.pmid]) if c.pmid in counts else c for c in candidates]
