"""The curate node: one MID judgment per shelf, then arithmetic.

The screener saw each paper alone. The curator sees a shelf whole, which is the only
point in the run where "these four papers report the same result from the same cohort"
is a thing anyone can notice. That is why curation is per shelf and not per paper, and
why it is worth a MID model at a few calls rather than a FAST one at a few hundred.

Everything after the judgment is `curation.enforce_bounds` — the ceiling, the floor, the
global target. Splitting it that way is what makes shelf sizes reproducible across runs
even though the judgment behind them is not.

**A re-query round re-curates only the shelves that came up short.** A gapped shelf is
by definition a small one, so it is re-offered whole rather than only its new papers:
seeing the shelf entire is the whole reason this node exists, and on a shelf under its
floor that costs almost nothing. Shelves that were already fine keep their first-round
decisions untouched, and are not paid for twice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from okf_loremaster.config import Role
from okf_loremaster.curation import enforce_bounds
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.prompts import CURATE_SYSTEM, curate_user
from okf_loremaster.ranking import shelf_affinity, similarity, text_tokens, tokens
from okf_loremaster.schemas import (
    Candidate,
    Charter,
    CurationDecision,
    CurationResult,
    ScreenVerdict,
    Shelf,
    ShelfCuration,
    ShelfGap,
)
from okf_loremaster.schemas.parse import parse_model, response_format_for

__all__ = ["curate_node"]

NODE = "curate"

# Papers offered to a shelf, as a multiple of its ceiling. Past this the call is paying
# tokens to reject papers the ranking has already put behind three times as many better
# ones, and they are all still in the reserve if the shelf comes up short.
OFFER_MULTIPLE = 3

# Room for one short decision per offered paper, plus the `missing` line.
CURATE_TOKENS_PER_PAPER = 45
CURATE_TOKENS_FLOOR = 512

# Reserve tiers, best first: papers the screener included that the shelf had no room to
# offer, then papers it excluded but rated relevant, then papers the curator saw and
# turned down. A curator's rejection is the best-informed "no" in the run, so it is the
# last thing a floor backfill reaches for.
_OVERFLOW, _BORDERLINE, _REJECTED = 0, 1, 2


async def curate_node(state: RunState, deps: Deps) -> dict[str, Any]:
    charter = state.get("charter")
    if charter is None:
        raise RuntimeError("curate reached without a charter — the graph is wired wrong")

    verdicts = {verdict.pmid: verdict for verdict in state.get("verdicts") or []}
    by_pmid = {c.pmid: c for c in (state.get("unique") or state.get("candidates") or [])}
    query_shelf = dict(state.get("query_shelf") or {})
    warnings = list(state.get("warnings") or [])
    prior = state.get("curation")

    with span(deps, NODE) as report:
        order = _rank(verdicts, [item.pmid for item in state.get("pool") or []])
        assigned = _assign(charter, verdicts, by_pmid, query_shelf)
        offers = _offers(charter, assigned, verdicts, order)

        decisions, missing = await _curate_all(
            deps, charter, offers, verdicts, by_pmid, order, prior, warnings
        )
        placement = enforce_bounds(
            charter,
            kept={slug: [d.pmid for d in ds if d.keep] for slug, ds in decisions.items()},
            rank=order,
            reserve=_reserve(charter, assigned, verdicts, decisions, order),
            missing=missing,
        )
        for note in placement.warnings:
            warnings.append(note)
            deps.warn(NODE, note)

        result = _result(placement.shelves, placement.gaps, decisions)
        report["summary"] = placement.summary()

    return {"curation": result, "shelves": placement.shelves, "warnings": warnings}


# --- deciding what each shelf is offered ------------------------------------


def _rank(verdicts: Mapping[str, ScreenVerdict], pool: Sequence[str]) -> dict[str, int]:
    """One total order over every screened paper: relevance first, then pool position.

    Relevance leads because it is the only signal in the run that read the paper against
    this review's question; the ranker's position is a proxy chosen before anyone had.
    Position breaks the ties inside a relevance band, and the PMID breaks what is left,
    so the order is total and a rerun on the same corpus reproduces it exactly.

    Papers that were screened but have since fallen out of the pool — a second search
    round grows the corpus and the pool is a fixed size — sort after the pooled ones
    rather than being dropped. We paid to screen them.
    """
    position = {pmid: index for index, pmid in enumerate(pool)}
    unpooled = len(position)
    ordered = sorted(
        verdicts,
        key=lambda pmid: (-verdicts[pmid].relevance, position.get(pmid, unpooled), pmid),
    )
    return {pmid: index for index, pmid in enumerate(ordered)}


def _assign(
    charter: Charter,
    verdicts: Mapping[str, ScreenVerdict],
    by_pmid: Mapping[str, Candidate],
    query_shelf: Mapping[str, str],
) -> dict[str, str]:
    """A shelf for every screened paper, or nothing if none of the three sources knows.

    Three sources, in descending order of how much they read: the screener's own hint,
    which shelf's query found the paper, and lexical overlap with the shelf's title,
    scope and seed terms. The third exists because the broad queries in a plan carry no
    shelf affinity at all, and without it every paper they alone found would be
    unofferable — a silent loss of exactly the papers a taxonomy did not anticipate.
    """
    known = set(charter.slugs)
    vectors = {shelf.slug: _shelf_tokens(shelf) for shelf in charter.shelf_taxonomy}

    assigned: dict[str, str] = {}
    for pmid, verdict in verdicts.items():
        candidate = by_pmid.get(pmid)
        if candidate is None:
            continue
        shelf = verdict.shelf if verdict.shelf in known else ""
        if not shelf:
            found = shelf_affinity(candidate, query_shelf)
            shelf = found if found in known else ""
        if not shelf:
            shelf = _nearest(candidate, charter, vectors)
        if shelf:
            assigned[pmid] = shelf
    return assigned


def _shelf_tokens(shelf: Shelf) -> frozenset[str]:
    return text_tokens(" ".join([shelf.title, shelf.scope, *shelf.seed_terms]))


def _nearest(
    candidate: Candidate, charter: Charter, vectors: Mapping[str, frozenset[str]]
) -> str:
    """The shelf whose own words this paper's words most overlap. `""` if none do.

    A weak signal, used only where there is no other. Strict `>` walking the charter's
    slug order means a tie goes to the earlier shelf every time rather than to whichever
    one a dict happened to yield first.
    """
    best, best_score = "", 0.0
    paper = tokens(candidate)
    for slug in charter.slugs:
        score = similarity(paper, vectors.get(slug, frozenset()))
        if score > best_score:
            best, best_score = slug, score
    return best


def _offers(
    charter: Charter,
    assigned: Mapping[str, str],
    verdicts: Mapping[str, ScreenVerdict],
    order: Mapping[str, int],
) -> dict[str, list[str]]:
    """The included papers each shelf is asked about, best first and capped."""
    cap = max(1, charter.shelf_max * OFFER_MULTIPLE)
    offers: dict[str, list[str]] = {slug: [] for slug in charter.slugs}
    for pmid, slug in assigned.items():
        if verdicts[pmid].include and slug in offers:
            offers[slug].append(pmid)
    for pmids in offers.values():
        pmids.sort(key=lambda pmid: order.get(pmid, len(order)))
        del pmids[cap:]
    return offers


# --- the judgment -----------------------------------------------------------


async def _curate_all(
    deps: Deps,
    charter: Charter,
    offers: Mapping[str, list[str]],
    verdicts: Mapping[str, ScreenVerdict],
    by_pmid: Mapping[str, Candidate],
    order: Mapping[str, int],
    prior: CurationResult | None,
    warnings: list[str],
) -> tuple[dict[str, list[CurationDecision]], dict[str, str]]:
    """One call per shelf that needs one. Returns decisions and `missing`, both by shelf."""
    carried = _carried(prior)
    todo = [
        shelf
        for shelf in charter.shelf_taxonomy
        if offers.get(shelf.slug) and _needs_call(shelf.slug, offers, prior, carried)
    ]

    decisions: dict[str, list[CurationDecision]] = {
        slug: list(found) for slug, found in carried.items()
    }
    missing: dict[str, str] = _carried_missing(prior)

    if todo and deps.router is None:
        note = "no model is available, so curation kept the screener's best-ranked papers"
        warnings.append(note)
        deps.warn(NODE, note)

    total = len(todo)
    if total:
        deps.progress(NODE, f"curating {total} shelf/shelves", current=0, total=total)
    results = await asyncio.gather(
        *(
            _curate_one(deps, charter, shelf, offers[shelf.slug], verdicts, by_pmid, warnings)
            for shelf in todo
        )
    )
    for done, (shelf, (judged, note)) in enumerate(zip(todo, results, strict=True), start=1):
        deps.progress(NODE, f"curated {shelf.slug}", current=done, total=total)
        decisions[shelf.slug] = _reconcile(shelf.slug, offers[shelf.slug], judged, order)
        missing[shelf.slug] = note

    return decisions, missing


def _needs_call(
    slug: str,
    offers: Mapping[str, list[str]],
    prior: CurationResult | None,
    carried: Mapping[str, list[CurationDecision]],
) -> bool:
    """Whether this shelf is worth a call on this round.

    Every shelf on the first round. On a re-query round, only a shelf that was under its
    floor *and* now has a paper it has not seen: a shelf that came up short and got
    nothing new has already given its answer, and re-asking would spend a call to
    receive it again.
    """
    if prior is None:
        return True
    if not any(gap.shelf == slug and gap.shortfall > 0 for gap in prior.gaps):
        return False
    seen = {decision.pmid for decision in carried.get(slug, ())}
    return any(pmid not in seen for pmid in offers.get(slug, ()))


def _carried(prior: CurationResult | None) -> dict[str, list[CurationDecision]]:
    if prior is None:
        return {}
    carried: dict[str, list[CurationDecision]] = {}
    for decision in prior.decisions:
        if decision.shelf:
            carried.setdefault(decision.shelf, []).append(decision)
    return carried


def _carried_missing(prior: CurationResult | None) -> dict[str, str]:
    return {} if prior is None else {gap.shelf: gap.missing for gap in prior.gaps}


async def _curate_one(
    deps: Deps,
    charter: Charter,
    shelf: Shelf,
    offered: list[str],
    verdicts: Mapping[str, ScreenVerdict],
    by_pmid: Mapping[str, Candidate],
    warnings: list[str],
) -> tuple[ShelfCuration | None, str]:
    """Ask about one shelf. `None` when the call could not be made or read."""
    if deps.router is None:
        return None, ""

    papers = [
        (
            pmid,
            by_pmid[pmid].title,
            verdicts[pmid].relevance,
            verdicts[pmid].reason,
        )
        for pmid in offered
        if pmid in by_pmid
    ]
    try:
        result = await deps.router.complete(
            Role.MID,
            [
                {"role": "system", "content": CURATE_SYSTEM},
                {
                    "role": "user",
                    "content": curate_user(
                        task=charter.task or charter.prompt,
                        shelf=shelf.slug,
                        scope=shelf.scope,
                        seed_terms=list(shelf.seed_terms),
                        floor=charter.shelf_min,
                        ceiling=charter.shelf_max,
                        papers=papers,
                    ),
                },
            ],
            node=NODE,
            max_tokens=max(CURATE_TOKENS_FLOOR, len(papers) * CURATE_TOKENS_PER_PAPER),
            response_format=response_format_for(ShelfCuration, name="shelf_curation"),
        )
        judged = parse_model(result.text, ShelfCuration)
    except Exception as exc:  # SchemaError included: both degrade the same way
        # A shelf whose call failed keeps the screener's best-ranked papers rather than
        # emptying: the screener read every one of them against this review's question,
        # and an empty shelf would be reported as a search failure it is not.
        note = (
            f"curation of the {shelf.slug} shelf failed ({type(exc).__name__}); "
            "kept the screener's best-ranked papers instead"
        )
        warnings.append(note)
        deps.warn(NODE, note)
        return None, ""

    return judged, judged.missing


def _reconcile(
    slug: str,
    offered: Sequence[str],
    judged: ShelfCuration | None,
    order: Mapping[str, int],
) -> list[CurationDecision]:
    """Turn one reply into a decision per offered paper, and only per offered paper.

    Two things the model may do are handled rather than trusted away: it may answer
    about a PMID nobody offered — dropped, since placing a paper on the strength of an
    id the curator invented is worse than losing it — and it may say nothing about a
    paper that was offered. Silence is not consent: an unanswered paper is not kept, and
    lands in the reserve where a thin shelf can still reach it.
    """
    if judged is None:
        # The failure path: keep the best-ranked, up to what a shelf would hold anyway.
        ranked = sorted(offered, key=lambda pmid: order.get(pmid, len(order)))
        return [
            CurationDecision(pmid=pmid, keep=True, shelf=slug, rationale="curation unavailable")
            for pmid in ranked
        ]

    answered = {decision.pmid: decision for decision in judged.decisions}
    return [
        CurationDecision(
            pmid=pmid,
            keep=answered[pmid].keep if pmid in answered else False,
            shelf=slug,
            rationale=(
                answered[pmid].rationale if pmid in answered else "the curator did not answer"
            ),
        )
        for pmid in offered
    ]


# --- what the bounds get to work with ---------------------------------------


def _reserve(
    charter: Charter,
    assigned: Mapping[str, str],
    verdicts: Mapping[str, ScreenVerdict],
    decisions: Mapping[str, Sequence[CurationDecision]],
    order: Mapping[str, int],
) -> dict[str, list[str]]:
    """The fallback queue per shelf, best candidate first.

    Everything a shelf could still take, tiered by how strong the "no" against it was
    (see `_OVERFLOW` / `_BORDERLINE` / `_REJECTED`) and ranked within each tier. A paper
    the screener called unrelated never appears here at any tier: refilling a shelf with
    those would make the floor a number rather than a claim about the literature.
    """
    turned_down = {
        slug: {d.pmid for d in found if not d.keep} for slug, found in decisions.items()
    }
    judged = {d.pmid for found in decisions.values() for d in found}

    queued: dict[str, list[tuple[int, int, str]]] = {slug: [] for slug in charter.slugs}
    for pmid, slug in assigned.items():
        if slug not in queued:
            continue
        if pmid in turned_down.get(slug, frozenset()):
            tier = _REJECTED
        elif pmid in judged:
            continue  # kept already, so not a fallback for anything
        elif verdicts[pmid].include:
            tier = _OVERFLOW
        elif verdicts[pmid].borderline:
            tier = _BORDERLINE
        else:
            continue
        queued[slug].append((tier, order.get(pmid, len(order)), pmid))

    return {slug: [pmid for *_, pmid in sorted(rows)] for slug, rows in queued.items()}


def _result(
    shelves: Mapping[str, Sequence[str]],
    gaps: Sequence[ShelfGap],
    decisions: Mapping[str, Sequence[CurationDecision]],
) -> CurationResult:
    """One record whose `by_shelf()` is the placement, not an earlier draft of it.

    The curator's decisions and the final shelves disagree by design — the bounds trim
    and backfill after the fact — so the decisions are rewritten to match what was
    actually placed. Keeping both versions would leave two answers to "is this paper in
    the bundle", and downstream would eventually read the wrong one.
    """
    rationales = {
        decision.pmid: decision.rationale
        for found in decisions.values()
        for decision in found
    }
    final: list[CurationDecision] = []
    placed: set[str] = set()
    for slug, pmids in shelves.items():
        for pmid in pmids:
            placed.add(pmid)
            final.append(
                CurationDecision(
                    pmid=pmid,
                    keep=True,
                    shelf=slug,
                    rationale=rationales.get(pmid, "backfilled to keep the shelf above its floor"),
                )
            )
    for slug, found in decisions.items():
        for decision in found:
            if decision.pmid not in placed:
                final.append(decision.model_copy(update={"keep": False, "shelf": slug}))
    return CurationResult(decisions=final, gaps=list(gaps))
