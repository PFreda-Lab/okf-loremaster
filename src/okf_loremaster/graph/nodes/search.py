"""The search node: plan queries, run them, and fetch what they returned.

Planning is judgment and goes to the balanced-tier model; everything after it is code. The model
proposes concepts and boolean structure, and `queries.with_filters` adds the language
and date filters afterward so that every query in a plan carries identical ones. A dry
run skips the model entirely and uses `queries.deterministic_plan`, which is what makes
`--dry-run` cost nothing while still reporting the hit counts a real run would see.

Every executed query keeps its `query_translation`, and every translation is inspected.
PubMed does not reject an unknown field tag — it rewrites the clause into a free-text
search, returns orders of magnitude more hits, and reports an empty error list. The
warning that raises is advisory: the papers are real, and a run is better off ranking
them with a note attached than refusing to continue.

One `efetch` covers every unique PMID across the whole plan, so a paper found by four
queries is fetched once and arrives already merged, carrying all four provenances and
its best rank among them.

**A second round adds to the corpus rather than replacing it.** LangGraph's state
channels are last-value-wins, so `candidates`, `executed` and `query_topic` are merged
with what the first round left behind — otherwise a re-query round aimed at two thin
topics would return holding only those two topics' papers. The plan for that round is
`queries.gap_plan`, built in code from the curator's own account of what each topic
lacks; no second planning call is made.
"""

from __future__ import annotations

from typing import Any

from okf_loremaster import queries
from okf_loremaster.clients.eutils import PubMedRecord
from okf_loremaster.config import Role
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.prompts import QUERY_PLAN_SYSTEM, query_plan_user
from okf_loremaster.schemas import Candidate, Charter, ExecutedQuery, QueryPlan
from okf_loremaster.schemas.parse import SchemaError, parse_model, response_format_for

__all__ = ["pending_gap_plan", "search_node"]

NODE = "search"


def pending_gap_plan(charter: Charter, state: RunState, deps: Deps) -> QueryPlan | None:
    """The plan a re-query round would run, or `None` if this is not one.

    Shared with the routing decision taken after curation, so that the edge back to
    search is followed only when there is something new to ask. Asking the same
    question twice is the one thing a bounded retry loop must not do, and the check is
    cheap enough to run in both places rather than to cache.
    """
    curation = state.get("curation")
    if curation is None or not curation.needs_more_search:
        return None
    return queries.gap_plan(
        charter,
        [gap for gap in curation.gaps if gap.shortfall > 0],
        already=[query.term for query in state.get("executed") or []],
        max_queries=deps.max_queries,
    )


async def search_node(state: RunState, deps: Deps) -> dict[str, Any]:
    charter = state.get("charter")
    warnings = list(state.get("warnings") or [])
    if charter is None:
        raise RuntimeError("search reached without a charter — the graph is wired wrong")

    prior_executed = list(state.get("executed") or [])
    prior_candidates = list(state.get("candidates") or [])
    query_topic = dict(state.get("query_topic") or {})
    rounds = int(state.get("rounds") or 0)

    with span(deps, NODE) as report:
        plan = await _plan(deps, charter, state, warnings)
        query_topic.update({q.term: q.topic for q in plan.queries})

        executed, hits = await _execute(deps, plan, warnings, search_round=rounds + 1)
        found = await _fetch(deps, hits)
        candidates = _merge(prior_candidates, found)

        suspect = sum(1 for q in executed if q.suspect)
        total_hits = sum(q.count for q in executed)
        added = len(candidates) - len(prior_candidates)
        report["summary"] = (
            f"{len(plan.queries)} queries, {total_hits:,} hits, "
            f"{len(found)} unique records"
            + (f" ({added} new)" if prior_candidates else "")
            + (f", {suspect} suspect" if suspect else "")
        )

    if not candidates and not state.get("dry_run"):
        # A run that retrieved nothing has nothing to screen, curate, extract or write,
        # and every one of those nodes is a well-behaved no-op on an empty list — so the
        # graph used to sail to the end and emit a bundle of eight empty topics that
        # passed validation and printed `complete ... bundle valid`. An empty answer
        # presented as a finished one is the same defect as a blank paper presented as a
        # read one: the tool reporting success for work it did not do. It stops here,
        # while the cause is still on screen and before the run spends anything more.
        # *Every* query, not most of them, which is the part worth reading twice. A bad
        # clause zeroes one query; only something the plan shares can zero all of them,
        # and what every query shares is the filters `with_filters` appends. So the
        # message names those first — it used to blame a long `population` or `outcome`
        # alone, which sent a real diagnosis looking at twelve well-formed queries for
        # an hour while the words `AND en[la]` sat at the end of each one.
        raise RuntimeError(
            "no papers retrieved: every query returned zero hits, so there is nothing "
            "to build a bundle from. The queries as PubMed ran them are in the log "
            "above. All of them failing points at what they share — the charter's "
            "`languages` and `min_year`, appended to each one — rather than at any "
            "single clause; failing that, an over-long `population` or `outcome` is "
            "searched as an exact phrase and matches nothing"
        )

    return {
        "plan": plan,
        "executed": [*prior_executed, *executed],
        "query_topic": query_topic,
        "candidates": candidates,
        "rounds": rounds + 1,
        "warnings": warnings,
    }


def _merge(prior: list[Candidate], found: list[Candidate]) -> list[Candidate]:
    """Union two rounds' candidates, folding a repeat sighting into the first.

    A paper both rounds returned is one paper found by more queries, and
    `merged_with` is what turns that into the ranking signal it is. Order is
    first-seen, so the second round appends rather than reshuffling.
    """
    if not prior:
        return found
    merged = list(prior)
    index = {candidate.pmid: position for position, candidate in enumerate(merged)}
    for candidate in found:
        position = index.get(candidate.pmid)
        if position is None:
            index[candidate.pmid] = len(merged)
            merged.append(candidate)
        else:
            merged[position] = merged[position].merged_with(candidate)
    return merged


# --- planning ---------------------------------------------------------------


async def _plan(deps: Deps, charter: Charter, state: RunState, warnings: list[str]) -> QueryPlan:
    gap = pending_gap_plan(charter, state, deps)
    if gap is not None:
        # A re-query round. Returning an empty plan when nothing new can be asked is
        # deliberate: it produces no new candidates, curation finds nothing to change,
        # and the run ends — rather than re-running round one's searches at full cost
        # to retrieve exactly the corpus already in hand.
        deps.progress(NODE, f"re-query round: {len(gap.queries)} query/queries for thin topics")
        return gap

    if deps.router is None:
        return queries.deterministic_plan(charter, max_queries=deps.max_queries)

    messages = [
        {"role": "system", "content": QUERY_PLAN_SYSTEM},
        {
            "role": "user",
            "content": query_plan_user(
                task=charter.task or charter.prompt,
                population=charter.population,
                outcome=charter.outcome,
                topics=[
                    (topic.slug, topic.scope, list(topic.seed_terms))
                    for topic in charter.topic_taxonomy
                ],
                max_queries=deps.max_queries,
            ),
        },
    ]
    try:
        result = await deps.router.complete(
            Role.BALANCED,
            messages,
            node=NODE,
            # One query per topic, each with its rationale, so this scales with the
            # taxonomy. 2048 was under it: a 4-topic plan measured 2,115 tokens and was
            # cut off, and the retry that fit wrote the same plan a second time at full
            # price. A ceiling costs nothing unspent.
            max_tokens=4096,
            response_format=response_format_for(QueryPlan, name="query_plan"),
        )
        planned = parse_model(result.text, QueryPlan)
    except SchemaError as exc:
        # A planning failure is recoverable in a way a charter failure is not: the
        # deterministic plan is worse but real, and losing the run over query syntax
        # would be a poor trade.
        note = f"query planning failed ({exc}); falling back to the deterministic plan"
        warnings.append(note)
        deps.warn(NODE, note)
        return queries.deterministic_plan(charter, max_queries=deps.max_queries)

    return _finalize(planned, charter, deps, warnings)


def _finalize(planned: QueryPlan, charter: Charter, deps: Deps, warnings: list[str]) -> QueryPlan:
    """Apply the charter's filters, drop duplicates, and check the topic slugs.

    A planner that names a topic the charter does not have is worth surfacing rather
    than failing on: the query is still valid, it just loses its quota affinity.
    """
    known = set(charter.slugs)
    seen: dict[str, None] = {}
    final = []
    for query in planned.queries:
        if not query.term.strip():
            continue
        if query.topic and query.topic not in known:
            note = f"planner named unknown topic {query.topic!r}; the query keeps no topic affinity"
            warnings.append(note)
            deps.warn(NODE, note)
            query = query.model_copy(update={"topic": ""})
        term = queries.with_filters(query.term, charter)
        if term in seen:
            continue
        seen[term] = None
        final.append(query.model_copy(update={"term": term}))

    if not final:
        note = "planner returned no usable queries; falling back to the deterministic plan"
        warnings.append(note)
        deps.warn(NODE, note)
        return queries.deterministic_plan(charter, max_queries=deps.max_queries)
    return QueryPlan(queries=final[: deps.max_queries])


# --- execution --------------------------------------------------------------


async def _execute(
    deps: Deps, plan: QueryPlan, warnings: list[str], *, search_round: int = 1
) -> tuple[list[ExecutedQuery], dict[str, dict[str, int]]]:
    """Run every query. Returns each query's record, and PMID to {query term: rank}.

    Every query that found a paper is kept, not just the best one. Convergence across
    independent queries is a quarter of the relevance score, and collapsing the map to
    one term per paper would silently zero that signal for the whole corpus.

    Queries run one after another rather than concurrently. They share one IP-enforced
    NCBI rate limit, so concurrency here buys nothing and only makes the progress
    reporting harder to follow.
    """
    executed: list[ExecutedQuery] = []
    hits: dict[str, dict[str, int]] = {}

    for index, query in enumerate(plan.queries, start=1):
        deps.progress(
            NODE,
            f"searching: {query.rationale or query.term[:60]}",
            current=index,
            total=len(plan.queries),
        )
        result = await deps.clients.eutils.esearch(
            query.term, retmax=deps.per_query_retmax, node=NODE
        )
        record = queries.executed(
            query.term,
            result,
            retrieved=len(result.ids),
            rationale=query.rationale,
            topic=query.topic,
            search_round=search_round,
        )
        executed.append(record)
        if record.suspect:
            note = f"{record.note} — query: {query.term}"
            warnings.append(note)
            deps.warn(NODE, note)

        for rank, pmid in enumerate(result.ids):
            found = hits.setdefault(pmid, {})
            if query.term not in found or rank < found[query.term]:
                found[query.term] = rank

    _report_empty(deps, executed, warnings)
    return executed, hits


def _report_empty(deps: Deps, executed: list[ExecutedQuery], warnings: list[str]) -> None:
    """Say when a search found nothing, and why it most likely did.

    A query that matches no papers is not an error and PubMed reports it as a perfectly
    successful search of zero results, so without this the run simply arrives at the
    retrieve pause with an empty pool and no account of itself. The cause is nearly
    always the same one: every clause is a quoted phrase, and a phrase longer than a
    paper would print matches nothing however well it describes the task.
    """
    empty = [query for query in executed if query.count == 0]
    if not empty or not executed:
        return

    if len(empty) == len(executed):
        note = (
            "no query returned a single hit. Every clause is searched as an exact "
            "phrase, so a long `outcome` or `population` in the charter matches "
            "nothing on its own — shorten them to phrases a paper would print, and "
            "rerun"
        )
    else:
        note = (
            f"{len(empty)} of {len(executed)} queries returned no hits; their topics "
            "will be thin or empty. The seed terms are searched as exact phrases"
        )
    warnings.append(note)
    deps.warn(NODE, note)


async def _fetch(deps: Deps, hits: dict[str, dict[str, int]]) -> list[Candidate]:
    """One efetch across every unique PMID, carrying its full search provenance."""
    if not hits:
        return []
    pmids = list(hits)
    deps.progress(NODE, f"fetching {len(pmids)} records", current=0, total=len(pmids))
    records: list[PubMedRecord] = await deps.clients.eutils.efetch(pmids, node=NODE)

    candidates: list[Candidate] = []
    for record in records:
        found = hits.get(record.pmid, {})
        candidate = Candidate.from_record(record)
        # Query order, not rank order: `found_by` is provenance, and the plan's order
        # is the one a reader can follow back to a rationale.
        candidate.found_by = list(found)
        candidate.best_rank = min(found.values()) if found else 0
        candidates.append(candidate)
    return candidates
