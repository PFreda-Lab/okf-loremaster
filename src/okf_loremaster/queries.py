"""Building PubMed queries, and catching the ones PubMed silently rewrote.

Two jobs, both deterministic:

**Construction.** Field tags, phrase quoting, and the language and date filters are
mechanical, so a model never writes a whole query string. The planner proposes concepts;
this module turns them into syntax. It also builds a plan with no model at all, which is
what makes `--dry-run` cost nothing.

**Verification.** PubMed does not reject an unknown field tag. It rewrites
`x[nosuchfield]` into a free-text search, returns orders of magnitude more hits, and
reports an *empty* `errorlist` while doing it. Recorded 2026-08-03:

    "postoperative respiratory failure"[Title/Abstract] AND "risk factors"[Title/Abstract]
        -> 79 hits, no errorlist at all

    postoperative respiratory failure[nosuchfield]
        -> 14,382 hits, fieldsnotfound: [], translation is a wall of [All Fields]

A query can therefore be malformed and successful at the same time, and the only
evidence is the translation. `inspect_translation` reads it.

Nothing here names a condition. Every term comes from the charter or the user's prompt.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from okf_loremaster.clients.eutils import ESearchResult
from okf_loremaster.schemas import Charter, ExecutedQuery, PlannedQuery, QueryPlan, TopicGap

__all__ = [
    "ALL_FIELDS",
    "MAX_GAP_TERMS",
    "MAX_PHRASE_WORDS",
    "deterministic_plan",
    "executed",
    "gap_plan",
    "inspect_translation",
    "keyphrases",
    "or_group",
    "phrase",
    "tiab",
    "untagged_clauses",
    "with_filters",
]

ALL_FIELDS = "[All Fields]"

# A phrase search on a long string finds nothing: PubMed matches it verbatim. Four words
# is about where a concept ends and a sentence begins.
MAX_PHRASE_WORDS = 4

# Concepts ORed into one gap query. Past this the clause matches on the weakest term in
# it, and a topic gets refilled with whatever that term happened to catch.
MAX_GAP_TERMS = 6

# Ordinary English function words. Nothing clinical, nothing project-specific.
# A wrapped block of words is reviewable at a glance in a way a quoted list is not,
# which is why SIM905 is waived here and for the two word lists that follow.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had has have
    how in into is it its may might must of on or over should so than that the their
    them then there these they this those to under upon was were what when where which
    while who whom why will with within would
    """.split()  # noqa: SIM905
)

# Words that describe the *task* rather than its subject. A prompt is written to a
# person ("build me a model that predicts X"), and only X belongs in a query.
# Every verb here carries its third-person form too: prompts are written both ways
# ("build a model to predict X" / "a model that predicts X"), and a missing inflection
# does not merely leave a stray word — it fuses two runs into one unsearchable phrase.
_TASK_WORDS = frozenset(
    """
    aim aims analysis analyze analyzes build builds building corpus data dataset
    develop develops developing engineer engineers engineering estimate estimates
    evidence feature features find finds identify identifies identifying literature
    model modeling models paper papers predict predicts predicting prediction
    predictive predictor predictors project research review reviews search searches
    study studies task using want wants work works
    """.split()  # noqa: SIM905
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
# A term carrying a field tag: either a quoted phrase or a bare token, then `[tag]`.
_TAGGED = re.compile(r'(?:"[^"]*"|[^\s()]+?)\s*\[[^\]]+\]')
_BOOLEAN = frozenset({"AND", "OR", "NOT"})


# --- construction -----------------------------------------------------------


def phrase(text: str) -> str:
    """Quote a term so PubMed treats it as a phrase rather than a boolean soup."""
    collapsed = " ".join(text.split())
    return f'"{collapsed}"'


def tiab(text: str) -> str:
    """Title/abstract search for one concept.

    `[tiab]` rather than `[mh]` on purpose: MeSH indexing lags publication by months,
    so a MeSH-only query systematically misses exactly the recent work a bundle is
    built to capture. MeSH enters later, as a ranking signal.
    """
    return f"{phrase(text)}[tiab]"


def or_group(terms: list[str]) -> str:
    """`(a OR b OR c)` over title/abstract, or `""` if nothing survives."""
    clauses = [tiab(t) for t in dict.fromkeys(t.strip() for t in terms) if t.strip()]
    if not clauses:
        return ""
    return clauses[0] if len(clauses) == 1 else "(" + " OR ".join(clauses) + ")"


def with_filters(term: str, charter: Charter) -> str:
    """Append the charter's language and date filters.

    Applied here rather than folded into each generated clause so that the filters are
    identical across every query in a plan, and so a translation can be read without
    untangling them.
    """
    parts = [f"({term})" if " " in term else term]
    if charter.languages:
        languages = " OR ".join(f"{code}[la]" for code in charter.languages)
        parts.append(f"({languages})" if len(charter.languages) > 1 else languages)
    if charter.min_year:
        parts.append(f"{charter.min_year}:3000[dp]")
    return " AND ".join(parts)


def keyphrases(text: str) -> list[str]:
    """Consecutive runs of content words, as phrase candidates.

    Stopwords and task words are what break a run, which turns "predict postoperative
    respiratory failure in adults" into `["postoperative respiratory failure",
    "adults"]`. Crude, and only ever used where no charter is available — a plan built
    this way measures the search surface of the prompt, not of a taxonomy.
    """
    runs: list[list[str]] = [[]]
    for match in _WORD.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        if lowered in _STOPWORDS or lowered in _TASK_WORDS or len(lowered) < 3:
            if runs[-1]:
                runs.append([])
            continue
        runs[-1].append(word)

    phrases: list[str] = []
    for run in runs:
        if run:
            phrases.append(" ".join(run[:MAX_PHRASE_WORDS]))
    return list(dict.fromkeys(phrases))


def deterministic_plan(charter: Charter, *, max_queries: int = 12) -> QueryPlan:
    """A query plan built with no model call at all.

    This is what `--dry-run` uses, and what a run falls back to if query planning
    fails. It is deliberately worse than the planned version — it can only recombine
    words already present in the charter — but it is real: the terms are valid PubMed
    syntax and the hit counts it reports are the ones the searches would return.
    """
    anchors = [text for text in (charter.outcome, charter.population) if text.strip()]
    if not anchors:
        anchors = keyphrases(charter.task or charter.prompt)[:2]

    base = " AND ".join(tiab(a) for a in anchors)
    queries: list[PlannedQuery] = []

    if base:
        queries.append(
            PlannedQuery(term=base, rationale="the charter's outcome and population")
        )

    for topic in charter.topic_taxonomy:
        facet = or_group(topic.seed_terms[:4]) or tiab(topic.title)
        term = f"{base} AND {facet}" if base else facet
        queries.append(
            PlannedQuery(term=term, rationale=f"seeds the {topic.slug} topic", topic=topic.slug)
        )

    if not charter.topic_taxonomy:
        # No taxonomy to fan out over, so widen on the prompt's own phrases instead.
        for candidate in keyphrases(charter.task or charter.prompt)[2:6]:
            term = f"{base} AND {tiab(candidate)}" if base else tiab(candidate)
            queries.append(
                PlannedQuery(term=term, rationale=f"broadens on {candidate!r} from the prompt")
            )

    seen: dict[str, None] = {}
    unique: list[PlannedQuery] = []
    for query in queries:
        filtered = with_filters(query.term, charter)
        if filtered in seen:
            continue
        seen[filtered] = None
        unique.append(query.model_copy(update={"term": filtered}))
    return QueryPlan(queries=unique[:max_queries])


def gap_plan(
    charter: Charter,
    gaps: Sequence[TopicGap],
    *,
    already: Collection[str] = (),
    max_queries: int = 12,
) -> QueryPlan:
    """Queries for the topics curation could not fill. No model call.

    The judgment has already been made and paid for: `TopicGap.missing` is the curator's
    own account of what its topic lacks, written as search concepts because the prompt
    asks for them that way. Turning that into syntax is code, like every other query
    here.

    Broader than the round that came up short, in a specific way — the population anchor
    is dropped and the curator's phrases are ORed in beside the topic's own seeds. A
    query already in `already` is skipped rather than run twice, so a plan that comes
    back empty is the honest answer that there is nothing new to ask, and the caller can
    stop instead of paying for a round that would return what it already has.
    """
    anchors = (
        [charter.outcome]
        if charter.outcome.strip()
        else keyphrases(charter.task or charter.prompt)[:1]
    )
    base = " AND ".join(tiab(a) for a in anchors if a.strip())

    seen = set(already)
    planned: list[PlannedQuery] = []
    for gap in gaps:
        if gap.shortfall <= 0:
            continue
        topic = charter.topic(gap.topic)
        seeds = list(topic.seed_terms) if topic is not None else []
        facet = or_group([*keyphrases(gap.missing), *seeds][:MAX_GAP_TERMS])
        if not facet:
            continue
        term = with_filters(f"{base} AND {facet}" if base else facet, charter)
        if term in seen:
            continue
        seen.add(term)
        planned.append(
            PlannedQuery(
                term=term,
                rationale=f"refills the {gap.topic} topic, {gap.shortfall} short",
                topic=gap.topic,
            )
        )
    return QueryPlan(queries=planned[:max_queries])


# --- verification -----------------------------------------------------------


def untagged_clauses(term: str) -> list[str]:
    """Tokens in a query that carry no field tag.

    Their presence is what makes `[All Fields]` in a translation *expected*: PubMed's
    automatic term mapping expands any untagged word that way. With none of them, an
    `[All Fields]` is unexplained, and that is the whole basis of the check below.
    """
    remainder = _TAGGED.sub(" ", term).replace("(", " ").replace(")", " ")
    return [token for token in remainder.split() if token.upper() not in _BOOLEAN]


def inspect_translation(term: str, result: ESearchResult) -> tuple[bool, str]:
    """(suspect, note) for one executed query.

    Advisory, never fatal: a suspect query still returned papers, and the run is better
    off ranking them with a warning attached than refusing to continue. The note is
    written to the run manifest and printed at the retrieve pause, which is the moment
    a human can still act on it.
    """
    if result.fields_not_found:
        tags = ", ".join(result.fields_not_found)
        return True, f"PubMed did not recognize field tag(s): {tags}"

    if untagged_clauses(term):
        # Automatic term mapping explains any [All Fields] here, so there is nothing
        # to conclude from the translation.
        return False, ""

    if ALL_FIELDS in result.query_translation:
        return True, (
            "every clause was field-tagged, but PubMed translated the query to "
            f"{ALL_FIELDS} — a tag was almost certainly rejected and rewritten, "
            f"which is why this returned {result.count:,} hits"
        )
    return False, ""


def executed(term: str, result: ESearchResult, *, retrieved: int) -> ExecutedQuery:
    """Record one search, translation and verdict included."""
    suspect, note = inspect_translation(term, result)
    return ExecutedQuery(
        term=term,
        translation=result.query_translation,
        count=result.count,
        retrieved=retrieved,
        fields_not_found=list(result.fields_not_found),
        suspect=suspect,
        note=note,
    )
