"""Query construction, and the check on what PubMed did with it.

The verification tests replay the recorded fixture rather than a hand-written one. The
behavior being defended against — a rejected field tag rewritten to `[All Fields]`, with
an empty `errorlist` and two orders of magnitude more hits — is surprising enough that a
synthetic example would only prove the test author believed it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from okf_loremaster import queries
from okf_loremaster.clients import Clients
from okf_loremaster.schemas import Charter, Topic

CASSETTE = Path(__file__).parent / "fixtures" / "ncbi.jsonl"

# The two terms the fixture recorder searched for. Kept in step with
# `scripts/record_fixtures.py`; a replay of anything else is not in the cassette.
CLEAN_TERM = "postoperative respiratory failure[tiab] AND risk factors[tiab]"
BAD_FIELD_TERM = "postoperative respiratory failure[nosuchfield]"
# `retmax` is part of the URL and therefore part of the cassette key: these are the
# values the recorder used, not free choices.
CLEAN_RETMAX = 10
BAD_FIELD_RETMAX = 5


def charter(**overrides: object) -> Charter:
    base: dict[str, object] = {
        "prompt": "predict a measured outcome after a procedure in adults",
        "task": "predict a measured outcome after a procedure in adults",
        "population": "adults",
        "outcome": "measured outcome",
        "topic_taxonomy": [
            Topic(slug="first", title="First", scope="s", seed_terms=["alpha", "alpha timing"]),
            Topic(slug="second", title="Second", scope="s", seed_terms=["beta"]),
        ],
    }
    return Charter.model_validate(base | overrides)


# --- construction -----------------------------------------------------------


def test_a_phrase_is_quoted_so_it_is_not_read_as_a_boolean() -> None:
    assert queries.tiab("risk factors") == '"risk factors"[tiab]'


def test_or_group_of_one_needs_no_parentheses() -> None:
    assert queries.or_group(["alpha"]) == '"alpha"[tiab]'
    assert queries.or_group(["alpha", "beta"]) == '("alpha"[tiab] OR "beta"[tiab])'


def test_or_group_of_nothing_is_empty_rather_than_broken_syntax() -> None:
    assert queries.or_group([]) == ""
    assert queries.or_group(["", "   "]) == ""


def test_filters_are_appended_once_per_query_not_per_clause() -> None:
    term = queries.with_filters('"alpha"[tiab]', charter(languages=["eng"], min_year=2015))
    assert term == '"alpha"[tiab] AND eng[la] AND 2015:3000[dp]'


def test_multiple_languages_are_parenthesized() -> None:
    term = queries.with_filters('"alpha"[tiab]', charter(languages=["eng", "fre"]))
    assert term == '"alpha"[tiab] AND (eng[la] OR fre[la])'


def test_keyphrases_drop_task_words_and_keep_the_subject() -> None:
    found = queries.keyphrases("build a model that predicts postoperative delirium in adults")
    assert "postoperative delirium" in found
    assert not any("model" in phrase for phrase in found)


def test_a_keyphrase_is_capped_so_the_phrase_search_can_match() -> None:
    long_run = "one two three four five six seven"
    assert all(len(p.split()) <= queries.MAX_PHRASE_WORDS for p in queries.keyphrases(long_run))


def test_a_short_anchor_stays_an_exact_phrase() -> None:
    """Precision is the whole point of a phrase search, where a phrase is what a paper
    would actually print."""
    assert queries.anchor("viral suppression") == '"viral suppression"[tiab]'


def test_a_long_anchor_asks_for_the_words_and_not_their_order() -> None:
    """A charter writes `population` and `outcome` for a person to read, and an exact
    phrase search for a description matches nothing at all. Measured against PubMed on
    2026-08-04: the phrase below returned 0 and the ANDed words returned 1,881.

    An anchor is ANDed into every query in a plan, so its over-precision does not cost
    one query, it costs the run. This one did: nine queries, zero hits, no papers
    retrieved, and a bundle emitted over eight empty topics that called itself valid.
    """
    built = queries.anchor("patients initiating antiretroviral therapy")

    assert built.startswith("(") and " AND " in built
    assert '"patients initiating antiretroviral therapy"' not in built
    for word in ("patients", "initiating", "antiretroviral", "therapy"):
        assert f"{word}[tiab]" in built, "every word the charter chose is still required"


def test_an_anchor_of_nothing_but_stopwords_is_empty_rather_than_broken_syntax() -> None:
    assert queries.anchor("the and of") == ""


def test_a_plan_never_anchors_on_a_phrase_that_cannot_match() -> None:
    """The regression that matters, at the level the run actually failed."""
    plan = queries.deterministic_plan(
        charter(
            population="patients initiating antiretroviral therapy", outcome="viral suppression"
        )
    )

    assert plan.queries
    for query in plan.queries:
        assert '"patients initiating antiretroviral therapy"' not in query.term


# --- the deterministic plan -------------------------------------------------


def test_deterministic_plan_covers_every_topic() -> None:
    plan = queries.deterministic_plan(charter())
    assert {q.topic for q in plan.queries} >= {"first", "second"}


def test_deterministic_plan_anchors_on_outcome_and_population() -> None:
    plan = queries.deterministic_plan(charter())
    assert plan.queries[0].term.startswith('("measured outcome"[tiab] AND "adults"[tiab])')
    assert plan.queries[0].topic == ""


def test_deterministic_plan_falls_back_to_the_prompt_with_no_taxonomy() -> None:
    plan = queries.deterministic_plan(
        Charter(prompt="predict postoperative delirium in older adults", task="")
    )
    assert plan.queries
    assert all("delirium" in q.term for q in plan.queries)


def test_deterministic_plan_respects_its_cap() -> None:
    many = charter(
        topic_taxonomy=[
            Topic(slug=f"s{i}", title=f"S{i}", scope="s", seed_terms=[f"term{i}"])
            for i in range(10)
        ]
    )
    assert len(queries.deterministic_plan(many, max_queries=4).queries) == 4


def test_deterministic_plan_drops_duplicate_terms() -> None:
    twins = charter(
        topic_taxonomy=[
            Topic(slug="one", title="One", scope="s", seed_terms=["same"]),
            Topic(slug="two", title="Two", scope="s", seed_terms=["same"]),
        ]
    )
    terms = [q.term for q in queries.deterministic_plan(twins).queries]
    assert len(terms) == len(set(terms))


# --- verification -----------------------------------------------------------


def test_untagged_clauses_sees_through_tags_and_booleans() -> None:
    assert queries.untagged_clauses('"alpha"[tiab] AND "beta"[tiab]') == []
    assert queries.untagged_clauses('"alpha"[tiab] AND beta') == ["beta"]


@pytest.mark.skipif(not CASSETTE.exists(), reason="fixtures not recorded")
async def test_a_well_formed_query_is_not_flagged(replay_clients: Clients) -> None:
    result = await replay_clients.eutils.esearch(CLEAN_TERM, retmax=CLEAN_RETMAX)
    record = queries.executed(CLEAN_TERM, result, retrieved=len(result.ids))
    assert not record.suspect
    assert record.note == ""


@pytest.mark.skipif(not CASSETTE.exists(), reason="fixtures not recorded")
async def test_a_rejected_field_tag_is_caught_by_the_translation_alone(
    replay_clients: Clients,
) -> None:
    """The case the whole verification path exists for.

    PubMed reports no error, returns far more hits, and the only evidence is that a
    fully tagged query came back translated to `[All Fields]`.
    """
    result = await replay_clients.eutils.esearch(BAD_FIELD_TERM, retmax=BAD_FIELD_RETMAX)
    assert result.fields_not_found == ()  # PubMed says nothing is wrong
    assert queries.ALL_FIELDS in result.query_translation

    clean = await replay_clients.eutils.esearch(CLEAN_TERM, retmax=CLEAN_RETMAX)
    assert result.count > clean.count * 10  # 14,382 against 79

    # The recorder searched the term unquoted, which leaves `postoperative respiratory`
    # untagged — and automatic term mapping expands untagged words to [All Fields] on
    # its own. The check declines to conclude anything, which is correct: the
    # translation is no longer evidence of anything once a word can explain it.
    assert not queries.executed(BAD_FIELD_TERM, result, retrieved=len(result.ids)).suspect

    # What this package actually emits is quoted and wholly tagged, because `tiab()`
    # quotes. Against the same recorded response, nothing is left to explain the
    # [All Fields] and the rewrite is caught.
    emitted = f"{queries.phrase('postoperative respiratory failure')}[nosuchfield]"
    record = queries.executed(emitted, result, retrieved=len(result.ids))
    assert record.suspect
    assert queries.ALL_FIELDS in record.translation
    assert "rejected and rewritten" in record.note
