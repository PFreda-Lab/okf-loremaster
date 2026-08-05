"""`screen -> curate`, and the conditional re-query edge that closes the loop.

Two halves. The first drives the two nodes directly, on the paths where something has
gone wrong — a budget reached, a call that failed, a curator that answered about a paper
nobody offered or said nothing about one that was. Those are the paths a happy-path run
never reaches and a real one reaches constantly.

The second runs the whole graph twice over. `fake_ncbi` withholds a slice of its corpus
behind a phrase no first-round query contains, and the only route to that phrase is a
curator saying its topic lacks that topic — so a second round that found the withheld
papers is proof the edge asked something new, not proof that it ran. A re-query edge
that quietly re-ran the first round's searches would come back with the corpus it
already had and fail here.

The model is `ScriptedLLM`, which reads the real prompts. What the nodes are asserted to
have asked is therefore what they actually sent.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from okf_loremaster.graph.nodes import curate_node, screen_node
from okf_loremaster.graph.state import RunState
from okf_loremaster.prompts import CURATE_SYSTEM, SCREEN_SYSTEM
from okf_loremaster.schemas import Candidate, Charter, ScoredCandidate, ScreenVerdict, Topic

from fake_llm import ScriptedLLM, curation, verdict
from fake_ncbi import RESCUE_COUNT, TOPICS, UNLOCK_PHRASE, rescue_pmids
from graph_runs import (
    DELTA_INCLUDED,
    PROMPT,
    TARGET,
    TOPIC_MAX,
    TOPIC_MIN,
    UNIQUE_FIRST_ROUND,
    full_run,
    node_deps,
    scripted_run,
)

# --- driving one node at a time ---------------------------------------------


def paper(n: int) -> Candidate:
    return Candidate(pmid=f"9{n:04d}", title=f"Paper {n}", abstract=f"Abstract number {n}.")


def pool_of(count: int) -> list[ScoredCandidate]:
    return [
        ScoredCandidate(candidate=paper(n), score=1.0 - n / 100, position=n) for n in range(count)
    ]


def two_topics() -> Charter:
    return Charter(
        prompt=PROMPT,
        task=PROMPT,
        topic_taxonomy=[Topic(slug="aa", title="Aa"), Topic(slug="bb", title="Bb")],
        topic_paper_min=1,
        topic_paper_max=8,
        target_papers=50,
    )


def screening(**fields: Any) -> Callable[[str], dict[str, Any]]:
    """A screener that answers the same way about every paper."""
    return lambda _paper: verdict(**fields)


# --- screen -----------------------------------------------------------------


async def test_the_screen_budget_stops_the_node_and_says_what_it_left(
    settings_factory: Any, tmp_path: Path
) -> None:
    scripted = ScriptedLLM(screen=screening(include=True, relevance=3), curate=lambda *_: {})
    state: RunState = {"charter": two_topics(), "pool": pool_of(5)}

    async with node_deps(settings_factory, tmp_path, scripted=scripted, screen_budget=3) as deps:
        update = await screen_node(state, deps)

    assert len(scripted.screened) == 3
    assert len(update["verdicts"]) == 3
    assert any("screen budget of 3 reached" in note for note in update["warnings"])
    assert any("2 pooled paper(s) were not screened" in note for note in update["warnings"])


async def test_a_second_round_screens_only_what_it_has_not_already_paid_for(
    settings_factory: Any, tmp_path: Path
) -> None:
    scripted = ScriptedLLM(screen=screening(include=True, relevance=3), curate=lambda *_: {})
    state: RunState = {
        "charter": two_topics(),
        "pool": pool_of(5),
        "verdicts": [ScreenVerdict(pmid=paper(0).pmid, include=True, relevance=3)],
    }

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await screen_node(state, deps)

    assert len(scripted.screened) == 4
    assert len(update["verdicts"]) == 5  # the carried one, plus the four new


async def test_a_screening_call_that_cannot_be_read_excludes_the_paper_and_reports_once(
    settings_factory: Any, tmp_path: Path
) -> None:
    """One warning for the batch, not one per paper — and a verdict that keeps the paper
    out of the floor backfill, since a call that failed is not evidence about it."""
    scripted = ScriptedLLM(screen=lambda _paper: {}, curate=lambda *_: {})
    state: RunState = {"charter": two_topics(), "pool": pool_of(4)}

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await screen_node(state, deps)

    verdicts = update["verdicts"]
    assert len(verdicts) == 4
    assert all(not v.include and v.relevance == 0 and not v.borderline for v in verdicts)
    failed = [note for note in update["warnings"] if "screening call(s) failed" in note]
    assert len(failed) == 1
    assert "4 of 4" in failed[0]
    assert any("not a judgment about the literature" in note for note in update["warnings"])


async def test_a_topic_the_charter_does_not_have_is_blanked_rather_than_carried(
    settings_factory: Any, tmp_path: Path
) -> None:
    scripted = ScriptedLLM(
        screen=screening(include=True, relevance=3, topic="invented"), curate=lambda *_: {}
    )
    state: RunState = {"charter": two_topics(), "pool": pool_of(2)}

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await screen_node(state, deps)

    assert all(v.topic == "" for v in update["verdicts"])
    assert any("invented" in note for note in update["warnings"])


async def test_a_topic_named_with_different_punctuation_is_recovered_not_discarded(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The charter has this topic. Only the typography differs, and that is not a reason
    to throw away a judgment already paid for — a run blanked 113 of 252 hints this way.
    """
    scripted = ScriptedLLM(
        screen=screening(include=True, relevance=3, topic="  AA  "), curate=lambda *_: {}
    )
    state: RunState = {"charter": two_topics(), "pool": pool_of(2)}

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await screen_node(state, deps)

    assert all(v.topic == "aa" for v in update["verdicts"]), "folded back to the real slug"
    assert not any("does not have" in note for note in update["warnings"])


async def test_with_no_model_nothing_is_screened_and_the_run_says_so(
    settings_factory: Any, tmp_path: Path
) -> None:
    state: RunState = {"charter": two_topics(), "pool": pool_of(3)}

    async with node_deps(settings_factory, tmp_path) as deps:
        update = await screen_node(state, deps)

    assert update["verdicts"] == []
    assert any("nothing was screened" in note for note in update["warnings"])


# --- curate -----------------------------------------------------------------


def curate_state(count: int, charter: Charter, **overrides: Any) -> RunState:
    """A pool the screener has already included, all of it pointed at one topic."""
    pool = pool_of(count)
    state: RunState = {
        "charter": charter,
        "pool": pool,
        "unique": [item.candidate for item in pool],
        "verdicts": [
            ScreenVerdict(pmid=item.pmid, include=True, relevance=3, topic="aa", reason="on point")
            for item in pool
        ],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


async def test_the_curator_is_asked_about_the_topic_by_name_with_the_screeners_notes(
    settings_factory: Any, tmp_path: Path
) -> None:
    scripted = ScriptedLLM(
        screen=screening(include=True, relevance=3),
        curate=lambda _slug, offered: curation(dict.fromkeys(offered, True)),
    )
    charter = two_topics()

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await curate_node(curate_state(3, charter), deps)

    # Only the topic with papers on it costs a call; `bb` was offered nothing.
    assert scripted.curated == ["aa"]
    assert scripted.offers["aa"] == [[paper(n).pmid for n in range(3)]]
    assert update["topics"] == {"aa": [paper(n).pmid for n in range(3)], "bb": []}


async def test_a_pmid_the_curator_invented_is_dropped_rather_than_placed(
    settings_factory: Any, tmp_path: Path
) -> None:
    def curate(_slug: str, offered: list[str]) -> dict[str, Any]:
        return curation({**dict.fromkeys(offered, True), "99999999": True})

    scripted = ScriptedLLM(screen=screening(include=True, relevance=3), curate=curate)

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await curate_node(curate_state(3, two_topics()), deps)

    assert "99999999" not in update["topics"]["aa"]
    assert len(update["topics"]["aa"]) == 3


async def test_silence_is_not_consent(settings_factory: Any, tmp_path: Path) -> None:
    """A paper the curator said nothing about is not kept."""

    def curate(_slug: str, offered: list[str]) -> dict[str, Any]:
        return curation({offered[0]: True})

    scripted = ScriptedLLM(screen=screening(include=True, relevance=3), curate=curate)

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await curate_node(curate_state(3, two_topics()), deps)

    assert update["topics"]["aa"] == [paper(0).pmid]


async def test_but_a_topic_under_its_floor_can_still_reach_an_unanswered_paper(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The reserve is what makes silence recoverable without a second search round."""

    def curate(_slug: str, offered: list[str]) -> dict[str, Any]:
        return curation({offered[0]: True})

    scripted = ScriptedLLM(screen=screening(include=True, relevance=3), curate=curate)
    charter = two_topics().model_copy(update={"topic_paper_min": 3})

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await curate_node(curate_state(3, charter), deps)

    assert len(update["topics"]["aa"]) == 3
    backfilled = [d for d in update["curation"].kept if d.rationale == "the curator did not answer"]
    assert len(backfilled) == 2
    # `bb` was offered nothing at all, so it is short and has no reserve to fix it.
    assert [gap.topic for gap in update["curation"].gaps] == ["bb"]


async def test_a_topic_whose_call_failed_keeps_the_screeners_best_ranked(
    settings_factory: Any, tmp_path: Path
) -> None:
    """An empty topic would be reported as a search failure it is not."""
    scripted = ScriptedLLM(
        screen=screening(include=True, relevance=3),
        curate=lambda *_: {"decisions": "not a list of decisions"},
    )

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await curate_node(curate_state(3, two_topics()), deps)

    assert update["topics"]["aa"] == [paper(n).pmid for n in range(3)]
    assert any("curation of the aa topic failed" in note for note in update["warnings"])


async def test_with_no_model_curation_falls_back_to_the_screeners_order(
    settings_factory: Any, tmp_path: Path
) -> None:
    async with node_deps(settings_factory, tmp_path) as deps:
        update = await curate_node(curate_state(3, two_topics()), deps)

    assert update["topics"]["aa"] == [paper(n).pmid for n in range(3)]
    assert any("no model is available" in note for note in update["warnings"])


# --- the whole graph, twice round -------------------------------------------
#
# The runner, the charter and the scripted policy are in `graph_runs.py`: extraction and
# verification drive the same run and ask different questions of it.


async def test_a_thin_topic_sends_the_run_back_to_search_and_the_second_round_fills_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 5 gate, end to end.

    `delta` comes back from the first round at 2 against a floor of 4, and the papers
    that fix it are ones no first-round query could return. That the topic ends up made
    entirely of them is the proof that the second round asked a new question.
    """
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    assert run.state["rounds"] == 2
    assert UNLOCK_PHRASE in run.fake.esearch_terms[-1]
    assert not run.gaps

    delta = run.topics["delta"]
    assert len(delta) >= TOPIC_MIN
    assert set(delta) <= set(rescue_pmids())


async def test_the_second_round_re_curates_only_the_topic_that_came_up_short(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A topic that was already fine keeps its first-round decisions and is not paid for
    twice."""
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    assert {topic: run.calls_for(topic) for topic in TOPICS} == {
        "alpha": 1,
        "beta": 1,
        "gamma": 1,
        "delta": 2,
    }


async def test_no_paper_is_screened_twice_across_the_two_rounds(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    screened = run.scripted.screened
    assert len(screened) == UNIQUE_FIRST_ROUND + RESCUE_COUNT
    assert len(set(screened)) == len(screened)


async def test_the_screen_budget_is_global_across_rounds(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retrieve pause approved a number of papers to spend on. A second round that
    quietly doubled it would make that approval mean nothing."""
    budget = UNIQUE_FIRST_ROUND + 4
    run = await full_run(settings_factory, tmp_path, monkeypatch, screen_budget=budget)

    assert run.state["rounds"] == 2
    assert len(run.scripted.screened) == budget
    assert any("screen budget" in note for note in run.state["warnings"])


async def test_a_spent_budget_stops_the_run_rather_than_paying_for_a_round_that_cannot_help(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Because the budget is global, a spent one makes another round pointless in advance.

    A real run did this: round two planned queries, retrieved 232 new records, ranked
    150 of them, screened zero, and re-curated the identical kept set. Nothing about the
    gap can change while the screener — the only thing that turns a candidate into a
    paper — is out of budget, so the round is refused before it is paid for. The gap is
    still reported, and the warning names the flag that would actually fix it, so the
    reason ends up in `log.md` rather than only in the routing.
    """
    run = await full_run(
        settings_factory, tmp_path, monkeypatch, screen_budget=UNIQUE_FIRST_ROUND
    )

    assert run.state["rounds"] == 1
    assert len(run.scripted.screened) == UNIQUE_FIRST_ROUND
    assert run.gaps == ["delta"]
    assert not any(UNLOCK_PHRASE in term for term in run.fake.esearch_terms)
    assert any("--screen-budget" in note for note in run.state["warnings"])


async def test_a_budget_with_room_left_still_goes_back_for_another_round(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard has to mean exactly "spent", not "nearly spent" — one paper of headroom
    is enough to be worth a round, and the run must not talk itself out of one."""
    run = await full_run(
        settings_factory, tmp_path, monkeypatch, screen_budget=UNIQUE_FIRST_ROUND + 1
    )

    assert run.state["rounds"] == 2
    assert len(run.scripted.screened) == UNIQUE_FIRST_ROUND + 1
    assert any(UNLOCK_PHRASE in term for term in run.fake.esearch_terms)


async def test_the_bounds_hold_on_the_finished_run(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    topics = run.topics
    assert list(topics) == list(TOPICS)
    assert all(TOPIC_MIN <= len(pmids) <= TOPIC_MAX for pmids in topics.values())
    assert sum(len(pmids) for pmids in topics.values()) == TARGET

    placed = [pmid for pmids in topics.values() for pmid in pmids]
    assert len(placed) == len(set(placed))


async def test_max_rounds_one_turns_the_edge_off_and_leaves_the_gap_reported(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is a bound, and a run that stops short says what it could not fill
    rather than presenting a thin topic as a finished one."""
    run = await full_run(settings_factory, tmp_path, monkeypatch, max_rounds=1)

    assert run.state["rounds"] == 1
    assert run.gaps == ["delta"]
    assert len(run.topics["delta"]) == DELTA_INCLUDED
    assert not any(UNLOCK_PHRASE in term for term in run.fake.esearch_terms)
    assert len(run.scripted.screened) == UNIQUE_FIRST_ROUND

    curation_result = run.state["curation"]
    assert curation_result is not None
    assert [gap.missing for gap in curation_result.gaps] == [UNLOCK_PHRASE]


async def test_a_gap_the_searches_cannot_close_ends_the_run_instead_of_asking_again(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A curator with nothing new to ask for produces no new query, and the run stops.

    The bound that matters most: `max_rounds` caps the cost, but this is what keeps a
    run from paying full price to retrieve the corpus it already has.
    """
    scripted = scripted_run()

    # The curator asks for the topic's own seed term back — which is exactly the query
    # the first round already ran.
    original = scripted.curate

    def curate(slug: str, offered: list[str]) -> dict[str, Any]:
        answer = original(slug, offered)
        if slug == "delta":
            answer["missing"] = "delta"
        return answer

    scripted.curate = curate
    run = await full_run(settings_factory, tmp_path, monkeypatch, scripted=scripted)

    assert run.state["rounds"] == 1
    assert len(run.scripted.screened) == UNIQUE_FIRST_ROUND
    assert run.gaps == ["delta"]


# --- what the scripted model is asserting on --------------------------------


def test_the_scripted_model_fails_loudly_if_a_prompt_stops_naming_its_subject() -> None:
    """The fake is only worth something if it would notice `prompts.py` changing shape."""
    scripted = ScriptedLLM(screen=screening(include=True, relevance=1), curate=lambda *_: {})

    with pytest.raises(AssertionError, match="Paper:"):
        scripted._reply(
            {
                "messages": [
                    {"role": "system", "content": SCREEN_SYSTEM},
                    {"role": "user", "content": "a review question, and no paper at all"},
                ]
            }
        )

    with pytest.raises(AssertionError, match="names the topic"):
        scripted._reply(
            {
                "messages": [
                    {"role": "system", "content": CURATE_SYSTEM},
                    {"role": "user", "content": "  - 123 [relevance 3] a title, on no topic"},
                ]
            }
        )


async def test_a_paper_the_screener_lists_several_topics_for_keeps_the_first(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The field holds one topic and the screener sometimes names four.

    Seen live: `'baseline-immunovirologic-status; regimen-and-pharmacology; ...'` in a
    single-topic field. Blanking that discards a real judgment over a delimiter, and
    the first topic named is better information for the curator than none.
    """
    scripted = ScriptedLLM(
        screen=screening(include=True, relevance=3, topic="bb; aa"), curate=lambda *_: {}
    )
    state: RunState = {"charter": two_topics(), "pool": pool_of(2)}

    async with node_deps(settings_factory, tmp_path, scripted=scripted) as deps:
        update = await screen_node(state, deps)

    assert all(v.topic == "bb" for v in update["verdicts"])
    assert not any("does not have" in note for note in update["warnings"])
