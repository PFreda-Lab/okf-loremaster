"""The topic bounds, as properties rather than examples.

This is the build step 5 gate. `enforce_bounds` reconciles three bounds that genuinely
conflict — a per-topic ceiling, a per-topic floor, and a bundle-wide target — and the
interesting cases are the ones where they cannot all hold at once. Enumerating those by
hand means enumerating the ones somebody thought of, so the bounds are stated as
invariants instead and checked against a few hundred generated curations: topics that
overflow, topics that are empty, curators that claim the same paper twice or name a
topic the charter never had, targets below the floors and above the ceilings.

The one bound allowed to fail is `target_papers`, and P3 is written to say exactly when:
over target is only permitted when no topic could give up a paper without falling under
its floor, and only with a warning saying so. That asymmetry is the design — a bundle
slightly over its target still reads, one with a two-paper topic in it reads as broken —
so it is asserted rather than left to the docstring.

`random` is seeded per case: a property that fails needs to fail the same way twice.
"""

from __future__ import annotations

import random

import pytest

from okf_loremaster.curation import Placement, enforce_bounds
from okf_loremaster.schemas import Charter, Topic

SEEDS = range(120)


# --- building a case --------------------------------------------------------


def charter_with(
    slugs: list[str], *, topic_min: int = 2, topic_max: int = 5, target: int = 100
) -> Charter:
    return Charter(
        prompt="a request",
        topic_taxonomy=[Topic(slug=slug, title=slug.title()) for slug in slugs],
        target_papers=target,
        topic_min=topic_min,
        topic_max=topic_max,
    )


def ranked(pmids: list[str]) -> dict[str, int]:
    return {pmid: index for index, pmid in enumerate(pmids)}


class Case:
    """One generated curation, and the charter it has to fit inside."""

    def __init__(self, rng: random.Random) -> None:
        count = rng.randint(1, 6)
        topic_min = rng.randint(1, 6)
        topic_max = topic_min + rng.randint(0, 8)
        self.slugs = [f"topic-{index}" for index in range(count)]
        self.charter = charter_with(
            self.slugs,
            topic_min=topic_min,
            topic_max=topic_max,
            # Spans every relationship the three bounds can have: under the taxonomy's
            # floor, inside its range, and past what it could hold.
            target=rng.randint(1, count * topic_max + 4),
        )

        universe = [str(11000 + n) for n in range(60)]
        # Topics sample independently, so the same paper landing on two of them is
        # ordinary rather than a special case somebody has to remember to generate.
        self.kept = {
            slug: rng.sample(universe, rng.randint(0, min(len(universe), topic_max + 4)))
            for slug in self.slugs
        }
        self.reserve = {slug: rng.sample(universe, rng.randint(0, 10)) for slug in self.slugs}
        if rng.random() < 0.25:
            # A curator naming a topic the charter does not have. Its papers are dropped,
            # which is only interesting if something generates the case.
            self.kept["not-a-topic"] = rng.sample(universe, rng.randint(1, 5))

        order = universe[:]
        rng.shuffle(order)
        # A few papers left out of the ranking entirely: a rank arrives from the pool and
        # a backfilled paper need not have been in it.
        self.rank = ranked(order[: len(order) - rng.randint(0, 5)])
        self.missing = {slug: f"more about {slug}" for slug in self.slugs}

    def run(self) -> Placement:
        return enforce_bounds(
            self.charter,
            kept=self.kept,
            rank=self.rank,
            reserve=self.reserve,
            missing=self.missing,
        )


# --- the invariants ---------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_p1_no_topic_exceeds_its_ceiling(seed: int) -> None:
    case = Case(random.Random(seed))
    placement = case.run()
    for slug, pmids in placement.topics.items():
        assert len(pmids) <= case.charter.topic_max, slug


@pytest.mark.parametrize("seed", SEEDS)
def test_p2_a_topic_under_its_floor_is_reported_as_a_gap(seed: int) -> None:
    """The floor is a claim, not a guarantee: what it guarantees is an account of itself."""
    case = Case(random.Random(seed))
    placement = case.run()
    gapped = {gap.topic for gap in placement.gaps}

    for slug, pmids in placement.topics.items():
        assert (len(pmids) >= case.charter.topic_min) is (slug not in gapped), slug

    for gap in placement.gaps:
        assert gap.floor == case.charter.topic_min
        assert gap.kept == len(placement.topics[gap.topic])
        assert gap.shortfall > 0
        # The curator's own words survive to the re-query round, which is the only
        # thing that makes a second search different from the first.
        assert gap.missing == case.missing[gap.topic]


@pytest.mark.parametrize("seed", SEEDS)
def test_p3_the_target_is_met_or_the_floors_are_the_reason(seed: int) -> None:
    case = Case(random.Random(seed))
    placement = case.run()
    if placement.total <= case.charter.target_papers:
        return

    # Over target is only allowed when every topic is already at or under its floor —
    # there is nothing left to trim that would not break a floor.
    assert all(len(pmids) <= case.charter.topic_min for pmids in placement.topics.values())
    assert any("not worth having" in warning for warning in placement.warnings)


@pytest.mark.parametrize("seed", SEEDS)
def test_p4_no_paper_is_on_two_topics(seed: int) -> None:
    case = Case(random.Random(seed))
    placement = case.run()
    placed = [pmid for pmids in placement.topics.values() for pmid in pmids]
    assert len(placed) == len(set(placed))


@pytest.mark.parametrize("seed", SEEDS)
def test_p5_every_placed_paper_was_offered_to_the_topic_holding_it(seed: int) -> None:
    """Nothing is invented, and nothing migrates between topics on the way through."""
    case = Case(random.Random(seed))
    placement = case.run()
    for slug, pmids in placement.topics.items():
        available = set(case.kept.get(slug, ())) | set(case.reserve.get(slug, ()))
        assert set(pmids) <= available, slug


@pytest.mark.parametrize("seed", SEEDS)
def test_p6_the_same_curation_places_the_same_papers_in_the_same_order(seed: int) -> None:
    case = Case(random.Random(seed))
    first, second = case.run(), case.run()
    assert list(first.topics.items()) == list(second.topics.items())
    assert first.gaps == second.gaps
    assert first.warnings == second.warnings


@pytest.mark.parametrize("seed", SEEDS)
def test_p7_the_counts_account_for_every_paper(seed: int) -> None:
    """`trimmed`, `backfilled` and `duplicates` add up, so the summary is a measurement."""
    case = Case(random.Random(seed))
    placement = case.run()
    offered = sum(len(case.kept.get(slug, ())) for slug in case.charter.slugs)
    assert placement.total == offered - placement.duplicates - placement.trimmed + (
        placement.backfilled
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_p8_every_topic_is_listed_and_every_topic_is_ordered(seed: int) -> None:
    """An empty topic is a finding about the search; dropping it would hide one."""
    case = Case(random.Random(seed))
    placement = case.run()
    assert list(placement.topics) == list(case.charter.slugs)

    fallback = len(case.rank) + 1
    for pmids in placement.topics.values():
        keys = [(case.rank.get(pmid, fallback), pmid) for pmid in pmids]
        assert keys == sorted(keys)


# --- the passes, one at a time ----------------------------------------------


def test_the_ceiling_trims_the_worst_ranked_first() -> None:
    charter = charter_with(["aa"], topic_min=1, topic_max=2)
    placement = enforce_bounds(
        charter, kept={"aa": ["3", "1", "2"]}, rank=ranked(["1", "2", "3"])
    )
    assert placement.topics == {"aa": ["1", "2"]}
    assert placement.trimmed == 1


def test_the_floor_backfills_in_the_order_the_reserve_was_given() -> None:
    """Not in rank order. The caller decided what "nearest miss" means, and it knows
    things the rank does not — whether the screener included it, whether the curator
    turned it down."""
    charter = charter_with(["aa"], topic_min=2, topic_max=8)
    placement = enforce_bounds(
        charter,
        kept={"aa": ["1"]},
        # "2" ranks better than "9" and is still not the one taken.
        reserve={"aa": ["9", "2"]},
        rank=ranked(["1", "2", "9"]),
    )
    assert placement.topics == {"aa": ["1", "9"]}
    assert placement.backfilled == 1


def test_a_backfill_never_takes_a_paper_another_topic_already_holds() -> None:
    charter = charter_with(["aa", "bb"], topic_min=2, topic_max=8)
    placement = enforce_bounds(
        charter,
        kept={"aa": ["1", "2"], "bb": ["3"]},
        reserve={"bb": ["1", "4"]},  # "1" is on aa
        rank=ranked(["1", "2", "3", "4"]),
    )
    assert placement.topics == {"aa": ["1", "2"], "bb": ["3", "4"]}


def test_a_paper_two_topics_wanted_stays_on_the_first_in_the_charters_order() -> None:
    charter = charter_with(["aa", "bb"], topic_min=1, topic_max=8)
    placement = enforce_bounds(
        charter,
        # bb asks first in the mapping's order, and the charter's order wins anyway.
        kept={"bb": ["1", "5"], "aa": ["1", "4"]},
        rank=ranked(["1", "4", "5"]),
    )
    assert placement.topics == {"aa": ["1", "4"], "bb": ["5"]}
    assert placement.duplicates == 1
    assert any("more than one topic" in warning for warning in placement.warnings)


def test_papers_kept_on_a_topic_the_charter_does_not_have_are_dropped_and_named() -> None:
    charter = charter_with(["aa"], topic_min=1, topic_max=8)
    placement = enforce_bounds(
        charter, kept={"aa": ["1"], "invented": ["2"]}, rank=ranked(["1", "2"])
    )
    assert placement.topics == {"aa": ["1"]}
    assert any("invented" in warning for warning in placement.warnings)


def test_every_charter_topic_appears_even_when_nothing_was_kept() -> None:
    charter = charter_with(["aa", "bb"], topic_min=1, topic_max=8)
    placement = enforce_bounds(charter, kept={}, rank={})
    assert placement.topics == {"aa": [], "bb": []}
    assert [gap.topic for gap in placement.gaps] == ["aa", "bb"]
    assert placement.filled == 0


def test_the_target_trims_the_widest_topic_and_breaks_ties_the_same_way_every_run() -> None:
    """Three equal topics, two papers over target. Which two go is not arbitrary."""
    charter = charter_with(["aa", "bb", "cc"], topic_min=1, topic_max=10, target=10)
    placement = enforce_bounds(
        charter,
        kept={
            "aa": ["1", "2", "3", "4"],
            "bb": ["5", "6", "7", "8"],
            "cc": ["9", "10", "11", "12"],
        },
        rank=ranked([str(n) for n in range(1, 13)]),
    )
    # Widest first, and on a tie the later slug — so cc gives one up, then bb.
    assert placement.topics == {
        "aa": ["1", "2", "3", "4"],
        "bb": ["5", "6", "7"],
        "cc": ["9", "10", "11"],
    }
    assert placement.total == charter.target_papers
    assert placement.trimmed == 2
    assert not placement.warnings


def test_the_target_gives_way_to_the_floors_and_says_it_did() -> None:
    """The one bound allowed to fail, failing out loud."""
    charter = charter_with(["aa", "bb"], topic_min=4, topic_max=8, target=5)
    placement = enforce_bounds(
        charter,
        kept={"aa": ["1", "2", "3", "4"], "bb": ["5", "6", "7", "8"]},
        rank=ranked([str(n) for n in range(1, 9)]),
    )
    assert placement.total == 8
    assert placement.trimmed == 0
    assert any("target of 5" in warning for warning in placement.warnings)
    assert not placement.gaps  # both topics are exactly at their floor


def test_the_target_trim_cannot_create_a_gap() -> None:
    """Gaps are computed before the trim, which is only honest because the trim stops at
    the floor. If it did not, a topic could end up short with no gap reported for it."""
    charter = charter_with(["aa", "bb"], topic_min=3, topic_max=8, target=6)
    placement = enforce_bounds(
        charter,
        kept={"aa": [str(n) for n in range(1, 8)], "bb": ["8", "9", "10"]},
        rank=ranked([str(n) for n in range(1, 11)]),
    )
    assert [len(pmids) for pmids in placement.topics.values()] == [3, 3]
    assert not placement.gaps


def test_a_paper_missing_from_the_ranking_sorts_last_rather_than_raising() -> None:
    charter = charter_with(["aa"], topic_min=1, topic_max=8)
    placement = enforce_bounds(charter, kept={"aa": ["9", "1"]}, rank=ranked(["1"]))
    assert placement.topics == {"aa": ["1", "9"]}


def test_the_summary_reads_as_a_sentence() -> None:
    charter = charter_with(["aa", "bb"], topic_min=2, topic_max=8)
    placement = enforce_bounds(
        charter,
        kept={"aa": ["1", "2"]},
        reserve={"bb": ["3"]},
        rank=ranked(["1", "2", "3"]),
    )
    assert placement.summary() == "3 papers across 2 of 2 topics, 1 backfilled, 1 under floor"
