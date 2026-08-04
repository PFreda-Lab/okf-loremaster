"""What recurs across the corpus, and the four ways gathering it goes silently wrong.

The clustering is the whole risk surface. A merge that should not have happened produces
a file that reads perfectly and claims something no paper said, so most of this module is
about merges that must not occur:

- **A polarity qualifier is not a qualifier.** `short sleep` and `long sleep` are the two
  ends of one axis, and merging them prints a U-shaped relationship as a contradiction.
- **A predictor is grouped with its outcome.** One paper reporting one exposure against
  six outcomes in six directions is six coherent findings; collapsed on the exposure it
  reads as a paper disagreeing with itself.
- **Nothing is summarized away.** Every group carries the row addresses it was built from,
  and every merged surface form is still printed, or the merge is one nobody can dispute.
- **The order is total.** Two runs over the same records have to produce the same file, or
  a bundle rebuilt from a cache shows a diff that means nothing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from okf_loremaster.recurrence import MIN_PAPERS, index_predictors, surface_key
from okf_loremaster.schemas import (
    ConceptRecord,
    Direction,
    Extraction,
    PredictorRow,
    RecurrenceIndex,
)


def row(
    predictor: str,
    outcome: str = "an outcome",
    direction: Direction = Direction.INCREASES,
    *,
    operationalization: str = "",
) -> PredictorRow:
    return PredictorRow(
        predictor=predictor,
        outcome=outcome,
        direction=direction,
        operationalization=operationalization,
    )


def paper(pmid: str, *rows: PredictorRow, domain: str = "some-topic") -> ConceptRecord:
    return ConceptRecord(
        pmid=pmid,
        domain=domain,
        authors=[f"Author{pmid} A"],
        year=2020,
        extraction=Extraction(predictors=list(rows)),
    )


def index(*records: ConceptRecord) -> RecurrenceIndex:
    """Index a handful of papers. The effect cell is the emitter's rule, not this one's."""
    return index_predictors(list(records), effect_of=lambda r: r.effect_raw)


def labels(result: RecurrenceIndex) -> list[str]:
    return [group.predictor for group in result.groups]


# --- normalization ----------------------------------------------------------


def test_word_order_and_plurals_are_not_evidence_of_two_predictors() -> None:
    assert surface_key("Sleep duration") == surface_key("duration of sleep")
    assert surface_key("Physical activity level") == surface_key("physical activity levels")


def test_a_papers_own_abbreviation_does_not_split_its_own_phrase() -> None:
    """`(CSR)` is something the paper defined for itself, not a different construct."""
    assert surface_key("Chronic sleep restriction (CSR)") == surface_key(
        "chronic sleep restriction"
    )
    # Digits and hyphens are part of an initialism, and a capitalized ordinary word is not
    # one — the letter after its first is lowercase.
    assert surface_key("Insulin resistance (HOMA-IR)") == surface_key("insulin resistance")
    assert "mediterranean" in surface_key("Diet quality (Mediterranean adherence)")


def test_a_cutoff_inside_parentheses_is_the_variable_and_must_not_be_dropped() -> None:
    """Papers put two unrelated things in parentheses, and only one of them is noise.

    Discarding the whole span was the first rule here, and it deleted the only word that
    said which end of the axis a row was about: `Sleep duration (short, <=6h/d)` and
    `Sleep duration (>=9h vs 7-9h)` both fell to a bare `{sleep, duration}`, matched
    exactly, and merged in the pass that runs before any polarity guard.
    """
    assert surface_key("Sleep duration (short, <=6h/d)") == surface_key("Short sleep duration")
    assert surface_key("Sleep duration (short, <=6h/d)") != surface_key(
        "Sleep duration (>=9h vs 7-9h)"
    )


def test_an_ending_that_only_looks_plural_survives() -> None:
    """Stripping the `s` from these would produce words that match nothing, including
    themselves."""
    for word in ("status", "analysis", "stress"):
        assert word in surface_key(f"baseline {word}")


# --- what may merge, and what may not ---------------------------------------


def test_a_narrowing_qualifier_merges_into_the_phrase_it_narrows() -> None:
    result = index(
        paper("1", row("Sleep restriction")),
        paper("2", row("Chronic sleep restriction (CSR)")),
    )
    assert labels(result) == ["Sleep restriction"]
    group = result.groups[0]
    assert group.papers == 2
    # Both spellings survive on the page. A merge a reader cannot see is one they cannot
    # argue with, and this one is a lexical guess about a field the reader knows better.
    assert set(group.surface_forms) == {"Sleep restriction", "Chronic sleep restriction (CSR)"}


def test_the_two_ends_of_one_axis_stay_two_predictors() -> None:
    """The merge that would matter most and be wrong. `short` and `long` are the same
    variable pointing opposite ways, and one entry saying `increases (1) · decreases (1)`
    would print a U-shaped relationship as a contradiction."""
    result = index(
        paper("1", row("Short sleep duration", direction=Direction.INCREASES)),
        paper("2", row("Short sleep duration", direction=Direction.INCREASES)),
        paper("3", row("Long sleep duration", direction=Direction.DECREASES)),
        paper("4", row("Long sleep duration", direction=Direction.DECREASES)),
    )
    assert sorted(labels(result)) == ["Long sleep duration", "Short sleep duration"]
    assert not any(group.contested for group in result.groups)


def test_a_polarity_qualifier_blocks_the_merge_it_would_otherwise_pass() -> None:
    """The general case behind the previous test: containment alone is not enough."""
    result = index(
        paper("1", row("Alcohol intake")),
        paper("2", row("Alcohol intake")),
        paper("3", row("Heavy alcohol intake")),
        paper("4", row("Heavy alcohol intake")),
    )
    assert sorted(labels(result)) == ["Alcohol intake", "Heavy alcohol intake"]


def test_the_same_cutoff_written_two_ways_is_one_predictor() -> None:
    """The other half of the parenthetical defect, and the half a reader sees. With the
    qualifier deleted these were not merely mis-merged, they were *both*: a `Sleep
    duration` entry that had swallowed the long-sleep rows sat directly above a separate
    `Short sleep duration` entry reporting the same construct."""
    result = index(
        paper("1", row("Short sleep duration")),
        paper("2", row("Sleep duration (short, <=6h/d)")),
    )
    assert labels(result) == ["Short sleep duration"]
    assert set(result.groups[0].surface_forms) == {
        "Short sleep duration",
        "Sleep duration (short, <=6h/d)",
    }


def test_a_phrase_is_not_absorbed_by_a_short_one_it_merely_contains() -> None:
    """Containment is a weak signal when the containing phrase is small, and every extra
    word is another chance the two stopped being about the same thing. A real corpus put
    all three of these under one two-token heading."""
    result = index(
        paper("1", row("Sleep duration")),
        paper("2", row("Sleep duration")),
        paper("3", row("Apnea duration during REM sleep")),
        paper("4", row("Sleep fragmentation without reduction in sleep duration")),
    )
    assert sorted(labels(result)) == ["Sleep duration"]
    assert result.groups[0].papers == 2
    assert result.once == 2


def test_a_statistic_computed_over_a_variable_is_not_a_narrower_reading_of_it() -> None:
    """`SD of sleep duration` is a different measurement from sleep duration, and the two
    letters saying so are exactly the ones a retrieval tokenizer throws away. It reached
    the corpus as a `Sleep duration` group named for the variability measure."""
    result = index(
        paper("1", row("Sleep duration")),
        paper("2", row("Sleep duration")),
        paper("3", row("SD of sleep duration")),
        paper("4", row("SD of sleep duration")),
    )
    assert sorted(labels(result)) == ["SD of sleep duration", "Sleep duration"]


def test_treating_a_thing_is_not_a_narrower_reading_of_having_it() -> None:
    """A corpus reported pain as increasing an outcome and pain *treatment* as decreasing
    it. One differing token and no polarity word, so they merged and the group carried
    both signs under a heading naming only the exposure."""
    result = index(
        paper("1", row("Postoperative pain", direction=Direction.INCREASES)),
        paper("2", row("Postoperative pain", direction=Direction.INCREASES)),
        paper("3", row("Postoperative pain treatment", direction=Direction.DECREASES)),
        paper("4", row("Postoperative pain treatment", direction=Direction.DECREASES)),
    )
    assert sorted(labels(result)) == ["Postoperative pain", "Postoperative pain treatment"]


def test_an_intervention_word_is_refused_wherever_it_sits_in_the_phrase() -> None:
    """The guard is on the differing token, not on where it appears — a prefix flips the
    claim exactly as far as a suffix does."""
    result = index(
        paper("1", row("Smoking")),
        paper("2", row("Smoking")),
        paper("3", row("Smoking cessation")),
        paper("4", row("Smoking cessation")),
        paper("5", row("Delirium screening")),
        paper("6", row("Delirium screening")),
        paper("7", row("Delirium")),
        paper("8", row("Delirium")),
    )
    assert sorted(labels(result)) == [
        "Delirium",
        "Delirium screening",
        "Smoking",
        "Smoking cessation",
    ]


def test_an_ordinary_qualifier_still_merges() -> None:
    """The new refusals must not turn the second pass off. `preoperative` narrows the
    reading; it does not change what is being measured."""
    result = index(
        paper("1", row("Cognitive impairment")),
        paper("2", row("Cognitive impairment")),
        paper("3", row("Preoperative cognitive impairment")),
    )
    assert labels(result) == ["Cognitive impairment"]


# --- what the heading over a group may claim --------------------------------


def test_a_heading_never_asserts_a_cutoff_the_group_disagrees_on() -> None:
    """Two papers dichotomizing one variable at different points are studying the same
    variable, so merging them is right — but naming the result after one of the cutoffs
    makes the heading false for the other row. Seen twice on real corpora, most recently
    `Comorbidity index score (≥1)` standing over a `≥8` row."""
    result = index(
        paper("1", row("Comorbidity index score (≥1)")),
        paper("2", row("Charlson comorbidity index score ≥8")),
    )
    assert labels(result) == ["Comorbidity index score"]

    group = result.groups[0]
    assert group.surface_forms == [
        "Comorbidity index score (≥1)",
        "Charlson comorbidity index score ≥8",
    ], "the cutoffs are not lost, only moved off the heading"


def test_a_form_the_corpus_wrote_without_a_cutoff_is_preferred_to_stripping_one() -> None:
    """Stripping is the fallback. If any paper named the bare variable, that is the
    heading — it is a phrase somebody actually wrote."""
    result = index(
        paper("1", row("Age ≥70 years")),
        paper("2", row("Age ≥79 years")),
        paper("3", row("Age")),
    )
    assert labels(result) == ["Age"]


def test_a_cutoff_every_paper_agrees_on_stays_in_the_heading() -> None:
    """Nothing to disambiguate, so nothing is taken away. Removing a shared cutoff would
    make the heading vaguer than the corpus."""
    result = index(
        paper("1", row("Sleep duration ≤6h/d")),
        paper("2", row("Sleep duration ≤6h/d")),
    )
    assert labels(result) == ["Sleep duration ≤6h/d"]


def test_a_predictor_that_is_only_a_number_keeps_its_name() -> None:
    """Stripping would leave nothing at all, and a group has to be called something."""
    result = index(
        paper("1", row("ASA ≥3")),
        paper("2", row("ASA 4")),
    )
    assert labels(result) == ["ASA"]


def test_two_phrases_that_merely_share_words_are_not_merged() -> None:
    """Overlap is not containment. These share `intake` and nothing else that matters."""
    result = index(
        paper("1", row("Fruit intake")),
        paper("2", row("Fruit intake")),
        paper("3", row("Sodium intake")),
        paper("4", row("Sodium intake")),
    )
    assert sorted(labels(result)) == ["Fruit intake", "Sodium intake"]


def test_the_heading_is_the_form_the_corpus_mostly_used() -> None:
    result = index(
        paper("1", row("sleep duration")),
        paper("2", row("Sleep duration")),
        paper("3", row("Sleep duration")),
    )
    assert labels(result) == ["Sleep duration"]


# --- predictor x outcome ----------------------------------------------------


def test_one_exposure_against_many_outcomes_is_many_findings_not_a_contradiction() -> None:
    """The case that decided the grouping. Six directions from one paper look like a
    paper arguing with itself right up until the outcome column is read."""
    result = index(
        paper(
            "1",
            row("Short sleep", "Total energy intake", Direction.INCREASES),
            row("Short sleep", "Fruit intake", Direction.DECREASES),
        ),
        paper(
            "2",
            row("Short sleep", "Total energy intake", Direction.INCREASES),
            row("Short sleep", "Fruit intake", Direction.DECREASES),
        ),
    )
    group = result.groups[0]
    # Two outcomes, alphabetical on the tie, and no contradiction anywhere — which is
    # exactly what one group of four rows pointing two ways would have printed.
    assert [outcome.outcome for outcome in group.outcomes] == [
        "Fruit intake",
        "Total energy intake",
    ]
    assert not group.contested


def test_papers_disagreeing_about_the_sign_of_one_relationship_are_marked() -> None:
    result = index(
        paper("1", row("Short sleep", "Weight gain", Direction.INCREASES)),
        paper("2", row("Short sleep", "Weight gain", Direction.DECREASES)),
    )
    outcome = result.groups[0].outcomes[0]
    assert outcome.contested
    assert outcome.directions == [(Direction.INCREASES, 1), (Direction.DECREASES, 1)]


def test_a_null_result_beside_an_effect_is_not_a_disagreement() -> None:
    """Different power or a different population at least as often as a contradiction.
    Flagging it would put a warning on most of the corpus and mean nothing."""
    result = index(
        paper("1", row("Short sleep", "Weight gain", Direction.INCREASES)),
        paper("2", row("Short sleep", "Weight gain", Direction.NONE)),
    )
    assert not result.groups[0].outcomes[0].contested


def test_rows_with_no_recorded_outcome_get_their_own_bucket() -> None:
    """Attaching them to whichever outcome happened to be commonest would be an
    invention; dropping them would hide rows the papers actually reported."""
    result = index(
        paper("1", row("Screen time", "Weight gain"), row("Screen time", "")),
        paper("2", row("Screen time", "Weight gain")),
    )
    outcomes = result.groups[0].outcomes
    assert [outcome.outcome for outcome in outcomes] == ["Weight gain", ""]
    assert len(outcomes[-1].sites) == 1


# --- what the index points at -----------------------------------------------


def test_every_row_carries_the_address_it_came_from() -> None:
    """The one property the whole file rests on: nothing here is readable instead of the
    corpus, because every line is a place in the corpus."""
    result = index(
        paper("11", row("a"), row("Sleep duration"), domain="rest"),
        paper("22", row("Sleep duration"), domain="diet"),
    )
    sites = result.groups[0].sites
    assert {(site.file, site.row) for site in sites} == {
        ("rest/11_Author11.md", 2),
        ("diet/22_Author22.md", 1),
    }


def test_a_predictor_spanning_topics_says_which_ones() -> None:
    result = index(
        paper("1", row("Sleep duration"), domain="rest"),
        paper("2", row("Sleep duration"), domain="diet"),
    )
    assert result.groups[0].topics == ["rest", "diet"]


def test_a_predictor_only_one_paper_reports_is_counted_but_not_listed() -> None:
    """It did not recur, which is what this file is about, and its own document already
    describes it in full. Counted so that what is left out is stated rather than true."""
    result = index(
        paper("1", row("Sleep duration"), row("Screen time")),
        paper("2", row("Sleep duration")),
    )
    assert labels(result) == ["Sleep duration"]
    assert result.once == 1
    assert result.rows == 3
    assert result.papers == 2


def test_a_row_cannot_reach_the_index_without_naming_a_predictor() -> None:
    """Handled in the schema rather than here. `PredictorRow.predictor` is stripped and
    has to be non-empty, so an unnamed row never becomes a record to be grouped."""
    with pytest.raises(ValidationError):
        row("   ")


def test_the_threshold_is_what_the_module_says_it_is() -> None:
    """Pinned so the prose in `predictors.md`, which prints this number, cannot drift
    from the rule that produced it."""
    assert MIN_PAPERS == 2


# --- determinism ------------------------------------------------------------


def test_the_same_records_in_a_different_order_produce_the_same_index() -> None:
    """A bundle rebuilt from cache has to diff clean, or every rebuild looks like a
    change and nobody reads the diffs."""
    records = [
        paper("1", row("Sleep duration", "Weight gain"), domain="rest"),
        paper("2", row("Sleep duration", "Weight gain"), domain="diet"),
        paper("3", row("Screen time", "Weight gain"), domain="rest"),
        paper("4", row("Screen time", "Weight gain"), domain="diet"),
    ]
    forward = index(*records)
    backward = index(*reversed(records))
    assert labels(forward) == labels(backward)
    assert [
        [(site.file, site.row) for site in group.sites] for group in forward.groups
    ] == [[(site.file, site.row) for site in group.sites] for group in backward.groups]


def test_groups_are_ordered_by_how_many_papers_have_to_be_opened() -> None:
    result = index(
        paper("1", row("Sleep duration"), row("Screen time")),
        paper("2", row("Sleep duration"), row("Screen time")),
        paper("3", row("Sleep duration")),
    )
    assert labels(result) == ["Sleep duration", "Screen time"]
