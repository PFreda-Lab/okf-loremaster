"""Evidence strength: the arithmetic, and the four things it must not become.

The scoring itself is small. What is worth pinning is everything around it, because each
of these has an obvious-looking implementation that is wrong in a way no output reveals:

- **A missing signal is not a bad one.** An abstract that never printed a design has to
  score as unmeasured, not as a weak study, or the bundle systematically marks down every
  paper we could only read the abstract of.
- **Unmeasured components keep their weight.** Renormalizing over what is known would let
  a paper scored on one signal reach the same number as one scored on four.
- **Magnitude is not strength.** A huge effect with a wide interval is the classic
  small-study artifact; scoring on size of effect would rank exactly the wrong papers up.
- **Nothing here knows what literature it is in.** Sample size scores against the
  charter's scale, so the same n has to come out differently under two charters.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from okf_loremaster.schemas import (
    Charter,
    Extraction,
    PredictorRow,
    StrengthGrade,
    StudyDesign,
    TextBasis,
)
from okf_loremaster.strength import (
    DEFAULT_PAPER_WEIGHTS,
    DEFAULT_ROW_WEIGHTS,
    grade_for,
    score_extraction,
)

# A charter whose only job is to carry a scale. The numbers are arbitrary and mean
# nothing outside these tests, which is the point: they come from the charter.
SCALE = Charter(prompt="anything", sample_size_typical=500, sample_size_large=50_000)


def paper(
    *,
    design: StudyDesign = StudyDesign.UNCLEAR,
    n: int | None = None,
    adjusted_for: list[str] | None = None,
    rows: list[PredictorRow] | None = None,
) -> Extraction:
    return Extraction(
        study_design=design.label,
        design=design,
        n=n,
        adjusted_for=adjusted_for or [],
        predictors=rows if rows is not None else [PredictorRow(predictor="something")],
    )


def row(
    *,
    effect: float | None = None,
    measure: str = "",
    low: float | None = None,
    high: float | None = None,
    p_value: str = "",
    adjusted: bool | None = None,
) -> PredictorRow:
    return PredictorRow(
        predictor="something",
        effect=effect,
        effect_measure=measure,
        ci_low=low,
        ci_high=high,
        p_value=p_value,
        adjusted=adjusted,
    )


# --- design -----------------------------------------------------------------


def test_the_evidence_hierarchy_is_the_conventional_one() -> None:
    scores = [
        score_extraction(paper(design=design), charter=SCALE).score
        for design in (
            StudyDesign.RANDOMIZED_TRIAL,
            StudyDesign.PROSPECTIVE_COHORT,
            StudyDesign.RETROSPECTIVE_COHORT,
            StudyDesign.CASE_CONTROL,
            StudyDesign.CROSS_SECTIONAL,
            StudyDesign.CASE_SERIES,
        )
    ]

    assert scores == sorted(scores, reverse=True)


def test_an_unstated_design_is_unmeasured_rather_than_weak() -> None:
    """The one that would quietly punish every abstract-only paper in a bundle."""
    unclear = score_extraction(paper(design=StudyDesign.UNCLEAR), charter=SCALE)
    weak = score_extraction(paper(design=StudyDesign.CASE_SERIES), charter=SCALE)

    assert "design" in unclear.unmeasured
    assert "design" not in weak.unmeasured
    assert unclear.score > weak.score


# --- sample size ------------------------------------------------------------


def test_the_same_sample_size_scores_differently_under_two_charters() -> None:
    """Field-relative by construction. 2,000 is a large study in one literature and a
    pilot in another, and no constant in `src/` can know which."""
    small_field = Charter(prompt="x", sample_size_typical=100, sample_size_large=2_000)
    large_field = Charter(prompt="x", sample_size_typical=10_000, sample_size_large=1_000_000)
    study = paper(design=StudyDesign.RETROSPECTIVE_COHORT, n=2_000)

    assert (
        score_extraction(study, charter=small_field).score
        > score_extraction(study, charter=large_field).score
    )


def test_sample_size_is_unmeasured_without_a_scale_to_measure_it_against() -> None:
    """No charter, or a charter that declined to guess: no size component either. A
    default scale here would be a constant that is right for one literature."""
    study = paper(design=StudyDesign.PROSPECTIVE_COHORT, n=4_000)

    assert "size" in score_extraction(study, charter=None).unmeasured
    assert "size" in score_extraction(study, charter=Charter(prompt="x")).unmeasured
    assert "size" not in score_extraction(study, charter=SCALE).unmeasured


def test_size_scales_by_order_of_magnitude_not_by_headcount() -> None:
    """Logarithmic, so 500 to 5,000 is the same step as 5,000 to 50,000. Linear scaling
    would compress every study below the largest into one indistinguishable band."""
    scores = [
        score_extraction(paper(n=n), charter=SCALE).parts["size"]
        for n in (500, 5_000, 50_000)
    ]
    first, second = scores[1] - scores[0], scores[2] - scores[1]

    assert first == pytest.approx(second, abs=1e-6)


def test_a_study_larger_than_the_charters_ceiling_does_not_run_off_the_scale() -> None:
    huge = score_extraction(paper(n=10_000_000), charter=SCALE)

    assert huge.parts["size"] == pytest.approx(DEFAULT_PAPER_WEIGHTS.size)


# --- adjustment -------------------------------------------------------------


def test_adjustment_is_read_off_the_rows_because_that_is_where_it_varies() -> None:
    """Papers print an unadjusted and an adjusted column, and they are different claims."""
    both = paper(rows=[row(adjusted=True), row(adjusted=False)])
    all_adjusted = paper(rows=[row(adjusted=True), row(adjusted=True)])

    assert score_extraction(both, charter=SCALE).parts["adjustment"] == pytest.approx(
        DEFAULT_PAPER_WEIGHTS.adjustment * 0.5
    )
    assert score_extraction(all_adjusted, charter=SCALE).parts["adjustment"] == pytest.approx(
        DEFAULT_PAPER_WEIGHTS.adjustment
    )


def test_a_stated_covariate_set_counts_for_something_but_not_for_everything() -> None:
    """It shows the paper adjusted *something*. It does not establish that these
    estimates are the adjusted ones, so it scores below a row that says so outright."""
    covariates = paper(adjusted_for=["age", "sex"], rows=[row()])
    stated = paper(rows=[row(adjusted=True)])
    silent = paper(rows=[row()])

    assert (
        score_extraction(silent, charter=SCALE).score
        < score_extraction(covariates, charter=SCALE).score
        < score_extraction(stated, charter=SCALE).score
    )
    assert "adjustment" in score_extraction(silent, charter=SCALE).unmeasured


# --- reading depth ----------------------------------------------------------


def test_reading_the_full_text_supports_more_than_reading_the_abstract() -> None:
    study = paper(design=StudyDesign.PROSPECTIVE_COHORT, n=5_000)

    assert (
        score_extraction(study, charter=SCALE, basis=TextBasis.FULL_TEXT).score
        > score_extraction(study, charter=SCALE, basis=TextBasis.ABSTRACT).score
    )


def test_reading_depth_is_always_measured() -> None:
    """We always know which of the two we read, so it is never an absence."""
    for basis in (TextBasis.FULL_TEXT, TextBasis.ABSTRACT):
        assert "basis" not in score_extraction(paper(), charter=SCALE, basis=basis).unmeasured


# --- precision --------------------------------------------------------------


def test_a_tight_interval_beats_a_wide_one() -> None:
    tight = score_extraction(
        paper(rows=[row(effect=1.8, measure="OR", low=1.7, high=1.9)]), charter=SCALE
    )
    wide = score_extraction(
        paper(rows=[row(effect=1.8, measure="OR", low=0.4, high=8.0)]), charter=SCALE
    )

    assert tight.rows[0].score > wide.rows[0].score


def test_a_big_effect_from_a_shaky_study_does_not_outscore_a_modest_precise_one() -> None:
    """The winner's curse, stated as a test. Underpowered studies survive publication by
    overstating, so scoring on magnitude would rank exactly the wrong papers up."""
    dramatic = score_extraction(
        paper(rows=[row(effect=9.0, measure="OR", low=1.1, high=74.0)]), charter=SCALE
    )
    modest = score_extraction(
        paper(rows=[row(effect=1.3, measure="OR", low=1.2, high=1.4)]), charter=SCALE
    )

    assert modest.rows[0].score > dramatic.rows[0].score


def test_ratio_intervals_are_read_on_a_log_scale() -> None:
    """OR 2.0 spanning 1.0-4.0 is exactly as wide as OR 0.5 spanning 0.25-1.0.
    Subtracting the bounds says otherwise, and says it in the direction that favors
    protective effects over harmful ones for no reason at all."""
    harmful = score_extraction(
        paper(rows=[row(effect=2.0, measure="OR", low=1.0, high=4.0)]), charter=SCALE
    )
    protective = score_extraction(
        paper(rows=[row(effect=0.5, measure="OR", low=0.25, high=1.0)]), charter=SCALE
    )

    assert harmful.rows[0].score == pytest.approx(protective.rows[0].score)


def test_a_measure_named_in_prose_is_not_mistaken_for_an_odds_ratio() -> None:
    """`\\bor\\b` inside "mean difference" would read an additive measure as a ratio and
    score its interval on the wrong scale."""
    additive = score_extraction(
        paper(rows=[row(effect=10.0, measure="mean difference", low=8.0, high=12.0)]),
        charter=SCALE,
    )
    ratio = score_extraction(
        paper(rows=[row(effect=10.0, measure="OR", low=8.0, high=12.0)]), charter=SCALE
    )

    # Same numbers, different measures, so necessarily different precision.
    assert additive.rows[0].parts["precision"] != ratio.rows[0].parts["precision"]


def test_no_interval_is_unmeasured_and_a_p_value_is_not_a_substitute() -> None:
    """A null result with a tight interval around no effect is precise evidence of
    nothing, and a p-value cannot tell that apart from a study too small to find
    anything."""
    scored = score_extraction(
        paper(rows=[row(effect=1.4, measure="OR", p_value="<0.001")]), charter=SCALE
    )

    assert "precision" in scored.rows[0].unmeasured


# --- how the parts combine --------------------------------------------------


def test_an_unmeasured_component_keeps_its_weight_instead_of_being_renormalized() -> None:
    """The subtle one. Redistributing an absent signal's weight over the known ones makes
    a paper scored on one signal look as authoritative as one scored on four."""
    everything = score_extraction(
        paper(
            design=StudyDesign.RANDOMIZED_TRIAL,
            n=50_000,
            rows=[row(adjusted=True)],
        ),
        charter=SCALE,
        basis=TextBasis.FULL_TEXT,
    )
    design_only = score_extraction(
        paper(design=StudyDesign.RANDOMIZED_TRIAL),
        charter=None,
        basis=TextBasis.FULL_TEXT,
    )

    assert everything.score > design_only.score
    assert design_only.unmeasured == ["size", "adjustment"]


def test_a_paper_with_nothing_measured_lands_mid_scale_and_says_why() -> None:
    scored = score_extraction(paper(), charter=None)

    assert scored.unmeasured == ["design", "size", "adjustment"]
    assert scored.score == pytest.approx(0.5 * (1 - DEFAULT_PAPER_WEIGHTS.basis) + 0.4 *
                                         DEFAULT_PAPER_WEIGHTS.basis)


def test_the_parts_add_up_to_the_score() -> None:
    """`parts` is the audit trail. If it does not reconstruct the total it is decoration,
    and a score nobody can decompose is a score nobody can argue with."""
    scored = score_extraction(
        paper(design=StudyDesign.PROSPECTIVE_COHORT, n=9_000, rows=[row(adjusted=True)]),
        charter=SCALE,
        basis=TextBasis.FULL_TEXT,
    )

    assert sum(scored.parts.values()) == pytest.approx(scored.score, abs=1e-3)
    assert sum(scored.rows[0].parts.values()) == pytest.approx(scored.rows[0].score, abs=1e-3)


# --- rows against their paper -----------------------------------------------


def test_rows_come_back_positionally_parallel_to_the_predictors() -> None:
    """The emitter pairs them by index. A length mismatch would shift every strength cell
    onto the wrong row, which renders perfectly and is entirely wrong."""
    rows = [row(adjusted=True), row(adjusted=False), row()]
    scored = score_extraction(paper(rows=rows), charter=SCALE)

    assert len(scored.rows) == len(rows)


def test_a_row_cannot_outrun_the_study_it_came_from() -> None:
    """Same row, two papers. Half the row's score is the paper's, so the design it was
    measured under has to move it."""
    measured = row(effect=1.5, measure="HR", low=1.4, high=1.6, adjusted=True)
    strong = score_extraction(
        paper(design=StudyDesign.RANDOMIZED_TRIAL, n=50_000, rows=[measured]),
        charter=SCALE,
        basis=TextBasis.FULL_TEXT,
    )
    weak = score_extraction(paper(design=StudyDesign.CASE_SERIES, rows=[measured]), charter=SCALE)

    assert strong.rows[0].score > weak.rows[0].score


def test_an_unadjusted_estimate_is_not_scored_as_zero() -> None:
    """A paper that says plainly it did not adjust has told the truth about a real
    result, and that is different from a paper that said nothing."""
    stated_no = score_extraction(paper(rows=[row(adjusted=False)]), charter=SCALE)

    assert stated_no.rows[0].parts["adjusted"] > 0.0
    assert stated_no.rows[0].parts["adjusted"] < DEFAULT_ROW_WEIGHTS.adjusted


# --- grades -----------------------------------------------------------------


def test_the_bands_are_ordered_and_wide() -> None:
    assert grade_for(0.95) is StrengthGrade.STRONG
    assert grade_for(0.60) is StrengthGrade.MODERATE
    assert grade_for(0.20) is StrengthGrade.LIMITED


def test_nothing_measured_is_ungraded_rather_than_moderate() -> None:
    """The score lands mid-scale by arithmetic, not by evidence. Printing that as
    `moderate` would dress an absence up as a finding."""
    assert grade_for(0.5, measured=False) is StrengthGrade.UNGRADED


def test_a_strong_paper_grades_strong_end_to_end() -> None:
    scored = score_extraction(
        paper(design=StudyDesign.RANDOMIZED_TRIAL, n=80_000, rows=[row(adjusted=True)]),
        charter=SCALE,
        basis=TextBasis.FULL_TEXT,
    )

    assert scored.grade is StrengthGrade.STRONG
    assert scored.graded


# --- the charter's scale ----------------------------------------------------


def test_half_a_scale_is_rejected_rather_than_half_applied() -> None:
    """One anchor scores nothing, so a charter carrying one is a silent no-op."""
    with pytest.raises(ValidationError, match="go together"):
        Charter(prompt="x", sample_size_typical=500)
    with pytest.raises(ValidationError, match="go together"):
        Charter(prompt="x", sample_size_large=50_000)


def test_an_inverted_scale_is_rejected() -> None:
    """Worse than a no-op: it would score every large study as small."""
    with pytest.raises(ValidationError, match="must exceed"):
        Charter(prompt="x", sample_size_typical=50_000, sample_size_large=500)


def test_a_charter_with_no_scale_at_all_is_legitimate() -> None:
    """A hand-written charter, or a model that declined to guess."""
    charter = Charter(prompt="x")

    assert charter.sample_size_typical is None
    assert charter.sample_size_large is None
