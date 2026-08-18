"""What one predictor cannot be engineered independently of.

The column exists for a reader who is about to build a feature. Knowing that a study
measured `maternal age` is not enough to build it — knowing that the same study found it
collinear with `parity`, and structurally exclusive of `pregnancy status`, is what
decides whether the two can go into a model together.

Four things are pinned here, and each is a way the column could quietly become wrong.

**Magnitude is computed.** The extraction records what the paper printed and nothing
else; the band comes from `band_interaction`. A model asked to grade its own finding
grades it well, and the bundle already carries four separate axes that read as "how much
do I believe this" — `strength`, `Confidence`, the OKF trust tiers and
`ranking.relevance`. A fifth the model could argue with is a fifth thing to reconcile.

**The mirror is derived, not asked for.** A paper states an interaction once, from
whichever variable its sentence started on. The other half is computed, marked, and
labeled with the inverse relationship — a mirrored `modifies` that kept its forward label
would reverse the claim.

**A dropped coefficient is not a dropped interaction.** Numeric verification takes the
number and leaves the relationship, because the relationship is what a reader wanted and
the number is what the model was likeliest to have paraphrased.

**Nothing crosses a paper boundary.** A mirror lands only on a row of the same document,
matched on an exact fold. Anything looser attributes a coefficient to a variable the study
never measured it against.
"""

from __future__ import annotations

from typing import Any

import pytest

from okf_loremaster.emitters.okf import body_for, interaction_cell
from okf_loremaster.interactions import (
    fold_variable,
    interaction_rows,
    mirror_interactions,
    same_variable,
    variable_rows,
)
from okf_loremaster.okf.layout import NONE_CELL, UNVERIFIED_CELL
from okf_loremaster.okf.reader import body_sections, markdown_table
from okf_loremaster.schemas import (
    MAX_INTERACTIONS,
    ConceptRecord,
    Extraction,
    Interaction,
    InteractionMagnitude,
    InteractionType,
    PredictorRow,
    TextBasis,
    band_interaction,
)


def interaction(feature: str, **fields: Any) -> Interaction:
    return Interaction(feature=feature, **fields)


def predictor(name: str, *interactions: Interaction, **fields: Any) -> PredictorRow:
    base: dict[str, Any] = {
        "predictor": name,
        "operationalization": "recorded at baseline",
        "outcome": "the measured outcome",
        "interacts_with": list(interactions),
    }
    return PredictorRow(**{**base, **fields})


def extraction_of(*rows: PredictorRow, **fields: Any) -> Extraction:
    base: dict[str, Any] = {
        "description": "A cohort study of several exposures and one outcome.",
        "bottom_line": "Several exposures were associated with the outcome.",
        "study_design": "cohort study",
        "population": "adults",
        "outcome_definition": "the outcome as measured",
        "predictors": list(rows),
        "caveats": "Observational.",
    }
    return Extraction(**{**base, **fields})


def record_of(extraction: Extraction, *, abstract: str = "") -> ConceptRecord:
    return ConceptRecord(
        pmid="10001",
        domain="alpha",
        title="A study",
        journal="Journal",
        year=2020,
        abstract=abstract,
        text_basis=TextBasis.FULL_TEXT,
        extraction=extraction,
    )


def section(body: str, heading: str) -> str:
    """The text under one `# ` heading, read back the way a downstream reader would.

    Through `body_sections` rather than a slice, so a test that passes is a test the
    bundle's own parser agrees with.
    """
    return next(text for name, text in body_sections(body) if name == heading)


# --- the magnitude vocabulary -------------------------------------------------


@pytest.mark.parametrize(
    ("measure", "value", "expected"),
    [
        ("r", 0.81, InteractionMagnitude.STRONG),
        ("r", 0.52, InteractionMagnitude.MODERATE),
        ("r", 0.11, InteractionMagnitude.WEAK),
        # Sign is not size. A correlation of -0.8 is as strong as one of +0.8, and the
        # direction is already in the relationship rather than in the coefficient.
        ("r", -0.81, InteractionMagnitude.STRONG),
        ("R²", 0.61, InteractionMagnitude.STRONG),
        ("VIF", 12.0, InteractionMagnitude.STRONG),
        # The one inverted scale in the table: tolerance is 1/VIF, so *small* is severe.
        # Banded the same direction as VIF would call the least collinear pair the worst.
        ("tolerance", 0.05, InteractionMagnitude.STRONG),
        ("tolerance", 0.9, InteractionMagnitude.WEAK),
        # Not in the table, and that is an ordinary answer rather than a failure.
        ("beta", 0.9, InteractionMagnitude.STATED),
        ("", 0.9, InteractionMagnitude.STATED),
        # A recognized measure with no number to band.
        ("r", None, InteractionMagnitude.STATED),
    ],
)
def test_a_coefficient_bands_by_its_own_scale(
    measure: str, value: float | None, expected: InteractionMagnitude
) -> None:
    assert band_interaction(InteractionType.CORRELATED, measure, value) == expected


@pytest.mark.parametrize("measure", ["r", "Pearson r", "Pearson's r", " R "])
def test_a_measure_is_recognized_however_the_paper_named_it(measure: str) -> None:
    """Papers write a statistic three ways, and only one of them is the bare symbol.

    Folding on the last token alone loses `pearsonr`; folding on the whole string alone
    loses `Pearson r`. Both are tried, which is why neither spelling silently drops to
    `stated` while the other bands.
    """
    assert (
        band_interaction(InteractionType.CORRELATED, measure, 0.81)
        is InteractionMagnitude.STRONG
    )


@pytest.mark.parametrize(
    "kind",
    [InteractionType.MUTUALLY_EXCLUSIVE, InteractionType.DERIVED_FROM, InteractionType.DERIVES],
)
def test_a_relationship_that_holds_by_construction_outranks_any_coefficient(
    kind: InteractionType,
) -> None:
    """`sex` and `pregnancy status` do not need a correlation to be inseparable.

    Checked before the cutoffs on purpose. A coefficient printed beside a structural
    relationship describes the study's design rather than its findings, and banding it
    would report a weak number where the honest answer is that the pair cannot co-occur.
    """
    assert band_interaction(kind, "r", 0.02) is InteractionMagnitude.STRUCTURAL


def test_every_relationship_has_an_inverse_and_inverting_twice_is_identity() -> None:
    for kind in InteractionType:
        assert kind.inverse.inverse is kind


# --- mirroring ----------------------------------------------------------------


def test_the_other_half_of_an_interaction_is_written_onto_the_other_row() -> None:
    mirrored = mirror_interactions(
        extraction_of(
            predictor("maternal age", interaction("parity", measure="r", value=0.41)),
            predictor("parity"),
        )
    )

    derived = mirrored.predictors[1].interacts_with
    assert [entry.feature for entry in derived] == ["maternal age"]
    assert derived[0].mirrored
    assert derived[0].value == 0.41
    # The stated half is left exactly as the extraction wrote it.
    assert not mirrored.predictors[0].interacts_with[0].mirrored


def test_a_directional_relationship_is_flipped_rather_than_copied() -> None:
    """The one failure that would be worse than not mirroring at all.

    `age modifies dose response` mirrored onto `dose response` as `modifies` would say
    the effect modification runs the other way — a claim the paper did not make and the
    opposite of the one it did.
    """
    mirrored = mirror_interactions(
        extraction_of(
            predictor("age", interaction("dose response", kind=InteractionType.MODIFIES)),
            predictor("dose response"),
        )
    )

    assert mirrored.predictors[1].interacts_with[0].kind is InteractionType.MODIFIED_BY


def test_a_variable_that_is_not_a_row_here_is_left_alone() -> None:
    """Nothing is invented to receive a mirror.

    A study routinely names variables it did not model as predictors. Those are real and
    worth recording against the row that mentioned them; they are simply not rows, so
    there is nowhere for the other half to go.
    """
    mirrored = mirror_interactions(
        extraction_of(
            predictor("maternal age", interaction("season of birth")),
            predictor("parity"),
        )
    )

    assert mirrored.predictors[1].interacts_with == []
    assert [e.feature for e in mirrored.predictors[0].interacts_with] == ["season of birth"]


def test_mirroring_twice_changes_nothing() -> None:
    """Reconcile runs this after verification, and a resumed run reconciles again."""
    once = mirror_interactions(
        extraction_of(
            predictor("maternal age", interaction("parity", measure="r", value=0.41)),
            predictor("parity"),
        )
    )

    assert mirror_interactions(once).model_dump() == once.model_dump()


def test_a_row_that_already_names_the_variable_is_not_given_a_second_line() -> None:
    """Blind to `kind` on purpose: the extraction had its say about this pair.

    A derived line beside a stated one about the same two variables reads as two findings
    where the paper reported one, and the derived one is the weaker of the two.
    """
    mirrored = mirror_interactions(
        extraction_of(
            predictor("maternal age", interaction("parity", kind=InteractionType.CONFOUNDS)),
            predictor("parity", interaction("maternal age", measure="r", value=0.41)),
        )
    )

    stated = mirrored.predictors[1].interacts_with
    assert len(stated) == 1
    assert not stated[0].mirrored


def test_a_row_never_mirrors_onto_itself() -> None:
    mirrored = mirror_interactions(
        extraction_of(
            predictor("maternal age", interaction("Maternal Age")),
            predictor("parity"),
        )
    )

    assert len(mirrored.predictors[0].interacts_with) == 1


def test_matching_folds_case_and_punctuation_and_nothing_else() -> None:
    assert same_variable("Maternal Age", "maternal  age")
    assert same_variable("BMI (kg/m2)", "bmi kg m2")
    # Timid by design. A near-miss left unmirrored costs one derived line; a wrong merge
    # attributes a coefficient to a variable the paper never measured it against.
    assert not same_variable("maternal age", "paternal age")
    assert not same_variable("", "")


def test_two_rows_for_one_predictor_resolve_to_the_first() -> None:
    """One exposure against two outcomes is two rows and one variable.

    An interaction belongs to one of them rather than to both — the second is reachable
    from the first through the `#` column either way.
    """
    rows = [
        predictor("maternal age", outcome="outcome one"),
        predictor("maternal age", outcome="outcome two"),
        predictor("parity"),
    ]

    assert variable_rows(rows)[fold_variable("maternal age")] == 0


def test_the_stated_half_is_listed_before_the_derived_one() -> None:
    row = predictor(
        "parity",
        interaction("maternal age", mirrored=True),
        interaction("gestational age"),
    )

    assert [entry.feature for entry in interaction_rows(row)] == [
        "gestational age",
        "maternal age",
    ]


# --- what the document says ---------------------------------------------------


def test_the_predictor_table_points_at_the_interactions_and_the_section_details_them() -> None:
    """Both, and they say different things.

    The column answers "can I build this independently" while scanning the table the
    reader is already in. The section answers "how do you know" without widening that
    table by three columns, which is what a terminal-width markdown table cannot afford.
    """
    body = body_for(
        record_of(
            mirror_interactions(
                extraction_of(
                    predictor(
                        "maternal age",
                        interaction("parity", measure="r", value=0.41, measure_raw="r = 0.41"),
                        interaction(
                            "pregnancy status", kind=InteractionType.MUTUALLY_EXCLUSIVE
                        ),
                    ),
                    predictor("parity"),
                )
            )
        )
    )

    listed = markdown_table(section(body, "Predictors reported"))
    assert listed[0]["Interacts with"] == "parity; pregnancy status"

    detailed = markdown_table(section(body, "Interactions"))
    assert [(r["Predictor"], r["Interacts with"]) for r in detailed] == [
        ("maternal age", "parity"),
        ("maternal age", "pregnancy status"),
        ("parity", "maternal age"),
    ]
    assert detailed[0]["Type"] == "correlated"
    assert detailed[0]["Magnitude"] == "moderate"
    assert detailed[1]["Magnitude"] == "structural"
    # The derived half says which row stated it, so nothing in the section is a claim
    # without a source inside the same document.
    assert detailed[2]["Evidence"] == "r = 0.41 (mirrored from row 1)"


def test_a_paper_reporting_no_interaction_gets_no_section_and_no_empty_column() -> None:
    """The common case, and it must not cost a reader anything.

    Most studies report none. An empty `# Interactions` heading on every one of those
    would train a reader to skip the section on the few papers that have one.
    """
    body = body_for(record_of(extraction_of(predictor("maternal age"))))

    assert "# Interactions" not in body
    listed = markdown_table(section(body, "Predictors reported"))
    assert listed[0]["Interacts with"] == ""


def test_a_coefficient_verification_removed_reads_as_unverified_rather_than_absent() -> None:
    """Two different answers that must not render as one cell.

    "The paper reported no number" and "the model wrote a number the paper does not
    print" are opposite claims about the same blank space, and a reader deciding whether
    to trust the row needs to know which one they are looking at.
    """
    stated = interaction_cell(interaction("parity"))
    dropped = interaction_cell(interaction("parity", measure_raw="r = 0.41", measure="r"))

    assert stated == NONE_CELL
    assert dropped == UNVERIFIED_CELL


def test_an_extraction_is_capped_at_a_readable_number_of_interactions_per_row() -> None:
    """A budget, not a judgment about which ones matter.

    A correlation matrix has an entry for every pair, and a model handed one will list
    them all. The row is a pointer for a reader, and a pointer with twenty destinations
    is not one.

    Cut by `enforce_budgets` rather than by a field validator, and so cut where every
    other budget is: `reconcile` runs it before verification, which is what keeps the
    verification counts describing what the bundle actually contains.
    """
    over = extraction_of(
        predictor(
            "maternal age",
            *(interaction(f"variable {n}") for n in range(MAX_INTERACTIONS + 4)),
        )
    )

    trimmed, warnings = over.enforce_budgets()

    assert len(trimmed.predictors[0].interacts_with) == MAX_INTERACTIONS
    # Named, not merely counted. A reader who wonders why a row lists four interactions
    # when the paper printed eight has the answer in the run's warnings.
    assert any(
        f"dropped 4 interaction(s) over the per-row limit of {MAX_INTERACTIONS}" in note
        and "maternal age (-4)" in note
        for note in warnings
    )
