"""Numbers checked against the text they were taken from.

The step 6 gate is the pair at the bottom: the same corpus read twice, once by an
extractor that copies its numbers out of the paper and once by one that invents a single
odds ratio. The faithful run has to come back clean and the fabricating one has to lose
exactly the invented number, keep everything around it, say so in a warning naming the
paper, and finish. A check that flagged nothing and a check that flagged everything
would each pass half of that.

The rest drives `verify_extraction` directly, on the cases a corpus of well-behaved
synthetic papers never produces: a Lancet middle dot, a true minus sign, an interval
whose hyphen is not a minus, a quote that scopes the check to one sentence, and an
effect that contradicts the string it was supposedly copied from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from okf_loremaster.schemas import Confidence, Extraction, NullFinding, PredictorRow
from okf_loremaster.verification import (
    EN_DASH,
    MIDDLE_DOT,
    MINUS,
    Quantity,
    Source,
    normalize,
    quantities_in,
    verify_extraction,
)

from fake_llm import supported
from fake_ncbi import REFERENCE_ONLY, TOPICS, effect_for, finding_sentence, has_full_text
from graph_runs import TARGET, full_run, scripted_run

# The number the fabricating extractor reports. Not in any paper, and far enough from
# every real effect that no rounding tolerance could reach it.
FABRICATED = 4.44
FABRICATED_RAW = "4.44 (95% CI 3.10-6.02)"


# --- reading numbers out of prose -------------------------------------------


def test_the_shapes_journals_actually_print_are_read_as_numbers() -> None:
    """A house style is not a fabrication. A paper printing `1·82` said 1.82."""
    text = f"OR 1{MIDDLE_DOT}82, beta {MINUS}0.44, n = 12,405"
    values = [q.value for q in quantities_in(text)]

    assert values == [1.82, -0.44, 12405]


def test_an_interval_is_not_a_pair_of_negative_numbers() -> None:
    """The one that silently corrupts everything: `1.21-2.74` is two positive bounds,
    and reading the second as -2.74 would make every real interval unverifiable."""
    assert [q.value for q in quantities_in("CI 1.21-2.74")] == [1.21, 2.74]
    assert [q.value for q in quantities_in(f"CI 1.21{EN_DASH}2.74")] == [1.21, 2.74]
    # Typeset with spaces around the dash, it is the same interval.
    assert [q.value for q in quantities_in("CI 1.21 - 2.74")] == [1.21, 2.74]
    # A hyphen inside a word is not a sign either.
    assert [q.value for q in quantities_in("at 30-day follow-up, 12 events")] == [30, 12]


def test_a_claim_may_be_less_precise_than_its_source_and_never_more() -> None:
    source = Source("the adjusted odds ratio was 1.84")

    assert source.holds(1.8)  # a rounding, not a fabrication
    assert source.holds(1.84)
    assert not source.holds(1.9)
    assert not source.holds(18.4)


def test_a_bare_integer_does_not_support_a_claimed_decimal() -> None:
    """Otherwise the check is no check: a paper is full of years, counts and table
    numbers, and rounding a claim to zero places matches a claimed 4.44 against any 4."""
    source = Source("Table 4 lists the 2019 cohort of 244 adults.")

    assert not source.holds(4.44)
    assert not source.holds(2.44)
    assert source.holds(244)


def test_precision_survives_a_tiny_effect() -> None:
    """`repr(0.00003)` is `3e-05`, which has no decimals at all. A `Quantity` that
    believed that would treat every small coefficient as an integer and check nothing."""
    assert Quantity.of(0.00003).decimals == 5


def test_quote_matching_ignores_punctuation_and_reflowed_whitespace() -> None:
    source = Source("The association held\nafter adjustment (P = 0.03).")

    assert source.holds_quote("The association held after adjustment (P = 0.03)")
    assert not source.holds_quote("The association did not hold after adjustment")
    assert normalize("A--B  c!") == "a b c"


# --- what verification does to an extraction --------------------------------


def one_row(**fields: Any) -> Extraction:
    return Extraction(predictors=[PredictorRow(**{"predictor": "exposure", **fields})])


SENTENCE = "In adjusted models the association was 1.82 (95% CI 1.21-2.74)."


def test_an_effect_the_source_does_not_contain_is_removed_and_the_row_kept() -> None:
    extraction = one_row(
        effect=3.91,
        effect_raw="3.91",
        operationalization="recorded at baseline",
        timing="before the outcome window",
        confidence=Confidence.HIGH,
    )

    check = verify_extraction(extraction, SENTENCE)
    row = check.extraction.predictors[0]

    assert row.effect is None
    assert row.confidence is Confidence.MEDIUM
    # The paper did report the predictor; only the magnitude was unsupported.
    assert row.predictor == "exposure"
    assert row.operationalization == "recorded at baseline"
    assert row.timing == "before the outcome window"
    assert check.effects_dropped == 1
    assert not check.clean


def test_a_supported_effect_with_an_unsupported_bound_keeps_the_point_estimate() -> None:
    extraction = one_row(
        effect=1.82, effect_raw="1.82", ci_low=1.21, ci_high=9.99, confidence=Confidence.HIGH
    )

    check = verify_extraction(extraction, SENTENCE)
    row = check.extraction.predictors[0]

    assert row.effect == 1.82
    assert row.ci_low is None and row.ci_high is None
    assert row.confidence is Confidence.MEDIUM
    assert check.intervals_dropped == 1
    assert check.effects_dropped == 0


def test_a_quote_scopes_the_check_to_one_sentence() -> None:
    """A full text has hundreds of numbers, so document-wide matching is a coincidence
    waiting to happen. A row that quoted its source is held to that sentence."""
    document = f"{SENTENCE}\n\nTable 2 lists 3.91 kg of something else entirely."
    quoted = one_row(effect=3.91, effect_raw="3.91", quote=SENTENCE)
    unquoted = one_row(effect=3.91, effect_raw="3.91")

    assert verify_extraction(quoted, document).effects_dropped == 1
    # No quote, so the whole document is the scope and the coincidence survives. The
    # weaker check, and the only alternative to none.
    assert verify_extraction(unquoted, document).effects_dropped == 0


def test_an_effect_that_contradicts_its_own_raw_string_is_caught_without_a_source() -> None:
    """A silent unit conversion: both numbers are in the paper and they disagree with
    each other, which `effect_raw` exists to make visible."""
    extraction = one_row(effect=3.91, effect_raw="1.82 (95% CI 1.21-2.74)")

    check = verify_extraction(extraction, f"{SENTENCE} A separate figure reports 3.91.")

    assert check.extraction.predictors[0].effect is None
    assert check.effects_dropped == 1


def test_a_quote_the_source_does_not_contain_is_dropped_and_counted() -> None:
    extraction = one_row(quote="The authors concluded the opposite of what they found.")

    check = verify_extraction(extraction, SENTENCE)

    assert check.extraction.predictors[0].quote == ""
    assert check.quotes_dropped == 1
    assert check.rows[0].quote_missing


def test_a_sample_size_nobody_stated_is_dropped() -> None:
    check = verify_extraction(Extraction(n=4096), SENTENCE)

    assert check.extraction.n is None
    assert check.sample_size_missing

    kept = verify_extraction(Extraction(n=1454), f"{SENTENCE} Among 1454 adults.")
    assert kept.extraction.n == 1454


def test_a_null_finding_loses_an_unsupported_quote_and_survives_it() -> None:
    """There is no magnitude here to remove — the whole claim is that there was none —
    so the finding has to outlive its quote or the check would delete evidence."""
    extraction = Extraction(
        null_findings=[NullFinding(predictor="another exposure", quote="never written")]
    )

    check = verify_extraction(extraction, SENTENCE)

    assert [f.predictor for f in check.extraction.null_findings] == ["another exposure"]
    assert check.extraction.null_findings[0].quote == ""
    assert check.quotes_dropped == 1


def test_an_extraction_with_no_source_at_all_fails_loudly_rather_than_passing() -> None:
    check = verify_extraction(one_row(effect=1.82, effect_raw="1.82"), "")

    assert check.effects_dropped == 1


# --- the step 6 gate, end to end --------------------------------------------


def fabricating(source: str) -> dict[str, Any]:
    """A faithful reading with one number replaced by one nobody wrote.

    Everything else stays: the predictor, the timing, the quote, the sample size. What
    survives is as much the subject as what does not.
    """
    reading = supported(source)
    reading["predictors"][0] |= {
        "effect": FABRICATED,
        "effect_measure": "adjusted OR",
        "effect_raw": FABRICATED_RAW,
        "ci_low": 3.10,
        "ci_high": 6.02,
    }
    return reading


async def test_a_faithful_reading_survives_verification_untouched(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control. Without it, a check that deleted every number would pass the gate."""
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    summary = run.state["verification"]
    assert summary is not None
    assert summary.clean, summary.examples
    assert summary.papers == TARGET
    assert summary.rows == TARGET

    full_text = [r for r in run.records if r.text_basis.value == "full_text"]
    assert full_text, "no paper was read in full, so nothing exercised the checked path"
    for record in full_text:
        row = record.extraction.predictors[0]
        assert row.effect == effect_for(record.pmid)
        assert row.quote == finding_sentence(record.pmid)


async def test_a_fabricated_odds_ratio_is_caught_and_the_run_continues(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 6 gate.

    `effect=None`, the confidence one step lower, a warning that names the paper and the
    number, and a finished bundle — every paper still has a record, every topic still has
    its papers, and everything the row said apart from the magnitude is still there.
    """
    run = await full_run(
        settings_factory, tmp_path, monkeypatch, scripted=scripted_run(extract=fabricating)
    )

    assert len(run.records) == TARGET
    assert sum(len(pmids) for pmids in run.topics.values()) == TARGET

    for record in run.records:
        row = record.extraction.predictors[0]
        assert row.effect is None
        assert row.ci_low is None and row.ci_high is None
        assert row.confidence is Confidence.MEDIUM  # claimed high
        assert row.predictor == "exposure"
        assert row.timing == "before the outcome window"

    summary = run.state["verification"]
    assert summary is not None
    assert summary.effects_dropped == TARGET

    dropped = [note for note in run.warnings if "removed" in note and "effect size" in note]
    assert len(dropped) == 1
    assert str(FABRICATED) in dropped[0]
    assert any(record.pmid in dropped[0] for record in run.records)


async def test_a_number_that_is_only_in_the_reference_list_is_not_in_the_paper(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scope of the check is the text the extractor read, and the reference list is
    not in it. A number lifted from a cited title is another study's, and attributing it
    to this one is the kind of error that reads as an ordinary finding forever after."""

    def from_the_references(source: str) -> dict[str, Any]:
        reading = supported(source)
        reading["predictors"][0] |= {
            "effect": REFERENCE_ONLY,
            "effect_raw": str(REFERENCE_ONLY),
            "ci_low": None,
            "ci_high": None,
            "quote": "",
        }
        return reading

    run = await full_run(
        settings_factory,
        tmp_path,
        monkeypatch,
        scripted=scripted_run(extract=from_the_references),
    )

    assert all(record.extraction.predictors[0].effect is None for record in run.records)


def test_the_corpus_really_does_serve_full_text_for_some_papers_and_not_others() -> None:
    """A fixture assertion. If BioC ever answered for everything, the abstract fallback
    would stop being exercised and nothing would say so."""
    pmids = [f"{10000 + topic * 100 + n}" for topic in range(len(TOPICS)) for n in range(20)]
    served = [pmid for pmid in pmids if has_full_text(pmid)]

    assert 0 < len(served) < len(pmids)
