"""Literals checked against the text they were taken from — numbers, quotes and codes.

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

from okf_loremaster.schemas import (
    CODE,
    MAX_LOCATED_QUOTE_WORDS,
    CodedAs,
    Confidence,
    Extraction,
    NullFinding,
    PredictorRow,
    VocabularyHint,
)
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
from fake_ncbi import (
    REFERENCE_ONLY,
    TOPICS,
    code_for,
    effect_for,
    finding_sentence,
    has_full_text,
)
from graph_runs import TARGET, full_run, scripted_run

# The number the fabricating extractor reports. Not in any paper, and far enough from
# every real effect that no rounding tolerance could reach it.
FABRICATED = 4.44
FABRICATED_RAW = "4.44 (95% CI 3.10-6.02)"

# A well-formed ICD-10 code that no paper in the corpus prints. `code_for` never produces
# a `Z` prefix, so this cannot collide with a real one however the fixture grows.
INVENTED_CODE = "Z99.9"


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


# --- quote locators ---------------------------------------------------------
#
# An extraction writes the opening words of a sentence rather than copying the whole of
# it, and `verify_extraction` grows them back from the source before anything is checked.
# The saving is real money — a quote is the longest field in a predictor row and there is
# one per row — but the reason these tests exist is that the check downstream narrows its
# scope to the quoted sentence. Expand to the wrong text and the numeric check is being
# run against the wrong sentence, which is worse than not narrowing at all.


def test_a_locator_grows_into_the_sentence_it_opens() -> None:
    document = f"Methods were preregistered.\n\n{SENTENCE}\n\nTable 2 follows."

    check = verify_extraction(one_row(quote="In adjusted models"), document)

    assert check.extraction.predictors[0].quote == SENTENCE
    assert check.quotes_dropped == 0


def test_a_locator_does_not_drag_the_heading_above_it_into_the_quote() -> None:
    """Full text is headings and captions with paragraphs between them, not prose end to
    end. `sentences` collapses whitespace, so without a line-aware split `## RESULTS`
    has no terminator, joins the sentence after it, and rides along on every quote taken
    from that section."""
    document = f"## RESULTS\n{SENTENCE}"

    check = verify_extraction(one_row(quote="In adjusted models"), document)

    assert check.extraction.predictors[0].quote == SENTENCE


def test_expanding_a_quote_that_is_already_whole_returns_it_unchanged() -> None:
    """Idempotent, so an extraction written before locators existed still verifies, and
    so re-running the check twice cannot grow a quote a second time."""
    check = verify_extraction(one_row(quote=SENTENCE), SENTENCE)

    assert check.extraction.predictors[0].quote == SENTENCE
    assert check.quotes_dropped == 0


def test_a_locator_scopes_the_numeric_check_to_the_sentence_it_found() -> None:
    """The point of the whole mechanism. Ten words have to buy the same narrow scope a
    copied sentence used to, or the saving was paid for out of the check."""
    document = f"{SENTENCE}\n\nTable 2 lists 3.91 kg of something else entirely."

    located = one_row(effect=3.91, effect_raw="3.91", quote="In adjusted models the")

    assert verify_extraction(located, document).effects_dropped == 1


def test_a_locator_matching_nothing_leaves_the_row_where_it_was() -> None:
    """Degrades to exactly the behavior that existed before locators did: the text is
    kept as written and the ordinary quote check drops it."""
    check = verify_extraction(one_row(quote="In unadjusted models"), SENTENCE)

    assert check.extraction.predictors[0].quote == ""
    assert check.quotes_dropped == 1


def test_a_table_is_not_a_sentence_however_the_splitter_reads_it() -> None:
    """BioC delivers a table as one unbroken line with no terminator in it, so the
    splitter returns the whole table as one "sentence". Uncapped, a locator landing
    inside one carried the entire table into the bundle once per row that quoted it —
    a real run produced a 10,281-word document that was one 1,166-word table, eight
    times."""
    table = "Characteristic " + " ".join(f"row {n} value {n}.0" for n in range(200))
    document = f"Table 2 follows.\n{table}\nThe discussion begins here."

    check = verify_extraction(one_row(quote="row 40 value"), document)
    quote = check.extraction.predictors[0].quote

    assert len(quote.split()) == MAX_LOCATED_QUOTE_WORDS
    # Opening at the locator, not at the table's first column: a locator names where the
    # finding starts, and the numbers it was written to point at come after it.
    assert quote.startswith("row 40 value 40.0 row 41")
    # Still a contiguous slice of the source, so still verbatim.
    assert quote in " ".join(document.split())
    assert check.quotes_dropped == 0


def test_windowing_a_table_keeps_the_row_the_locator_pointed_at_in_scope() -> None:
    """The window is a narrower numeric scope than the table, not a broken one. The
    point of expanding at all is that `scope` checks a row's numbers against its quote —
    and a quote holding a whole table holds every number in it, which is a check that
    has stopped discriminating."""
    table = "Characteristic " + " ".join(f"row {n} value {n}.0" for n in range(200))

    near = one_row(effect=40.0, effect_raw="40.0", quote="row 40 value")
    far = one_row(effect=180.0, effect_raw="180.0", quote="row 40 value")

    assert verify_extraction(near, table).effects_dropped == 0
    # In the table, and no longer in the quote — which is the whole reason for the cap.
    assert verify_extraction(far, table).effects_dropped == 1


def test_a_long_sentence_that_is_really_a_sentence_is_left_whole() -> None:
    """The cap is set above what prose reaches — median 32 words across two finished
    bundles, p75 47 — so it must not start editing papers that were merely wordy."""
    long_sentence = (
        "In models adjusted for " + ", ".join(f"covariate {n}" for n in range(20)) + ", "
        "the association was 1.82 (95% CI 1.21-2.74)."
    )
    assert len(long_sentence.split()) < MAX_LOCATED_QUOTE_WORDS

    check = verify_extraction(one_row(quote="In models adjusted"), long_sentence)

    assert check.extraction.predictors[0].quote == long_sentence


def test_a_null_finding_locator_expands_too() -> None:
    document = f"{SENTENCE}\n\nAge showed no association with the outcome (p = 0.41)."
    extraction = Extraction(
        null_findings=[NullFinding(predictor="age", quote="Age showed no association")]
    )

    check = verify_extraction(extraction, document)

    assert check.extraction.null_findings[0].quote == (
        "Age showed no association with the outcome (p = 0.41)."
    )
    assert check.quotes_dropped == 0


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


# --- vocabulary codes --------------------------------------------------------
# A model asked for a condition's ICD-10 code will supply a plausible one from its own
# memory whether or not the paper printed it, and a fabricated code is worse than a
# fabricated effect size: it is short, well-formed, and indistinguishable from a real one.

CODED = "Cases were identified by ICD-10 E11.9 and the SNOMED CT concept 44054006."


def hint(concept: str, *codes: tuple[str, str]) -> Extraction:
    return Extraction(
        vocabulary_hints=[
            VocabularyHint(
                concept=concept,
                codes=[CodedAs(system=system, code=code) for system, code in codes],
            )
        ]
    )


def test_a_code_the_paper_printed_survives() -> None:
    check = verify_extraction(
        hint("type 2 diabetes", ("ICD-10", "E11.9"), ("SNOMED", "44054006")), CODED
    )

    assert check.codes_dropped == 0
    assert [(c.system, c.code) for c in check.extraction.vocabulary_hints[0].codes] == [
        ("icd10", "E11.9"),
        ("snomed", "44054006"),
    ]


def test_a_code_the_paper_never_printed_is_dropped_and_its_concept_kept() -> None:
    """The concept is the part a reader was going to read. Only the code was invented."""
    check = verify_extraction(hint("type 2 diabetes", ("ICD-10", "E10.9")), CODED)

    assert check.codes_dropped == 1
    assert check.codes_missing == (("type 2 diabetes", "icd10", "E10.9"),)
    assert [h.concept for h in check.extraction.vocabulary_hints] == ["type 2 diabetes"]
    assert check.extraction.vocabulary_hints[0].codes == []
    assert not check.clean
    # Tagged with the check that took it, so the code warning names it and no other does.
    assert (CODE, "'type 2 diabetes': icd10 E10.9 is not in the source text") in check.notes()


def test_one_invented_code_does_not_take_a_real_one_with_it() -> None:
    check = verify_extraction(
        hint("type 2 diabetes", ("ICD-10", "E11.9"), ("SNOMED", "73211009")), CODED
    )

    assert check.codes_dropped == 1
    assert [c.code for c in check.extraction.vocabulary_hints[0].codes] == ["E11.9"]


def test_a_code_is_matched_through_the_punctuation_a_typesetter_reflows() -> None:
    """`E11.9`, `E11·9` and `E11 9` are one code; the dot is typography, not identity."""
    for printed in ("code E11.9 was", f"code E11{MIDDLE_DOT}9 was", "code E11 9 was"):
        assert Source(printed).holds_code("E11.9"), printed


def test_a_short_numeric_code_is_not_found_inside_a_longer_number() -> None:
    """`250` is a real ICD-9 code, and an unbounded search finds it in every page range.

    This is the difference between the code check and the quote check: matching a code
    the way quotes are matched would accept almost any short code ever invented.
    """
    assert not Source("A cohort of 1250 adults, pages 2500-2507.").holds_code("250")
    assert Source("classified under 250 in the registry.").holds_code("250")


def test_a_concept_with_no_codes_passes_untouched() -> None:
    """The normal case: most papers name a variable and never code it."""
    check = verify_extraction(hint("clinical frailty scale score"), CODED)

    assert check.codes_dropped == 0
    assert check.clean


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
        # The code the paper printed survives, so a check that dropped every code would
        # fail here rather than only in the fabricating run below.
        codes = record.extraction.vocabulary_hints[0].codes
        assert [c.code for c in codes] == [code_for(record.pmid)]


async def test_a_looked_up_code_the_paper_never_printed_is_caught(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same gate as the fabricated odds ratio, for the field that invites it more.

    Asked for a condition's ICD-10 code, a model will supply a plausible one from memory
    whether or not the paper printed it — and unlike a wrong number, a wrong code is
    well-formed and unremarkable to everything downstream. The concept has to survive,
    because the paper did name that variable.
    """

    def looking_it_up(source: str) -> dict[str, Any]:
        reading = supported(source)
        reading["vocabulary_hints"] = [
            {"concept": "the exposure", "codes": [{"system": "icd10", "code": INVENTED_CODE}]}
        ]
        return reading

    run = await full_run(
        settings_factory, tmp_path, monkeypatch, scripted=scripted_run(extract=looking_it_up)
    )

    assert len(run.records) == TARGET
    for record in run.records:
        hint = record.extraction.vocabulary_hints[0]
        assert hint.concept == "the exposure"
        assert hint.codes == []

    summary = run.state["verification"]
    assert summary is not None
    assert summary.codes_dropped == TARGET
    assert summary.effects_dropped == 0  # nothing else was touched

    dropped = [note for note in run.warnings if "vocabulary code" in note]
    assert len(dropped) == 1
    assert "concepts they were attached to were kept" in dropped[0]


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
