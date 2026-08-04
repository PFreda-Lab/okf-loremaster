"""Schema behavior — the invariants that are code rather than prompt instructions.

Every test here exists because the thing it checks is one a model, or a later edit,
would otherwise get wrong invisibly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from okf_loremaster.clients import Clients
from okf_loremaster.okf.layout import FULL_TEXT_BASIS
from okf_loremaster.schemas import (
    MAX_BOTTOM_LINE_SENTENCES,
    MAX_DESCRIPTION_CHARS,
    MAX_NULL_FINDINGS,
    MAX_PREDICTOR_ROWS,
    MAX_VOCABULARY_HINTS,
    NONE_REPORTED,
    Candidate,
    Charter,
    CodedAs,
    ConceptRecord,
    Confidence,
    CostSummary,
    CurationDecision,
    CurationResult,
    Direction,
    EvidenceType,
    Extraction,
    NullFinding,
    PredictorRow,
    RunManifest,
    ScreenVerdict,
    SourceRef,
    TextBasis,
    Topic,
    TopicGap,
    VocabularyHint,
    is_export_safe,
    slugify,
)
from okf_loremaster.schemas.limits import sentences, truncate_chars
from okf_loremaster.schemas.parse import SchemaError, extract_json, parse_model


def a_topic(slug: str = "risk-factors") -> Topic:
    return Topic(slug=slug, title=slug.replace("-", " ").title(), scope="what belongs here")


def a_charter(**overrides: object) -> Charter:
    base: dict[str, object] = {
        "prompt": "build a bundle",
        "topic_taxonomy": [a_topic(), a_topic("outcome-definition")],
    }
    base.update(overrides)
    return Charter.model_validate(base)


def a_row(**overrides: object) -> PredictorRow:
    base: dict[str, object] = {
        "predictor": "age at index admission",
        "operationalization": "years, from the admission record",
        "timing": "at admission",
        "outcome": "the modeled outcome",
        "evidence_type": EvidenceType.OBSERVATIONAL_ASSOCIATION,
        "effect": 1.82,
        "effect_measure": "adjusted OR",
        "effect_raw": "1.82 (95% CI 1.21-2.74)",
        "ci_low": 1.21,
        "ci_high": 2.74,
        "direction": Direction.INCREASES,
        "confidence": Confidence.HIGH,
        "quote": "Each decade of age carried an adjusted OR of 1.82 (95% CI 1.21-2.74).",
    }
    base.update(overrides)
    return PredictorRow.model_validate(base)


# --- null findings: the invariant that cannot be a prompt instruction -------


def test_an_empty_null_findings_list_becomes_the_sentinel() -> None:
    """A missing section and a section reporting nothing must not render the same."""
    extraction = Extraction(bottom_line="Something predicts something.")

    assert len(extraction.null_findings) == 1
    assert extraction.null_findings[0].predictor == NONE_REPORTED
    assert extraction.null_findings[0].is_sentinel
    assert not extraction.reports_null_findings


def test_the_sentinel_survives_a_json_round_trip() -> None:
    """An extraction parsed straight from a model reply gets it too, not just ours."""
    parsed = parse_model('{"bottom_line": "x", "null_findings": []}', Extraction)
    assert parsed.null_findings[0].predictor == NONE_REPORTED

    reloaded = Extraction.model_validate_json(parsed.model_dump_json())
    assert reloaded.null_findings[0].is_sentinel


def test_real_null_findings_are_kept_and_recognized() -> None:
    extraction = Extraction(
        null_findings=[NullFinding(predictor="a tested factor", detail="no association")]
    )
    assert extraction.reports_null_findings
    assert not extraction.null_findings[0].is_sentinel


# --- vocabulary hints: a concept, and whatever the paper coded it as -------


def test_a_hint_keeps_its_codes_attached_to_the_concept_they_belong_to() -> None:
    """The whole point of the shape.

    Parallel lists of concepts and codes would leave a reader guessing which code went
    with which variable; here the association is the data structure.
    """
    hint = VocabularyHint(
        concept="type 2 diabetes",
        codes=[CodedAs(system="icd10", code="E11.9"), CodedAs(system="snomed", code="44054006")],
    )
    assert hint.concept == "type 2 diabetes"
    assert [(c.system, c.code) for c in hint.codes] == [
        ("icd10", "E11.9"),
        ("snomed", "44054006"),
    ]


def test_a_concept_with_no_codes_is_normal_rather_than_invalid() -> None:
    """Most papers name their variables and code none of them."""
    assert VocabularyHint(concept="frailty").codes == []


def test_a_system_name_is_canonical_however_the_model_spelled_it() -> None:
    """`ICD-10`, `icd 10` and `icd10` are one system, not three."""
    spellings = ["ICD-10", "icd 10", "icd10", "ICD10"]
    assert {CodedAs(system=s, code="E11.9").system for s in spellings} == {"icd10"}


def test_a_code_repeated_under_one_concept_is_recorded_once() -> None:
    hint = VocabularyHint(
        concept="hba1c",
        codes=[
            CodedAs(system="loinc", code="4548-4"),
            CodedAs(system="LOINC", code="4548-4"),
            CodedAs(system="loinc", code="17856-6"),
        ],
    )
    assert [c.code for c in hint.codes] == ["4548-4", "17856-6"]


def test_a_code_cannot_be_recorded_without_a_concept_beside_it() -> None:
    """A bare code is unusable to a reader that thinks in English.

    Enforced by the schema rather than asked for in a prompt, because an extraction
    that answered with codes alone would otherwise produce a file that looks complete.
    """
    with pytest.raises(ValidationError):
        VocabularyHint(concept="", codes=[CodedAs(system="icd10", code="E11.9")])


# --- length budgets: truncate and warn --------------------------------------


def test_an_over_long_description_is_trimmed_on_a_word_boundary() -> None:
    long_text = "word " * 100
    trimmed, warnings = Extraction(description=long_text).enforce_budgets()

    assert len(trimmed.description) <= MAX_DESCRIPTION_CHARS
    assert not trimmed.description.endswith("wor…"), "cuts between words, not inside one"
    assert any("description" in w for w in warnings)


def test_bottom_line_is_cut_to_two_sentences() -> None:
    trimmed, warnings = Extraction(bottom_line="One. Two. Three. Four.").enforce_budgets()

    assert len(sentences(trimmed.bottom_line)) == MAX_BOTTOM_LINE_SENTENCES
    assert trimmed.bottom_line == "One. Two."
    assert any("bottom_line" in w for w in warnings)


def test_dropped_predictor_rows_are_named_in_the_warning() -> None:
    """Silent truncation would look identical to a paper that reported less."""
    rows = [a_row(predictor=f"factor {i}") for i in range(MAX_PREDICTOR_ROWS + 2)]
    trimmed, warnings = Extraction(predictors=rows).enforce_budgets()

    assert len(trimmed.predictors) == MAX_PREDICTOR_ROWS
    dropped_warning = next(w for w in warnings if "predictor row" in w)
    assert f"factor {MAX_PREDICTOR_ROWS}" in dropped_warning
    assert f"factor {MAX_PREDICTOR_ROWS + 1}" in dropped_warning


def test_dropped_null_findings_are_named_in_the_warning() -> None:
    """`null_findings` had no ceiling at all until it was found to be where a reply spent
    the tokens it needed to finish — and a reply that runs out is a lost paper."""
    findings = [NullFinding(predictor=f"null {i}") for i in range(MAX_NULL_FINDINGS + 2)]
    trimmed, warnings = Extraction(null_findings=findings).enforce_budgets()

    assert len(trimmed.null_findings) == MAX_NULL_FINDINGS
    dropped_warning = next(w for w in warnings if "null finding" in w)
    assert f"null {MAX_NULL_FINDINGS}" in dropped_warning
    assert f"null {MAX_NULL_FINDINGS + 1}" in dropped_warning


def test_dropped_vocabulary_hints_are_named_in_the_warning() -> None:
    hints = [VocabularyHint(concept=f"term {i}") for i in range(MAX_VOCABULARY_HINTS + 2)]
    trimmed, warnings = Extraction(vocabulary_hints=hints).enforce_budgets()

    assert len(trimmed.vocabulary_hints) == MAX_VOCABULARY_HINTS
    dropped_warning = next(w for w in warnings if "vocabulary hint" in w)
    assert f"term {MAX_VOCABULARY_HINTS}" in dropped_warning
    assert f"term {MAX_VOCABULARY_HINTS + 1}" in dropped_warning


def test_a_generous_paper_still_fits_under_every_list_ceiling() -> None:
    """The ceilings exist to stop a reply running out of room, not to edit papers. One
    reporting a full slate of findings has to pass through untrimmed, or the budget is
    costing evidence instead of tokens."""
    extraction = Extraction(
        predictors=[a_row(predictor=f"factor {i}") for i in range(MAX_PREDICTOR_ROWS)],
        null_findings=[NullFinding(predictor=f"null {i}") for i in range(MAX_NULL_FINDINGS)],
        vocabulary_hints=[
            VocabularyHint(concept=f"term {i}") for i in range(MAX_VOCABULARY_HINTS)
        ],
    )

    trimmed, warnings = extraction.enforce_budgets()

    assert len(trimmed.predictors) == MAX_PREDICTOR_ROWS
    assert len(trimmed.null_findings) == MAX_NULL_FINDINGS
    assert len(trimmed.vocabulary_hints) == MAX_VOCABULARY_HINTS
    assert not [w for w in warnings if "dropped" in w]


def test_predictor_order_is_the_models_order_not_resorted() -> None:
    """The model's ordering is its judgment of importance; the tail goes, not the middle."""
    rows = [
        a_row(predictor="first", confidence=Confidence.LOW),
        a_row(predictor="second", confidence=Confidence.HIGH),
    ]
    trimmed, _ = Extraction(predictors=rows).enforce_budgets()
    assert [r.predictor for r in trimmed.predictors] == ["first", "second"]


def test_a_within_budget_extraction_produces_no_warnings() -> None:
    trimmed, warnings = Extraction(
        description="Short.", bottom_line="One sentence.", predictors=[a_row()]
    ).enforce_budgets()
    assert warnings == []
    assert trimmed.description == "Short."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two.", 2),
        ("Measured in mg vs. placebo. Then stopped.", 2),
        ("Reported by Smith J. et al. in a cohort.", 1),
        ("No terminator at all", 1),
        ("", 0),
    ],
)
def test_sentence_splitting_handles_the_abbreviations_that_matter(
    text: str, expected: int
) -> None:
    assert len(sentences(text)) == expected


def test_truncation_never_lengthens_short_text() -> None:
    assert truncate_chars("already short", 200) == ("already short", False)


# --- confidence and numeric verification ------------------------------------


def test_confidence_downgrade_saturates_at_low() -> None:
    assert Confidence.HIGH.downgraded is Confidence.MEDIUM
    assert Confidence.MEDIUM.downgraded is Confidence.LOW
    assert Confidence.LOW.downgraded is Confidence.LOW


def test_a_downgraded_row_loses_its_numbers_but_keeps_its_claim() -> None:
    """What step 6 does to a number it cannot find in the source text.

    Dropping the whole row would discard a real reported relationship in order to
    punish one unsupported field.
    """
    original = a_row()
    downgraded = original.downgraded()

    assert downgraded.effect is None
    assert downgraded.ci_low is None and downgraded.ci_high is None
    assert not downgraded.has_effect
    assert downgraded.confidence is Confidence.MEDIUM
    assert downgraded.predictor == original.predictor
    assert downgraded.timing == original.timing
    assert downgraded.effect_raw == original.effect_raw, "the unverified claim is kept, visibly"
    assert original.effect == 1.82, "the original is not mutated"


def test_p_value_stays_a_string() -> None:
    """`<0.001` and `NS` are how papers report this; coercing loses the distinction."""
    assert a_row(p_value="<0.001").p_value == "<0.001"
    assert a_row(p_value="NS").p_value == "NS"


# --- licensing --------------------------------------------------------------


@pytest.mark.parametrize(
    ("license_text", "safe"),
    [
        ("CC BY", True),
        ("cc-by", True),
        ("CC0", True),
        ("CC BY-SA", True),
        ("CC BY-NC", False),
        ("CC BY-NC-ND", False),
        ("CC BY-ND", False),
        ("NO-CC CODE", False),
        ("", False),
        ("all rights reserved", False),
    ],
)
def test_export_safety_is_conservative(license_text: str, safe: bool) -> None:
    """An unknown license is a no. ND forbids derivatives, and this file is one."""
    assert is_export_safe(license_text) is safe


# --- slugs and filenames ----------------------------------------------------


def test_a_topic_slug_must_be_a_slug() -> None:
    """The slug is a directory name and the `domain` frontmatter value at once."""
    with pytest.raises(ValidationError):
        Topic(slug="Risk Factors", title="x")
    with pytest.raises(ValidationError):
        Topic(slug="-leading", title="x")
    assert Topic(slug="risk-factors", title="x").slug == "risk-factors"


def test_slugify_folds_arbitrary_text() -> None:
    assert slugify("Risk Factors & Timing") == "risk-factors-timing"
    assert slugify("!!!") == ""


def test_a_compound_surname_survives_into_the_filename() -> None:
    """The defect from step 2, locked at the other end of the pipeline."""
    record = ConceptRecord(
        pmid="33745404",
        authors=["Ferrari Silva B", "Someone A"],
        domain="risk-factors",
        extraction=Extraction(),
    )
    assert record.first_author_surname == "Ferrari Silva"
    assert record.filename == "33745404_Ferrari-Silva.md"


def test_a_record_with_no_authors_still_gets_a_filename() -> None:
    record = ConceptRecord(pmid="1", domain="risk-factors", extraction=Extraction())
    assert record.filename == "1_Anon.md"


# --- concept record provenance ---------------------------------------------


def test_export_safe_is_computed_from_the_license_not_supplied() -> None:
    record = ConceptRecord(
        pmid="1", domain="risk-factors", license="CC BY", extraction=Extraction()
    )
    assert record.export_safe
    assert "export_safe" not in record.model_dump(), "a property, so a model cannot set it"


def test_an_abstract_only_record_is_never_export_safe() -> None:
    """No license was ever served for it, and unknown is not permission."""
    record = ConceptRecord(
        pmid="1", domain="risk-factors", text_basis=TextBasis.ABSTRACT, extraction=Extraction()
    )
    assert record.license == ""
    assert not record.export_safe


def test_the_writer_and_the_reader_spell_full_text_the_same_way() -> None:
    """The underscore, held from both ends.

    The prose everywhere says "full text"; the value on disk does not. A reader that
    compared against the prose would report an entire corpus as abstract-only and never
    raise, so the two spellings are pinned to each other rather than to a literal.
    """
    assert TextBasis.FULL_TEXT.value == FULL_TEXT_BASIS


def test_sources_are_derived_from_the_identifiers_we_hold() -> None:
    record = ConceptRecord(
        pmid="30035690",
        pmcid="PMC6340782",
        doi="10.1177/1750458918788978",
        domain="risk-factors",
        extraction=Extraction(),
    )
    ids = [s.id for s in record.default_sources()]
    assert ids == ["pmid:30035690", "pmc:PMC6340782", "doi:10.1177/1750458918788978"]
    assert record.default_sources()[0].resource.endswith("/30035690/")


def test_a_record_with_no_pmc_id_gets_no_pmc_source() -> None:
    record = ConceptRecord(pmid="1", domain="risk-factors", extraction=Extraction())
    assert [s.id for s in record.default_sources()] == ["pmid:1"]


def test_extraction_and_provenance_stay_separable() -> None:
    """`record.title` came from an API; `record.extraction.*` came from a model."""
    record = ConceptRecord(
        pmid="1",
        title="From PubMed",
        domain="risk-factors",
        extraction=Extraction(description="From a model"),
    )
    assert "title" not in record.extraction.model_dump()
    assert "description" not in record.model_dump(exclude={"extraction"})


# --- charter ---------------------------------------------------------------


def test_languages_are_lowercased_and_stripped() -> None:
    assert a_charter(languages=[" ENG ", "fre", ""]).languages == ["eng", "fre"]


def test_duplicate_topic_slugs_are_rejected() -> None:
    """They are directory names; the second would silently overwrite the first."""
    with pytest.raises(ValidationError, match="unique"):
        a_charter(topic_taxonomy=[a_topic(), a_topic()])


def test_an_inverted_topic_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="topic_min"):
        a_charter(topic_min=20, topic_max=10)


def test_a_charter_round_trips_through_yaml() -> None:
    """The file is meant to be edited by hand between the pause and the build."""
    original = a_charter(min_year=2010, target_papers=120)
    restored = Charter.from_yaml(original.to_yaml())

    assert restored == original
    assert restored.slugs == original.slugs


def test_charter_yaml_keeps_declaration_order() -> None:
    lines = a_charter().to_yaml().splitlines()
    keys = [line.split(":")[0] for line in lines if line and not line.startswith((" ", "-"))]
    assert keys.index("prompt") < keys.index("topic_taxonomy") < keys.index("languages")
    assert keys.index("topic_min") < keys.index("topic_max")


def test_the_charter_digest_ignores_when_it_was_generated() -> None:
    """Two runs from the same edited charter must agree."""
    first = a_charter(generated_at=datetime(2026, 1, 1, tzinfo=UTC), generated_by="a")
    second = a_charter(generated_at=datetime(2026, 8, 3, tzinfo=UTC), generated_by="b")
    assert first.digest() == second.digest()

    assert a_charter(target_papers=42).digest() != first.digest()


def test_a_target_the_taxonomy_cannot_hold_is_reported() -> None:
    problems = a_charter(target_papers=500, topic_max=10).problems()
    assert any("exceeds what the taxonomy can hold" in p for p in problems)


def test_a_workable_charter_has_no_problems() -> None:
    assert a_charter(target_papers=40, topic_min=8, topic_max=40).problems() == []


def test_capacity_reflects_the_taxonomy_size() -> None:
    charter = a_charter(topic_min=5, topic_max=25)
    assert charter.capacity() == (10, 50)


# --- screening and curation -------------------------------------------------


def test_a_near_miss_is_flagged_borderline_for_reconsideration() -> None:
    """Cheaper and better-informed than another search round when a topic is thin."""
    assert ScreenVerdict(pmid="1", include=False, relevance=2).borderline
    assert not ScreenVerdict(pmid="2", include=False, relevance=0).borderline
    assert ScreenVerdict(
        pmid="3", include=True, relevance=3, confidence=Confidence.LOW
    ).borderline


def test_relevance_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ScreenVerdict(pmid="1", include=True, relevance=7)


def test_curation_groups_kept_papers_by_topic() -> None:
    result = CurationResult(
        decisions=[
            CurationDecision(pmid="1", keep=True, topic="risk-factors"),
            CurationDecision(pmid="2", keep=False),
            CurationDecision(pmid="3", keep=True, topic="risk-factors"),
        ]
    )
    assert result.by_topic() == {"risk-factors": ["1", "3"]}
    assert len(result.kept) == 2


def test_a_topic_under_its_floor_drives_the_requery_edge() -> None:
    result = CurationResult(gaps=[TopicGap(topic="risk-factors", kept=3, floor=8)])
    assert result.gaps[0].shortfall == 5
    assert result.needs_more_search

    filled = CurationResult(gaps=[TopicGap(topic="risk-factors", kept=9, floor=8)])
    assert not filled.needs_more_search


# --- candidates -------------------------------------------------------------


async def test_a_candidate_is_built_from_a_real_pubmed_record(
    replay_clients: Clients,
) -> None:
    fetched = await replay_clients.eutils.efetch(["9500320", "20301425", "33745404"])
    records = {r.pmid: r for r in fetched}
    candidate = Candidate.from_record(records["33745404"], found_by="a query", rank=4)

    assert candidate.pmid == "33745404"
    assert candidate.authors[0] == "Ferrari Silva B", "the compound surname survives"
    assert candidate.has_abstract
    assert candidate.found_by == ["a query"]
    assert candidate.best_rank == 4
    assert set(candidate.mesh_major) <= set(candidate.mesh_terms)
    assert candidate.citation().startswith("Ferrari Silva B et al.")


def test_merging_two_sightings_unions_provenance_and_keeps_the_best_rank() -> None:
    """Found by four independent queries is itself a ranking signal."""
    first = Candidate(pmid="1", title="T", found_by=["q1"], best_rank=9)
    second = Candidate(pmid="1", title="T", abstract="filled in", found_by=["q2"], best_rank=2)

    merged = first.merged_with(second)

    assert merged.found_by == ["q1", "q2"]
    assert merged.best_rank == 2
    assert merged.abstract == "filled in"
    assert first.best_rank == 9, "inputs are not mutated"


def test_merging_is_idempotent_on_provenance() -> None:
    candidate = Candidate(pmid="1", found_by=["q1"])
    assert candidate.merged_with(candidate).found_by == ["q1"]


def test_screening_text_excludes_mesh_terms() -> None:
    """MeSH is assigned months after publication, so including it favors old papers."""
    candidate = Candidate(
        pmid="1", title="A title", abstract="An abstract.", mesh_terms=["Indexed Term"]
    )
    assert "Indexed Term" not in candidate.screening_text
    assert candidate.screening_text == "A title\n\nAn abstract."


def test_normalized_title_catches_the_same_study_under_two_pmids() -> None:
    """Preprint and journal version, or a corrected reprint, carry different PMIDs."""
    first = Candidate(pmid="1", title="A Study: Of Things (Preliminary)")
    second = Candidate(pmid="2", title="A study of things preliminary")
    assert first.normalized_title == second.normalized_title


def test_citation_renders_without_a_year() -> None:
    candidate = Candidate(pmid="1", authors=["Smith J", "Jones A"], journal_abbrev="J Test")
    assert candidate.citation() == "Smith J et al. J Test n.d."


# --- manifest ---------------------------------------------------------------


def test_a_manifest_cannot_record_a_dollar_amount_for_an_all_unpriced_run() -> None:
    """The `$0.00` trap, enforced at the point of persistence.

    LiteLLM returns 0.0 for a model it cannot price, so free and unknown arrive as the
    same float. A manifest that stores the first when it meant the second reads as good
    news forever after.
    """
    with pytest.raises(ValidationError, match="unpriced"):
        CostSummary(calls=3, unpriced_calls=3, usd=0.0, display="$0.00")


def test_a_manifest_cannot_hide_partly_unpriced_calls() -> None:
    with pytest.raises(ValidationError, match="unpriced"):
        CostSummary(calls=5, unpriced_calls=2, usd=1.23, display="$1.2300")

    mixed = CostSummary(calls=5, unpriced_calls=2, usd=1.23, display="$1.2300 + 2 unpriced")
    assert mixed.tokens == 0


def test_zero_dollars_is_allowed_only_when_nothing_was_called() -> None:
    summary = CostSummary(calls=0, unpriced_calls=0, display="$0.00")
    assert summary.display == "$0.00"


def test_an_all_unpriced_run_records_the_honest_string() -> None:
    summary = CostSummary(calls=3, unpriced_calls=3, display="cost unavailable")
    assert summary.display == "cost unavailable"


def test_staleness_is_derived_from_the_build_date() -> None:
    manifest = RunManifest(run_id="r1").with_staleness(built_on=date(2026, 8, 3), days=180)
    assert manifest.stale_after == date(2027, 1, 30)
    assert RunManifest(run_id="r1").stale_after is None


def test_manifest_duration_needs_both_ends() -> None:
    started = datetime(2026, 8, 3, 10, 0, tzinfo=UTC)
    assert RunManifest(run_id="r", started_at=started).duration_seconds is None
    finished = datetime(2026, 8, 3, 10, 2, tzinfo=UTC)
    manifest = RunManifest(run_id="r", started_at=started, finished_at=finished)
    assert manifest.duration_seconds == 120.0


# --- parsing model replies --------------------------------------------------


def test_a_fenced_reply_parses() -> None:
    reply = '```json\n{"pmid": "1", "include": true}\n```'
    assert parse_model(reply, ScreenVerdict).pmid == "1"


def test_a_reply_with_preamble_and_trailing_prose_parses() -> None:
    reply = 'Here is my judgment:\n{"pmid": "1", "include": false}\nLet me know if that helps.'
    assert parse_model(reply, ScreenVerdict).include is False


def test_a_brace_inside_a_quoted_value_does_not_end_the_object() -> None:
    """Verbatim source quotes contain braces often enough for this to matter."""
    reply = '{"pmid": "1", "include": true, "reason": "the set {a, b} was tested"}'
    assert parse_model(reply, ScreenVerdict).reason == "the set {a, b} was tested"


def test_a_trailing_comma_is_repaired_rather_than_retried() -> None:
    """A retry would cost a whole call to fix punctuation."""
    assert parse_model('{"pmid": "1", "include": true,}', ScreenVerdict).pmid == "1"


def test_an_extra_key_is_ignored_not_fatal() -> None:
    reply = '{"pmid": "1", "include": true, "confidence_score": 0.9}'
    assert parse_model(reply, ScreenVerdict).pmid == "1"


def test_a_missing_required_field_raises_with_a_field_level_hint() -> None:
    with pytest.raises(SchemaError) as caught:
        parse_model('{"include": true}', ScreenVerdict)
    assert "pmid" in caught.value.hint
    assert "JSON" in caught.value.hint


def test_a_reply_with_no_json_raises() -> None:
    with pytest.raises(SchemaError, match="no JSON"):
        parse_model("I could not answer that.", ScreenVerdict)


def test_an_unterminated_object_raises_rather_than_parsing_a_prefix() -> None:
    with pytest.raises(SchemaError, match="unterminated"):
        extract_json('{"pmid": "1", "include": tru')


def test_extract_json_finds_the_first_complete_value() -> None:
    assert extract_json('noise {"a": 1} more {"b": 2}') == '{"a": 1}'


def test_a_nested_extraction_reply_parses_end_to_end() -> None:
    reply = """```json
    {
      "description": "One line about the paper.",
      "bottom_line": "The finding.",
      "study_design": "retrospective cohort",
      "n": 1454,
      "predictors": [
        {"predictor": "a factor", "evidence_type": "randomized_intervention",
         "effect": 1.5, "effect_raw": "1.5 (1.1-2.0)", "direction": "increases",
         "confidence": "high"}
      ],
      "null_findings": [],
      "vocabulary_hints": [{"concept": "a variable", "codes": []}]
    }
    ```"""
    extraction = parse_model(reply, Extraction)

    assert extraction.n == 1454
    assert extraction.predictors[0].evidence_type is EvidenceType.RANDOMIZED_INTERVENTION
    assert extraction.predictors[0].confidence is Confidence.HIGH
    assert extraction.null_findings[0].is_sentinel, "the invariant survives the parse"


def test_evidence_types_serialize_as_their_string_value() -> None:
    """A downstream reader parses the bundle without importing this package."""
    dumped = a_row().model_dump(mode="json")
    assert dumped["evidence_type"] == "observational_association"
    assert dumped["direction"] == "increases"
    assert dumped["confidence"] == "high"


def test_source_refs_carry_optional_usage_counts() -> None:
    ref = SourceRef(id="pmid:1", resource="https://example.org/1")
    assert ref.usage_count is None
