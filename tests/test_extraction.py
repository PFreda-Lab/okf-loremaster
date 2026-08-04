"""`fulltext -> extract -> reconcile`, one node at a time.

The end-to-end behavior is `test_verification.py`'s subject; this is the machinery under
it, on the paths a healthy run never takes. What each node has to get right is different
in kind:

- `fulltext` decides what the extractor will be shown, and therefore what verification
  will later check against. A budget applied to the wrong string, sections handed over
  shuffled, or a license inferred rather than read are all invisible until much later.
- `extract` is the most expensive call in the pipeline, so what it does with a reply it
  cannot parse — repair once, then drop the paper and say so — is worth pinning exactly.
- `reconcile` runs four deterministic steps in an order that matters, and drops papers
  from topics that curation worked to fill. Silence there would be a lie about the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from okf_loremaster.graph.nodes import extract_node, fulltext_node, reconcile_node
from okf_loremaster.graph.state import RunState
from okf_loremaster.schemas import (
    MAX_PREDICTOR_ROWS,
    Candidate,
    Extraction,
    PaperText,
    TextBasis,
)

from fake_llm import ExtractFn, ScriptedLLM, extraction, row, supported
from fake_ncbi import (
    REFERENCE_ONLY,
    FakeNCBI,
    abstract_for,
    finding_sentence,
    has_full_text,
    license_for,
    pmcid_for,
    sample_size,
    title_for,
)
from graph_runs import charter_for, node_deps

# One open-access paper and one PMC declines to serve. Both from the `alpha` topic, so a
# charter topic slug is the same word in every state below.
OPEN = "10000"
CLOSED = "10002"

# What `_select` charges for a section: its text, its type name, and the `## ` heading
# with the blank line after it. Restated here so the budgets below are arithmetic a
# reader can check rather than numbers somebody once observed passing.
TITLE_ABSTRACT_RESULTS_TABLE = 350
METHODS_COST = 120
DISCUSS_COST = 65


def candidate_for(pmid: str) -> Candidate:
    return Candidate(
        pmid=pmid,
        title=title_for(pmid),
        abstract=abstract_for(pmid),
        journal_abbrev="J Alpha Stud",
        year=2018,
        authors=["Author00 A"],
        pmcid=pmcid_for(pmid),
    )


def retrieval_state(*pmids: str) -> RunState:
    return {
        "charter": charter_for(),
        "topics": {"alpha": list(pmids)},
        "unique": [candidate_for(pmid) for pmid in pmids],
    }


def budget(monkeypatch: pytest.MonkeyPatch, chars: int) -> None:
    monkeypatch.setattr("okf_loremaster.graph.nodes.fulltext.MAX_SOURCE_CHARS", chars)


# --- fulltext ---------------------------------------------------------------


async def test_the_budget_keeps_the_sections_the_evidence_is_actually_in(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Priority order is about where a paper puts its numbers, not about page order.

    Under a budget that fits four sections, the four kept are the title, the abstract,
    the results and the table caption — and the introduction, which mostly restates other
    people's papers, is the first thing gone.
    """
    budget(monkeypatch, TITLE_ABSTRACT_RESULTS_TABLE + 5)

    async with node_deps(settings_factory, tmp_path) as deps:
        update = await fulltext_node(retrieval_state(OPEN), deps)

    source = update["texts"][OPEN]
    assert source.sections == ["TITLE", "ABSTRACT", "RESULTS", "TABLE"]
    assert source.truncated
    assert source.basis is TextBasis.FULL_TEXT
    assert finding_sentence(OPEN) in source.text


async def test_a_section_too_large_to_fit_is_skipped_rather_than_ending_the_selection(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One enormous methods section must not cost a paper everything below it.

    The budget here has room for the discussion but not for the methods that outrank it.
    A loop that stopped at the first section that did not fit would keep neither.
    """
    budget(monkeypatch, TITLE_ABSTRACT_RESULTS_TABLE + DISCUSS_COST)
    assert METHODS_COST > DISCUSS_COST  # the arithmetic the test rests on

    async with node_deps(settings_factory, tmp_path) as deps:
        update = await fulltext_node(retrieval_state(OPEN), deps)

    source = update["texts"][OPEN]
    assert "METHODS" not in source.sections
    assert "DISCUSS" in source.sections
    # And reading order is restored: the discussion follows the results it discusses,
    # rather than arriving wherever priority happened to put it.
    assert source.sections == ["TITLE", "ABSTRACT", "RESULTS", "TABLE", "DISCUSS"]


async def test_the_reference_list_never_reaches_the_extractor(
    settings_factory: Any, tmp_path: Path
) -> None:
    """A number in a cited title belongs to the cited study. Sending it to the extractor
    is how one paper's effect size ends up filed under another paper's PMID."""
    async with node_deps(settings_factory, tmp_path) as deps:
        update = await fulltext_node(retrieval_state(OPEN), deps)

    source = update["texts"][OPEN]
    assert "REF" not in source.sections
    assert str(REFERENCE_ONLY) not in source.text
    assert not source.truncated  # the real budget fits this paper whole


async def test_a_paper_pmc_will_not_serve_falls_back_to_its_abstract(
    settings_factory: Any, tmp_path: Path
) -> None:
    """Most of any corpus is not open access, so this is the ordinary path, not the sad
    one. The license stays empty because none was ever served to us."""
    fake = FakeNCBI()
    assert not has_full_text(CLOSED)

    async with node_deps(settings_factory, tmp_path, fake=fake) as deps:
        update = await fulltext_node(retrieval_state(CLOSED), deps)

    source = update["texts"][CLOSED]
    assert source.basis is TextBasis.ABSTRACT
    assert source.license == ""
    assert source.sections == []
    assert abstract_for(CLOSED) in source.text
    # It was asked for. A fallback that skipped the request would look identical here
    # and would quietly stop reading full text for the whole corpus.
    assert fake.bioc_requests == [pmcid_for(CLOSED)]
    assert not update["warnings"]  # not being open access is not a failure


async def test_the_license_is_recorded_verbatim_and_never_inferred(
    settings_factory: Any, tmp_path: Path
) -> None:
    """Whatever BioC said, character for character. `export_safe` is computed from this
    later, so a tidied-up license string is a redistribution decision made by accident."""
    async with node_deps(settings_factory, tmp_path) as deps:
        update = await fulltext_node(retrieval_state(OPEN), deps)

    source = update["texts"][OPEN]
    assert source.license == license_for(OPEN)
    assert source.pmcid == pmcid_for(OPEN)
    assert f"license {license_for(OPEN)}" in source.text


async def test_a_paper_already_read_is_not_fetched_again(
    settings_factory: Any, tmp_path: Path
) -> None:
    """Retrieval is the slow half of this node and the answer does not change between
    rounds, so a re-queried or resumed run pays for the papers it has not read."""
    fake = FakeNCBI()
    state = retrieval_state(OPEN, CLOSED)
    state["texts"] = {OPEN: PaperText(pmid=OPEN, text="already read")}

    async with node_deps(settings_factory, tmp_path, fake=fake) as deps:
        update = await fulltext_node(state, deps)

    assert fake.bioc_requests == [pmcid_for(CLOSED)]
    assert update["texts"][OPEN].text == "already read"


# --- extract ----------------------------------------------------------------


def extract_state(*pmids: str, text: str | None = None) -> RunState:
    """A state that starts where `fulltext` left off."""
    return {
        "charter": charter_for(),
        "topics": {"alpha": list(pmids)},
        "unique": [candidate_for(pmid) for pmid in pmids],
        "texts": {
            pmid: PaperText(
                pmid=pmid,
                basis=TextBasis.FULL_TEXT,
                license=license_for(pmid),
                pmcid=pmcid_for(pmid),
                text=text
                if text is not None
                else f"Title: {title_for(pmid)}\n\n"
                f"## ABSTRACT\n{abstract_for(pmid)}\n\n"
                f"## RESULTS\n{finding_sentence(pmid)}",
            )
            for pmid in pmids
        },
    }


UNPARSEABLE: dict[str, Any] = {"predictors": "one row, in prose"}


def failing(times: int) -> ExtractFn:
    """An extractor whose first `times` replies do not satisfy the schema."""
    attempts = 0

    def extract(source: str) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        return UNPARSEABLE if attempts <= times else supported(source)

    return extract


def scripted(extract: ExtractFn) -> ScriptedLLM:
    def unreachable(*_: Any) -> dict[str, Any]:
        raise AssertionError("these tests drive `extract` alone")

    return ScriptedLLM(screen=unreachable, curate=unreachable, extract=extract)


async def test_a_reply_that_does_not_parse_is_repaired_once_and_the_paper_survives(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The retry exists because `SchemaError.hint` names the field that was wrong, and a
    model told which field it broke usually fixes it — cheaper than losing the paper."""
    model = scripted(failing(1))

    async with node_deps(settings_factory, tmp_path, scripted=model) as deps:
        update = await extract_node(extract_state(OPEN), deps)

    assert len(model.extracted) == 2
    assert update["extractions"][OPEN].predictors[0].effect is not None
    assert not update["warnings"]


async def test_a_reply_that_never_parses_drops_the_paper_after_exactly_one_retry(
    settings_factory: Any, tmp_path: Path
) -> None:
    """Once, not twice. A second failure is a model that cannot satisfy the schema, and
    a third reasoning-tier call to confirm that is the most expensive way to learn nothing."""
    model = scripted(failing(99))

    async with node_deps(settings_factory, tmp_path, scripted=model) as deps:
        update = await extract_node(extract_state(OPEN), deps)

    assert len(model.extracted) == 2
    assert update["extractions"] == {}

    warnings = update["warnings"]
    assert any(OPEN in note and "did not parse" in note for note in warnings)
    # Every paper failed, so the bundle is not a reading of the literature and the run
    # says so in as many words rather than emitting a confident-looking empty topic.
    assert any("more than half the extractions failed" in note for note in warnings)


async def test_with_no_model_nothing_is_extracted_and_the_run_says_why(
    settings_factory: Any, tmp_path: Path
) -> None:
    async with node_deps(settings_factory, tmp_path) as deps:
        update = await extract_node(extract_state(OPEN), deps)

    assert update["extractions"] == {}
    assert any("no model is available" in note for note in update["warnings"])


async def test_the_model_is_shown_exactly_the_string_verification_will_check_against(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The invariant the whole numeric check rests on. If `extract` rebuilt, re-budgeted
    or decorated the source, verification would be checking against a string the model
    never saw — and correct extractions would start reporting as fabricated."""
    model = scripted(supported)
    state = extract_state(OPEN)

    async with node_deps(settings_factory, tmp_path, scripted=model) as deps:
        await extract_node(state, deps)

    assert model.extracted == [(state["texts"] or {})[OPEN].text.strip()]


async def test_a_paper_already_extracted_is_not_read_again(
    settings_factory: Any, tmp_path: Path
) -> None:
    model = scripted(supported)
    state = extract_state(OPEN, CLOSED)
    state["extractions"] = {OPEN: Extraction.model_validate(extraction())}

    async with node_deps(settings_factory, tmp_path, scripted=model) as deps:
        update = await extract_node(state, deps)

    assert len(model.extracted) == 1
    assert set(update["extractions"]) == {OPEN, CLOSED}


# --- reconcile --------------------------------------------------------------


def reconcile_state(extractions: dict[str, Extraction], **overrides: Any) -> RunState:
    pmids = list(extractions)
    state = extract_state(*pmids)
    state["extractions"] = extractions
    for key, value in overrides.items():
        state[key] = value  # type: ignore[literal-required]
    return state


def read_of(pmid: str, **fields: Any) -> Extraction:
    """An extraction of `pmid` whose numbers are the ones that paper actually printed."""
    faithful = supported(finding_sentence(pmid) + f" A cohort of {sample_size(pmid)} adults.")
    return Extraction.model_validate({**faithful, **fields})


async def test_length_budgets_are_applied_before_verification_not_after(
    settings_factory: Any, tmp_path: Path
) -> None:
    """Order matters twice over: nothing is checked that a budget was about to drop, and
    the verification counts describe the bundle rather than the model's reply."""
    over = read_of(
        OPEN, predictors=[row(predictor=f"exposure {index}") for index in range(20)]
    )

    async with node_deps(settings_factory, tmp_path) as deps:
        update = await reconcile_node(reconcile_state({OPEN: over}), deps)

    assert len(update["records"][0].extraction.predictors) == MAX_PREDICTOR_ROWS
    # The count describes the bundle. Verified first, it would have been 20 — a number
    # for rows that no reader of the bundle can ever see.
    assert update["verification"].rows == MAX_PREDICTOR_ROWS
    assert any("ran over a length budget" in note for note in update["warnings"])


async def test_a_vocabulary_key_the_charter_did_not_list_is_kept_out_of_frontmatter(
    settings_factory: Any, tmp_path: Path
) -> None:
    """`unmapped_vocab` is the escape hatch for the one call made before any paper was
    read. It is recorded and reported, and it never reaches a frontmatter block."""
    hinting = read_of(OPEN, vocabulary_hints={"icd10": ["A00"], "snomed": ["44054006"]})

    async with node_deps(settings_factory, tmp_path) as deps:
        update = await reconcile_node(reconcile_state({OPEN: hinting}), deps)

    record = update["records"][0]
    assert record.extraction.vocabulary_hints == {"icd10": ["A00"]}
    assert record.unmapped_vocab == {"snomed": ["44054006"]}
    assert any("snomed" in note and "did not list" in note for note in update["warnings"])


async def test_a_paper_with_no_extraction_leaves_its_topic_and_is_named(
    settings_factory: Any, tmp_path: Path
) -> None:
    """A topic entry with no file behind it is a broken link in the emitted bundle. The
    shortfall this causes is ours rather than the literature's, and the warning says so."""
    state = extract_state(OPEN, CLOSED)
    state["extractions"] = {OPEN: read_of(OPEN)}

    async with node_deps(settings_factory, tmp_path) as deps:
        update = await reconcile_node(state, deps)

    assert update["topics"] == {"alpha": [OPEN]}
    assert [record.pmid for record in update["records"]] == [OPEN]
    assert any(CLOSED in note and "no usable extraction" in note for note in update["warnings"])


async def test_provenance_names_the_model_that_read_the_paper_and_the_ids_it_read_from(
    settings_factory: Any, tmp_path: Path
) -> None:
    model = scripted(supported)

    async with node_deps(settings_factory, tmp_path, scripted=model) as deps:
        update = await reconcile_node(reconcile_state({OPEN: read_of(OPEN)}), deps)

    record = update["records"][0]
    assert record.generated_by == "okf-loremaster/extract/fake/deep"
    assert [ref.id for ref in record.sources] == [f"pmid:{OPEN}", f"pmc:{pmcid_for(OPEN)}"]
    assert record.domain == "alpha"
    assert record.license == license_for(OPEN)
    assert record.text_basis is TextBasis.FULL_TEXT


async def test_an_extraction_with_no_stored_source_is_checked_against_nothing_and_fails(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The graph makes this impossible, which is exactly why it is worth pinning: an
    unchecked extraction must not pass as a checked one just because the check had
    nothing to run against."""
    state = reconcile_state({OPEN: read_of(OPEN)})
    state["texts"] = {}

    async with node_deps(settings_factory, tmp_path) as deps:
        update = await reconcile_node(state, deps)

    assert update["records"][0].extraction.predictors[0].effect is None
    assert update["verification"].effects_dropped == 1
