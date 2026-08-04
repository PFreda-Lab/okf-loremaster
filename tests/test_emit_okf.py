"""The bundle a finished run leaves on disk, read back the way a consumer will read it.

The step 7 gate is `test_the_golden_bundle_passes_the_hard_gate`: one whole run over the
synthetic corpus, emitted, then validated from the files rather than from the objects
that wrote them. Everything else in this module is a property of those same files, so
the suite is asking about one bundle from several directions rather than building a new
one per assertion.

Nothing here reads a `ConceptRecord` to decide whether the bundle is right. The gap
between what the pipeline believes and what it wrote is exactly the gap the emitter can
fall into, and a test that closed it by looking at run state would never notice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from okf_loremaster.okf.frontmatter import load, parse, split
from okf_loremaster.okf.layout import (
    BODY_SECTIONS,
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    INDEX_FILENAME,
    LOG_FILENAME,
    UNVERIFIED_CELL,
)
from okf_loremaster.okf.reader import read_bundle
from okf_loremaster.okf.validate import validate_bundle
from okf_loremaster.review import HUMAN_PREFIX, Signoff
from okf_loremaster.schemas import ConceptRecord, VerificationSummary
from test_verification import fabricating

from fake_ncbi import TOPICS
from graph_runs import TARGET, Run, full_run, scripted_run

# What a flat frontmatter value may open with once the emitter has had its way: a quote
# for a scalar, a bracket or brace for the nested blocks OKF v0.2 requires.
QUOTED_OR_FLOW = ('"', "[", "{")


async def golden(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> tuple[Run, Path]:
    run = await full_run(settings_factory, tmp_path, monkeypatch, **overrides)
    return run, Path(run.state["bundle"])


# --- the gate ---------------------------------------------------------------


async def test_the_golden_bundle_passes_the_hard_gate(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 7 gate. A run that emitted a bundle its own validator rejects has
    shipped a contract violation to the only consumer that will ever notice."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    report = validate_bundle(bundle)
    assert report.errors == (), report.lines()
    assert report.ok
    assert report.documents == TARGET
    assert report.shelves == len(TOPICS)

    # And the run reached the same verdict in-graph, over the same code path.
    assert run.state["validated"] is True
    assert run.state["validation_errors"] == []


async def test_the_bundle_has_every_file_a_consumer_is_promised(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for filename in (
        INDEX_FILENAME,
        CATALOG_FILENAME,
        DESCRIPTOR_FILENAME,
        LOG_FILENAME,
        CHARTER_FILENAME,
    ):
        assert (bundle / filename).is_file(), filename

    for slug in TOPICS:
        assert (bundle / slug / INDEX_FILENAME).is_file(), slug

    # `<pmid>_<Author>.md`, which is the shape AFCE's three-way resolver expects.
    documents = list(read_bundle(bundle).documents())
    assert len(documents) == TARGET
    for document in documents:
        assert document.filename == f"{document.pmid}_{document.filename.split('_', 1)[1]}"
        assert document.filename.endswith(".md")


# --- frontmatter ------------------------------------------------------------


async def test_every_flat_value_is_quoted_and_every_nested_one_is_flow_style(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The downstream parser is dependency-free and hands back strings. A bare `n: 200`
    is a number to one reader and a string to the other, which is the ambiguity the
    whole discipline exists to remove — so it is checked on the bytes, not the model."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for path in sorted(bundle.rglob("*.md")):
        block, _ = split(path.read_text(encoding="utf-8"))
        for line in block.splitlines():
            if not line.strip():
                continue
            _key, _, raw = line.partition(":")
            assert raw.strip().startswith(QUOTED_OR_FLOW), f"{path.name}: {line}"


async def test_both_parsers_see_the_same_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One key per line for a line-parser, valid YAML for a spec consumer. A block that
    means two different things depending on who opens it is the failure flow style
    prevents, and only reading it twice can prove it did not happen."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for path in sorted(bundle.rglob("*.md")):
        block, _ = split(path.read_text(encoding="utf-8"))
        strict = parse(block)
        loaded = yaml.safe_load(block)
        assert strict == loaded, path.name
        # And the nested blocks really did survive as structures rather than as text.
        if "generated" in strict:
            assert isinstance(strict["generated"], dict)
            assert set(strict["generated"]) == {"by", "at"}


async def test_the_key_is_domain_it_equals_the_folder_and_shelf_never_appears(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Shelf" is the human word. A file carrying both keys is shelved by whichever one
    the reader happens to prefer, which is a bundle that reorganizes itself per tool."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)
    parsed = read_bundle(bundle)

    for document in parsed.documents():
        assert document.domain == document.path.parent.name
        assert "shelf" not in document.fields
    for shelf in parsed.shelves:
        assert shelf.index is not None
        assert shelf.index.domain == shelf.slug
        assert "shelf" not in shelf.index.fields

    # The root sits in no domain folder, so it must not claim one.
    assert parsed.index is not None
    assert "domain" not in parsed.index.fields


async def test_the_required_pair_is_present_on_every_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for document in read_bundle(bundle).documents():
        assert document.title
        assert document.domain
        # Optional, but the search surface is title + tags + journal and an untagged
        # document is findable by its title alone.
        assert document.tags


# --- the body ---------------------------------------------------------------


async def test_the_five_sections_are_always_present_in_order(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for document in read_bundle(bundle).documents():
        headings = [name for name, _ in document.sections()]
        assert headings == list(BODY_SECTIONS), document.path.name
        for _name, text in document.sections():
            assert text.strip(), document.path.name


async def test_a_paper_with_no_null_finding_says_so_rather_than_omitting_the_section(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant, checked where it is actually consumed. An absent section and an
    absent finding are different claims, and a downstream agent cannot tell them apart
    once the section is gone."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    empties = 0
    for document in read_bundle(bundle).documents():
        section = document.section("Null or non-significant findings")
        assert section is not None
        assert section.strip()
        # The sentinel is rendered as prose, never as a row reading "none reported".
        if "None reported" in section:
            empties += 1
            assert "|" not in section
    assert empties, "no paper exercised the sentinel, so nothing was proved"


async def test_the_quotes_are_reproduced_verbatim_beside_the_row_they_belong_to(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Numbered against the table's `#` column. A quote list that did not key back to a
    row would be a paragraph of sentences with nothing to attach them to."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)
    by_pmid = {record.pmid: record for record in run.records}

    checked = 0
    for document in read_bundle(bundle).documents():
        record = by_pmid[document.pmid]
        section = document.section("Predictors reported") or ""
        for index, row in enumerate(record.extraction.predictors, start=1):
            if not row.quote:
                continue
            assert f"{index}. {row.quote}" in section, document.path.name
            checked += 1
    assert checked


async def test_an_effect_verification_removed_is_marked_rather_than_printed(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The number the extractor invented must not reappear in the emitted table.

    `downgraded()` keeps `effect_raw` so the paper trail survives, which means the
    fabricated string is still on the row when the emitter reaches it. Rendering it
    would undo step 6 in the one artifact anybody reads.
    """
    run, bundle = await golden(
        settings_factory, tmp_path, monkeypatch, scripted=scripted_run(extract=fabricating)
    )
    summary: VerificationSummary | None = run.state["verification"]
    assert summary is not None and summary.effects_dropped == TARGET

    for document in read_bundle(bundle).documents():
        section = document.section("Predictors reported") or ""
        assert "4.44" not in section, document.path.name
        assert UNVERIFIED_CELL in section, document.path.name

    # And the bundle is still a valid bundle: a dropped number is a downgrade, not a
    # structural failure.
    assert validate_bundle(bundle).ok


# --- the catalog ------------------------------------------------------------


async def test_the_catalog_names_every_document_and_nothing_else(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    rows = [
        json.loads(line)
        for line in (bundle / CATALOG_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == TARGET

    for row in rows:
        assert (bundle / row["file"]).is_file()
        assert row["domain"] == Path(row["file"]).parent.name
        for key in ("pmid", "title", "domain", "description", "design", "n", "tags"):
            assert key in row, key


# --- sign-off ---------------------------------------------------------------


class StubReviewer:
    """A sign-off surface with nobody behind it. Records what it was shown."""

    def __init__(self, signoff: Signoff) -> None:
        self.signoff = signoff
        self.shown: list[ConceptRecord] = []

    async def sign_off(
        self,
        records: Any,
        *,
        shelves: dict[str, list[str]],
        verification: VerificationSummary | None,
        warnings: Any,
    ) -> Signoff:
        self.shown = list(records)
        return self.signoff


def attach(monkeypatch: pytest.MonkeyPatch, signoff: Signoff) -> StubReviewer:
    """Stand in for the console reviewer `--review` would have built."""
    stub = StubReviewer(signoff)

    def build(signer: str, console: Any = None) -> StubReviewer:
        return stub

    monkeypatch.setattr("okf_loremaster.ui.review.ConsoleReviewer", build)
    return stub


async def test_without_review_no_document_claims_to_have_been_verified(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default, and the honest one. `unverified` is the tier a machine extraction
    has earned, and the way OKF spells it is by the block not being there."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    assert run.state.get("verified_by", "") == ""
    for document in read_bundle(bundle).documents():
        assert "verified" not in document.fields
    # Nothing was asked, so nothing is reported as a shortfall.
    assert not [note for note in run.warnings if "sign-off" in note]


async def test_review_signs_every_document_with_a_named_human(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 7 gate for `--review`: `verified: [{by: "human:<id>", ...}]`, on disk."""
    stub = attach(monkeypatch, Signoff.granted("human:tester"))
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch, review=True)

    assert len(stub.shown) == TARGET
    assert run.state["verified_by"] == "human:tester"

    for document in read_bundle(bundle).documents():
        entries = document.fields["verified"]
        assert isinstance(entries, list) and len(entries) == 1
        assert entries[0]["by"] == "human:tester"
        assert entries[0]["by"].startswith(HUMAN_PREFIX)
        assert entries[0]["at"].endswith("Z")

    # And it is still one key on one line, nested value and all.
    sample = next(iter(read_bundle(bundle).documents())).path
    block, _ = split(sample.read_text(encoding="utf-8"))
    verified = [line for line in block.splitlines() if line.startswith("verified:")]
    assert len(verified) == 1
    assert verified[0].endswith("}]")

    assert validate_bundle(bundle).ok


async def test_a_declined_sign_off_still_emits_the_bundle_unverified(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining is not a failure. Refusing to emit would punish the reviewer for
    reading carefully, and the files are simply written at the tier they earned."""
    attach(monkeypatch, Signoff.declined("the delta shelf is thin"))
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch, review=True)

    assert run.state.get("verified_by", "") == ""
    documents = list(read_bundle(bundle).documents())
    assert len(documents) == TARGET
    for document in documents:
        assert "verified" not in document.fields

    declined = [note for note in run.warnings if "sign-off was not given" in note]
    assert len(declined) == 1
    assert "the delta shelf is thin" in declined[0]
    assert validate_bundle(bundle).ok


# --- the manifest -----------------------------------------------------------


async def test_the_bundle_can_say_what_it_was_built_from_without_the_tool(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)
    manifest = run.state["manifest"]
    assert manifest is not None

    descriptor = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert descriptor["kind"] == "okf"
    assert descriptor["id"] == run.state["run_id"]
    assert descriptor["documents"] == TARGET
    assert sorted(descriptor["domains"]) == sorted(TOPICS)
    assert descriptor["charter_digest"] == manifest.charter_digest
    # Advisory, not an expiry — but it has to be there, and it has to be later.
    assert manifest.stale_after is not None
    assert descriptor["stale_after"] == manifest.stale_after.isoformat()

    counts = manifest.counts
    assert counts.emitted == TARGET
    assert counts.unique >= counts.curated >= counts.emitted

    # The charter the descriptor points at is in the bundle, so the pointer resolves.
    assert (bundle / descriptor["charter"]).is_file()


async def test_the_log_never_reports_a_dollar_amount_for_an_unpriced_model(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fake/*` is in nobody's price map, and LiteLLM answers that with 0.0 rather than
    an error. A log saying `$0.00` would read as good news about a run that spent."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    log = (bundle / LOG_FILENAME).read_text(encoding="utf-8")
    assert "cost unavailable" in log
    assert "$0.00" not in log
    manifest = run.state["manifest"]
    assert manifest is not None and manifest.cost.calls > 0


async def test_the_log_and_the_root_index_are_readable_documents_in_their_own_right(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for filename in (INDEX_FILENAME, LOG_FILENAME):
        fields, body = load((bundle / filename).read_text(encoding="utf-8"))
        assert fields["title"]
        assert fields["type"]
        assert body.strip()
