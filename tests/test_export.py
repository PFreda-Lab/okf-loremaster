"""`export` and `inspect`, against a real bundle rather than a fixture of one.

The claim worth testing is that an export is a *bundle*, not a directory of files: it is
validated here through the same `validate_bundle` the build gate runs, because a filtered
copy that a consumer refuses to attach is worse than no export at all.

The filter is asserted on the licenses `fake_ncbi` cycles — CC BY, CC BY-NC, CC0, and one
unrecognized code — which is why the exportable subset is a strict subset. A corpus that
was all permissive or none would pass a filter that does nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from okf_loremaster.cli import app
from okf_loremaster.emitters.export import export_bundle
from okf_loremaster.okf.layout import (
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    INDEX_FILENAME,
    LOG_FILENAME,
)
from okf_loremaster.okf.overview import read_overview
from okf_loremaster.okf.reader import read_bundle
from okf_loremaster.okf.validate import validate_bundle

from graph_runs import TARGET, full_run

runner = CliRunner()


async def built(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    return Path(run.state["bundle"])


# --- the copy is a bundle ----------------------------------------------------


async def test_an_unfiltered_export_is_the_same_corpus_and_still_validates(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    destination = tmp_path / "exported"

    result = export_bundle(bundle, destination)

    assert result.documents == TARGET
    assert result.omitted == ()
    report = validate_bundle(destination)
    assert report.errors == (), report.lines()
    assert report.documents == TARGET


async def test_a_permissive_export_keeps_a_strict_subset_and_still_validates(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate for `--permissive-only`, and the string-truthiness trap underneath it.

    `export_safe` reads back off disk as the string `"false"`, which is truthy. An
    exporter that tested it for truth would copy every document and report the copy as
    filtered — no error, no warning, and a redistribution of things that may not be
    redistributed. `0 < kept < TARGET` is what catches it.
    """
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    destination = tmp_path / "permissive"

    result = export_bundle(bundle, destination, permissive_only=True)

    assert 0 < result.documents < TARGET
    assert result.omitted_count == TARGET - result.documents
    report = validate_bundle(destination)
    assert report.errors == (), report.lines()

    # Every file that survived says so itself, and every license behind one is permissive.
    for document in read_bundle(destination).documents():
        assert document.export_safe is True
        assert str(document.fields["license"]).lower() in {"cc by", "cc0"}


async def test_a_retained_document_is_copied_byte_for_byte(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-rendering is where a verbatim quote stops being verbatim."""
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    destination = tmp_path / "exported"

    export_bundle(bundle, destination)

    for document in read_bundle(destination).documents():
        original = bundle / document.topic / document.filename
        assert document.path.read_bytes() == original.read_bytes(), document.filename


async def test_an_emptied_topic_keeps_its_directory_and_says_why(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent topic and one nothing survived on are different claims."""
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    for path in sorted((bundle / "alpha").glob("*.md")):
        if path.name == INDEX_FILENAME:
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace('export_safe: "true"', 'export_safe: "false"')
            .replace('license: "CC BY"', 'license: "publisher copyright"')
            .replace('license: "CC0"', 'license: "publisher copyright"'),
            encoding="utf-8",
        )
    destination = tmp_path / "permissive"

    export_bundle(bundle, destination, permissive_only=True)

    topic = read_bundle(destination).topics
    alpha = next(item for item in topic if item.slug == "alpha")
    assert alpha.documents == ()
    assert alpha.index is not None
    assert "license that permits redistribution" in alpha.index.body
    # And the taxonomy still reads the same from the root.
    assert validate_bundle(destination).errors == ()


# --- the descriptor ----------------------------------------------------------


async def test_the_export_gets_its_own_id_and_no_vector_pointer(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two corpora sharing a resource id is a collision a consumer cannot notice.

    The dropped `vectors` key is the same problem in the other direction: the store is
    not copied, and it embeds documents the filter removed.
    """
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    source = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    destination = tmp_path / "permissive"

    result = export_bundle(bundle, destination, permissive_only=True)

    copied = yaml.safe_load((destination / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert copied["id"] != source["id"]
    assert copied["id"].startswith(str(source["id"]))
    assert copied["derived_from"] == source["id"]
    assert copied["export_filter"] == "permissive-only"
    assert copied["documents"] == result.documents
    assert copied["omitted"] == result.omitted_count
    assert "vectors" not in copied
    # The taxonomy survives whole, including a topic that kept nothing.
    assert set(copied["domains"]) == {topic.slug for topic in read_bundle(bundle).topics}


async def test_the_catalog_is_filtered_rather_than_rebuilt(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`unmapped_vocab` lives only in the catalog, so rebuilding one would lose it."""
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    source_rows = {
        row["file"]: row
        for row in (
            json.loads(line)
            for line in (bundle / CATALOG_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    destination = tmp_path / "permissive"

    export_bundle(bundle, destination, permissive_only=True)

    copied = [
        json.loads(line)
        for line in (destination / CATALOG_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert copied
    for row in copied:
        assert row == source_rows[row["file"]]


async def test_the_log_and_charter_come_along_unchanged(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    destination = tmp_path / "exported"

    export_bundle(bundle, destination)

    for name in (LOG_FILENAME, CHARTER_FILENAME):
        assert (destination / name).read_bytes() == (bundle / name).read_bytes()


# --- what it refuses ---------------------------------------------------------


async def test_a_non_empty_destination_is_refused_before_anything_is_written(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "notes.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        export_bundle(bundle, destination)

    assert sorted(p.name for p in destination.iterdir()) == ["notes.txt"]


async def test_an_export_into_its_own_source_is_refused(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It would be read back as a topic of the bundle it came from."""
    bundle = await built(settings_factory, tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="is inside"):
        export_bundle(bundle, bundle / "copy")
    with pytest.raises(ValueError, match="the destination is the bundle itself"):
        export_bundle(bundle, bundle)


async def test_a_filter_that_keeps_nothing_refuses_rather_than_writing_an_empty_bundle(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    for path in sorted(bundle.rglob("*.md")):
        if path.name == INDEX_FILENAME:
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('export_safe: "true"', 'export_safe: "false"'), "utf-8")
    destination = tmp_path / "permissive"

    with pytest.raises(ValueError, match="no document carries a license"):
        export_bundle(bundle, destination, permissive_only=True)

    assert not destination.exists()


async def test_a_flag_that_disagrees_with_its_license_is_excluded_and_named(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is derived from the license, so disagreement means a hand edit.

    Whether to hand something to a third party is the wrong question to resolve
    optimistically, so the export takes the conservative side and says which file.
    """
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    edited = next(
        document
        for document in read_bundle(bundle).documents()
        if not document.export_safe
    )
    edited.path.write_text(
        edited.path.read_text(encoding="utf-8").replace(
            'export_safe: "false"', 'export_safe: "true"'
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "permissive"

    result = export_bundle(bundle, destination, permissive_only=True)

    assert not (destination / edited.topic / edited.filename).exists()
    assert any(edited.filename in note for note in result.warnings)


# --- the commands ------------------------------------------------------------


async def test_export_writes_a_bundle_the_validate_command_accepts(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    destination = tmp_path / "shipped"

    result = runner.invoke(
        app, ["export", str(bundle), "-o", str(destination), "--permissive-only"]
    )

    assert result.exit_code == 0, result.output
    assert "left behind" in result.output
    assert runner.invoke(app, ["validate", str(destination)]).exit_code == 0


async def test_export_reports_a_refusal_as_one_line_rather_than_a_traceback(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)

    result = runner.invoke(app, ["export", str(bundle), "-o", str(bundle / "inside")])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "is inside" in result.output


# --- inspect -----------------------------------------------------------------


async def test_inspect_counts_what_is_on_disk_not_what_the_run_believed(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)

    overview = read_overview(bundle)

    assert overview.documents == TARGET
    assert overview.catalog_rows == TARGET
    assert sum(topic.documents for topic in overview.topics) == TARGET
    assert overview.full_text + overview.abstract_only == TARGET
    assert 0 < overview.exportable < TARGET
    assert overview.predictors > 0
    assert overview.with_effect + overview.unverified <= overview.predictors
    assert overview.designs and all(count > 0 for _, count in overview.designs)
    # A count of papers per key, not the codes themselves: `Counter.update` on a fact
    # list adds its *values*, and a tally of `"A00A00A00"` renders without erroring.
    assert overview.vocabularies == (("icd10", TARGET),)
    assert overview.notes == ()
    assert overview.problems == ()
    # The run facts come off the root index, not off a run that is no longer in memory.
    assert overview.index_facts["Run id"]
    assert overview.resource_id == overview.index_facts["Run id"]


async def test_inspect_reads_a_bundle_whose_run_is_long_gone(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moved, renamed, and with no catalog — a bundle off a USB stick still summarizes."""
    bundle = await built(settings_factory, tmp_path, monkeypatch)
    moved = tmp_path / "somewhere-else"
    export_bundle(bundle, moved)
    (moved / CATALOG_FILENAME).unlink()

    overview = read_overview(moved)

    assert overview.documents == TARGET
    assert overview.catalog_rows == 0
    assert overview.designs  # read from the documents instead
    assert any(CATALOG_FILENAME in note for note in overview.notes)


async def test_inspect_prints_the_summary_and_exits_zero(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await built(settings_factory, tmp_path, monkeypatch)

    result = runner.invoke(app, ["inspect", str(bundle)])

    assert result.exit_code == 0, result.output
    # Substrings that survive Rich's highlighter, which styles `row(s)` as a call and
    # splits it with escape codes. Asserting on those would be asserting on a theme.
    for expected in ("topics", "corpus", "median reported sample size", "effect sizes"):
        assert expected in result.output, expected
    assert "no vector index" in result.output  # this run built none


def test_inspect_says_a_bundle_is_missing_rather_than_summarizing_nothing(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path / "not-a-bundle")])

    assert result.exit_code == 1
    assert "not-a-bundle" in result.output
