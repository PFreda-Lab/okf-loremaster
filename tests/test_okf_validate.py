"""What the validator refuses, proved by breaking a bundle that was passing.

A gate is only worth what it catches, and a gate tested only against good input reports
"passes" for a validator that returns `ok` unconditionally. So every check here starts
from the golden bundle, breaks exactly one thing on disk, and asserts the report went
from clean to naming that thing — the mutation is the test.

The frontmatter unit tests below cover what a whole run never produces: indentation, a
duplicate key, a value with a newline in it. Those are the shapes a hand-edited bundle
arrives in, and hand-authoring is explicitly as valid as building.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from okf_loremaster.okf.frontmatter import (
    FrontmatterError,
    load,
    parse,
    render,
    split,
    stamp,
)
from okf_loremaster.okf.layout import CATALOG_FILENAME, INDEX_FILENAME
from okf_loremaster.okf.reader import body_sections
from okf_loremaster.okf.validate import UNMAPPED_SHARE, Severity, validate_bundle
from okf_loremaster.schemas import Topic

from graph_runs import charter_for, full_run

# --- frontmatter, on its own ------------------------------------------------


def test_a_block_survives_the_round_trip_it_will_actually_make() -> None:
    fields = {
        "type": "Literature Evidence",
        "title": 'A study of "quoted" things',
        "domain": "some-topic",
        "n": 1454,
        "export_safe": False,
        "tags": ["one", "two"],
        "generated": {"by": "model", "at": "2026-01-01T00:00:00Z"},
        "sources": [{"id": "pmid:1", "resource": "https://example.org/1"}],
    }
    text = render(fields)
    block, body = split(text + "\nbody\n")

    assert body.strip() == "body"
    assert parse(block) == {**fields, "n": "1454", "export_safe": "false"}
    # One key per line, no exceptions — that is what a naive reader depends on.
    assert len([line for line in block.splitlines() if line.strip()]) == len(fields)


def test_a_newline_inside_a_value_cannot_end_the_line_it_is_on() -> None:
    """The failure this module exists to prevent: a title with a line break in it turns
    the rest of the title into a keyless line, and a keyless line is a dropped key."""
    text = render({"title": "First half\nsecond half", "domain": "x"})
    block, _ = split(text)

    assert parse(block)["title"] == "First half second half"
    assert len(block.strip().splitlines()) == 2


def test_an_empty_value_is_omitted_but_a_false_one_is_written() -> None:
    block, _ = split(render({"title": "t", "journal": "", "tags": [], "export_safe": False}))
    fields = parse(block)

    assert "journal" not in fields
    assert "tags" not in fields
    assert fields["export_safe"] == "false"


@pytest.mark.parametrize(
    ("block", "why"),
    [
        ('title: "a"\n  by: "b"', "indented"),
        ('title: "a"\njust a sentence', "no key"),
        ('title: "a"\ntitle: "b"', "repeats"),
        ('ti tle: "a"', "unusable key"),
    ],
)
def test_the_line_parser_refuses_what_a_line_parser_would_misread(block: str, why: str) -> None:
    with pytest.raises(FrontmatterError) as caught:
        parse(block)

    assert why in str(caught.value)


def test_a_document_with_no_frontmatter_is_an_error_not_a_body(tmp_path: Path) -> None:
    """A file with no frontmatter has no `domain`, so reading it as body-only would
    produce a document that cannot be filed."""
    with pytest.raises(FrontmatterError):
        load("# just a heading\n")
    with pytest.raises(FrontmatterError):
        split("---\ntitle: \"a\"\nnever closed\n")


def test_a_timestamp_has_one_spelling(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    naive = datetime(2026, 8, 3, 12, 30, 5)
    assert stamp(naive) == "2026-08-03T12:30:05Z"
    assert stamp(naive.replace(tzinfo=UTC)) == "2026-08-03T12:30:05Z"


def test_a_subheading_belongs_to_its_section_rather_than_starting_one() -> None:
    sections = body_sections("# One\n\ntext\n\n## Inner\n\nmore\n\n# Two\n\ntail\n")

    assert [name for name, _ in sections] == ["One", "Two"]
    assert "## Inner" in sections[0][1]


# --- the validator, against a bundle that was passing -----------------------


async def bundle_for(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    path = Path(run.state["bundle"])
    assert validate_bundle(path).ok, "the fixture bundle was already broken"
    return path


def documents_in(bundle: Path) -> list[Path]:
    return sorted(
        path
        for topic in sorted(bundle.iterdir())
        if topic.is_dir()
        for path in sorted(topic.glob("*.md"))
        if path.name != INDEX_FILENAME
    )


def one_document(bundle: Path) -> Path:
    return documents_in(bundle)[0]


def messages(bundle: Path) -> str:
    report = validate_bundle(bundle)
    return "\n".join(f.line(relative_to=bundle) for f in report.errors)


async def test_a_domain_that_does_not_match_its_folder_is_an_error(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 2 of the downstream contract, and almost always a copy-paste bug. Silently
    re-filing it would hide the bug and move the paper."""
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    document = one_document(bundle)
    document.write_text(
        document.read_text(encoding="utf-8").replace(
            f'domain: "{document.parent.name}"', 'domain: "somewhere-else"'
        ),
        encoding="utf-8",
    )

    assert "declares domain 'somewhere-else'" in messages(bundle)


async def test_a_bare_scalar_is_an_error_even_though_yaml_accepts_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    document = one_document(bundle)
    block, body = split(document.read_text(encoding="utf-8"))
    unquoted = re.sub(r'^n: "(\d+)"$', r"n: \1", block, flags=re.MULTILINE)
    assert unquoted != block, "the sample document had no `n` to unquote"
    document.write_text(f"---\n{unquoted}\n---\n\n{body}", encoding="utf-8")

    assert "bare scalar" in messages(bundle)


async def test_a_topic_key_in_frontmatter_is_an_error(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    document = one_document(bundle)
    document.write_text(
        document.read_text(encoding="utf-8").replace(
            "domain:", 'topic: "sneaked-in"\ndomain:', 1
        ),
        encoding="utf-8",
    )

    assert "the frontmatter key is `domain`" in messages(bundle)


async def test_a_moved_section_is_an_error(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The body is a contract too: a downstream agent reads by heading, and a bundle
    whose sections drift is one an agent has to re-learn per document."""
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    document = one_document(bundle)
    document.write_text(
        document.read_text(encoding="utf-8").replace("# Caveats", "# Limitations"),
        encoding="utf-8",
    )

    assert "body sections are" in messages(bundle)


async def test_an_empty_section_is_an_error(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    document = one_document(bundle)
    block, body = split(document.read_text(encoding="utf-8"))
    trimmed = body[: body.index("# Caveats")] + "# Caveats\n\n"
    document.write_text(f"---\n{block}\n---\n\n{trimmed}", encoding="utf-8")

    assert "is empty" in messages(bundle)


async def test_a_document_missing_from_the_catalog_is_an_error_and_so_is_the_reverse(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both directions. The catalog is what a consumer reads instead of walking the
    tree, so a row with no file misleads exactly as badly as a file with no row."""
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    catalog = bundle / CATALOG_FILENAME
    lines = catalog.read_text(encoding="utf-8").splitlines()
    catalog.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

    assert "has no row in the catalog" in messages(bundle)

    catalog.write_text(
        "\n".join([*lines, '{"file": "alpha/99999_Nobody.md", "pmid": "99999"}']) + "\n",
        encoding="utf-8",
    )
    assert "which is not in the bundle" in messages(bundle)


async def test_a_link_that_does_not_resolve_is_an_error(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle is meant to be readable offline. A cross-link into it that dead-ends
    costs a downstream agent a whole turn to discover."""
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    index = bundle / INDEX_FILENAME
    index.write_text(
        index.read_text(encoding="utf-8").replace("(charter.yaml)", "(charter-v2.yaml)"),
        encoding="utf-8",
    )

    assert "does not resolve" in messages(bundle)


async def test_two_documents_with_one_id_is_an_error(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handle that names two documents resolves to whichever is found first, which is
    a citation that silently points at the wrong paper."""
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    first, second = documents_in(bundle)[:2]
    second.write_text(
        second.read_text(encoding="utf-8").replace(
            f'id: "{second.stem.split("_")[0]}"', f'id: "{first.stem.split("_")[0]}"'
        ),
        encoding="utf-8",
    )

    assert "is already used by" in messages(bundle)


async def test_a_reserved_name_cannot_be_a_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    topic = next(p for p in sorted(bundle.iterdir()) if p.is_dir())
    (topic / "log.md").write_text('---\ntitle: "x"\n---\n\n# Bottom line\n\nx\n', encoding="utf-8")

    assert "is reserved and cannot be a document" in messages(bundle)


async def test_a_missing_root_index_is_an_error(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    (bundle / INDEX_FILENAME).unlink()

    assert "at the bundle root" in messages(bundle)


async def test_a_bundle_that_is_not_there_raises_rather_than_failing_quietly(
    tmp_path: Path,
) -> None:
    """The one thing the validator does raise for. "No bundle" and "a bundle with 30
    errors" are different answers, and returning the second for the first would send
    someone reading a report about files that do not exist."""
    with pytest.raises(FileNotFoundError):
        validate_bundle(tmp_path / "nothing-here")


# --- the advisory half ------------------------------------------------------


async def test_an_untagged_document_is_a_warning_and_not_a_failure(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It makes the bundle worse without making it wrong: retrieval matches over title,
    tags and journal, so the paper is findable by its title alone."""
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    document = one_document(bundle)
    document.write_text(
        "\n".join(
            line
            for line in document.read_text(encoding="utf-8").splitlines()
            if not line.startswith("tags:")
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_bundle(bundle)
    assert report.ok
    assert any("no `tags`" in f.message for f in report.warnings)


async def test_a_vocabulary_key_the_charter_never_listed_arrives_with_its_fix(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One extraction reaching for an unlisted key is a judgment call; a fifth of the
    corpus reaching for the same one is a hole in the charter — so past the threshold
    the warning carries the exact command that closes it."""
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    catalog = bundle / CATALOG_FILENAME
    lines = catalog.read_text(encoding="utf-8").splitlines()
    reached = max(1, int(len(lines) * UNMAPPED_SHARE) + 1)
    patched = [
        line[:-1] + ', "unmapped_vocab": {"snomed": ["12345"]}}' if index < reached else line
        for index, line in enumerate(lines)
    ]
    catalog.write_text("\n".join(patched) + "\n", encoding="utf-8")

    report = validate_bundle(bundle)
    assert report.ok  # advisory, never a gate
    note = next(f.message for f in report.warnings if "snomed" in f.message)
    assert "the charter did not list" in note
    assert "--vocab icd10,snomed" in note
    assert "--charter" in note


async def test_one_stray_vocabulary_key_is_reported_without_a_rerun_command(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await bundle_for(settings_factory, tmp_path, monkeypatch)
    catalog = bundle / CATALOG_FILENAME
    lines = catalog.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0][:-1] + ', "unmapped_vocab": {"snomed": ["12345"]}}'
    catalog.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = validate_bundle(bundle)
    note = next(f.message for f in report.warnings if "snomed" in f.message)
    assert "--vocab" not in note


async def test_a_topic_the_literature_could_not_fill_is_emitted_empty_and_warned_about(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty folder is a claim about the literature. Omitting it would make a topic
    nobody could fill indistinguishable from one nobody planned — so the charter asks
    for a facet the corpus does not contain, and the bundle has to say so."""
    charter = charter_for(
        topic_taxonomy=[
            *charter_for().topic_taxonomy,
            Topic(slug="epsilon", title="Epsilon", scope="a facet nobody wrote about",
                  seed_terms=["epsilon"]),
        ]
    )
    run = await full_run(settings_factory, tmp_path, monkeypatch, charter=charter)
    bundle = Path(run.state["bundle"])

    empty = bundle / "epsilon"
    assert (empty / INDEX_FILENAME).is_file()
    assert not [p for p in empty.glob("*.md") if p.name != INDEX_FILENAME]

    report = validate_bundle(bundle)
    assert report.ok, report.lines()
    assert any("retained no papers" in f.message for f in report.warnings)


def test_a_remote_embedding_model_is_flagged_before_a_consumer_rejects_it(
    tmp_path: Path,
) -> None:
    """Downstream verifies the embedder on attach and refuses a remote one. Finding that
    out at the far end is the worst place to find it out."""
    bundle = tmp_path / "empty"
    bundle.mkdir()

    report = validate_bundle(bundle, embed_model="openai/text-embedding-3-large")
    assert any("looks remote" in f.message for f in report.warnings)
    # A local hub checkpoint is `org/name`, and must not trip it.
    local = validate_bundle(bundle, embed_model="pritamdeka/S-PubMedBert-MS-MARCO")
    assert not any("looks remote" in f.message for f in local.warnings)


def test_warnings_never_fail_the_gate_and_errors_always_do(tmp_path: Path) -> None:
    bundle = tmp_path / "empty"
    bundle.mkdir()

    report = validate_bundle(bundle)
    assert not report.ok
    assert report.errors
    assert all(f.severity is Severity.ERROR for f in report.errors)
    assert all(f.severity is Severity.WARNING for f in report.warnings)
    assert "fails with" in report.summary()
