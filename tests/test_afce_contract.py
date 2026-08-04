"""The downstream contract, checked from the downstream side.

Every other test in this suite reads a bundle with our own reader, which can only ever
prove the bundle agrees with itself. This one throws that reader away and re-implements
the consumer described in the AFCE spec — a dependency-free line parser, a resolver that
accepts three reference forms, and a fuzzy-match haystack — then drops a finished bundle
into a `resources/` directory and asks whether it resolves.

The rules are numbered as they are in that spec (§6.4.1), and each test names the one it
covers. What they have in common is that every one of them fails *silently* on a real
corpus: a `domain` that does not match its folder shelves a paper where nobody looks, a
frontmatter block that YAML and a line parser read differently means two things at once,
and an empty search surface makes a perfectly good document unfindable rather than
missing. None of those raise, and none are visible in the bundle we wrote.

The consumer here is deliberately not the one in FE_Demo2 — a test that imported a
sibling repository would break when that repository moved, and would prove the bundle
matches one implementation rather than the contract. FE_Demo2's real `okf_*` resolver
was run against this output by hand at the step 10 gate; see `Build_Progress.md`.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from okf_loremaster.okf.layout import (
    CATALOG_FILENAME,
    DESCRIPTOR_FILENAME,
    INDEX_FILENAME,
)

from graph_runs import TARGET, full_run

# The three keys OKF v0.2 nests. Rule 7 allows flow style on one line for these and
# nothing else.
NESTED_KEYS = ("generated", "verified", "sources")

# Rule 4: what retrieval matches over. Anything not in here is invisible to a search.
HAYSTACK_FIELDS = ("title", "description", "tags", "journal")


# --- a consumer, written from the spec rather than from our writer ------------


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """One key per line, values left exactly as written. No YAML.

    Returns raw values so a test can assert on the quoting, which is the half of rule 7
    a parser that unquoted eagerly would hide.
    """
    if not text.startswith("---\n"):
        raise AssertionError("no frontmatter block")
    end = text.index("\n---", 3)
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            raise AssertionError(f"not one key per line: {line!r}")
        key, _, rest = line.partition(":")
        fields[key.strip()] = rest.strip()
    return fields, text[end + 4 :].lstrip("\n")


def scalar(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return raw


def string_list(raw: str) -> list[str]:
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    return [scalar(part.strip()) for part in inner.split(",")]


class Consumer:
    """Everything the spec says a reader does, and nothing it says a producer does."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.descriptor: dict[str, Any] = {}
        # Rule 8: optional, authoritative when present.
        if (root / DESCRIPTOR_FILENAME).exists():
            loaded = yaml.safe_load((root / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
            self.descriptor = loaded if isinstance(loaded, dict) else {}

        self.records: list[dict[str, Any]] = []
        self.domains: dict[str, str] = {}
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            slug = directory.name
            # Rule 5: domains come from the corpus. Titles from the descriptor when it
            # has them, else from the domain index — never from a list in our code.
            title = str(self.descriptor.get("domains", {}).get(slug) or "")
            if not title and (directory / INDEX_FILENAME).exists():
                fields, _ = parse_frontmatter(
                    (directory / INDEX_FILENAME).read_text(encoding="utf-8")
                )
                title = scalar(fields.get("domain_title", "")) or scalar(fields.get("title", ""))
            self.domains[slug] = title
            for path in sorted(directory.glob("*.md")):
                if path.name == INDEX_FILENAME:  # rule 3
                    continue
                self.records.append(self._record(path, slug))

    def _record(self, path: Path, slug: str) -> dict[str, Any]:
        raw, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        fields = {key: scalar(value) for key, value in raw.items()}
        return {
            "raw": raw,
            "body": body,
            "path": f"{slug}/{path.name}",
            "filename": path.name,
            "domain": fields.get("domain", ""),
            "title": fields.get("title", ""),
            "description": fields.get("description", ""),
            "journal": fields.get("journal", ""),
            "pmid": fields.get("pmid", ""),
            # Rule 1: `id` falls back to the filename stem.
            "id": fields.get("id") or path.stem,
            "tags": string_list(raw["tags"]) if raw.get("tags") else [],
        }

    def browse(self, domain: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record["domain"] == domain]

    def read(self, ref: str) -> dict[str, Any] | None:
        """Rule 6: `id`/PMID, bare filename, or `domain/file.md`."""
        for key in ("id", "pmid", "filename", "path"):
            for record in self.records:
                if record[key] and record[key] == ref:
                    return record
        return None

    def search(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        """Token-set overlap over the rule-4 haystack. Ranking, not rapidfuzz."""
        wanted = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = [
            (len(wanted & set(re.findall(r"[a-z0-9]+", haystack(record)))) / len(wanted), record)
            for record in self.records
        ]
        return sorted(scored, key=lambda pair: -pair[0])


def haystack(record: dict[str, Any]) -> str:
    parts = [record["title"], record["description"], " ".join(record["tags"]), record["journal"]]
    return " ".join(parts).lower()


async def dropped_in(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A finished bundle, in a `resources/okf/` directory the way a user leaves it."""
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    resources = tmp_path / "resources" / "okf"
    resources.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(run.state["bundle"]), resources)
    return resources


# --- the rules ---------------------------------------------------------------


async def test_a_bundle_dropped_into_resources_needs_no_configuration(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 5 — domains come from the corpus, not from a list in the consumer's code."""
    consumer = Consumer(await dropped_in(settings_factory, tmp_path, monkeypatch))

    assert len(consumer.records) == TARGET
    assert consumer.domains  # discovered by walking, nothing configured
    assert all(title for title in consumer.domains.values()), consumer.domains
    assert sum(len(consumer.browse(slug)) for slug in consumer.domains) == TARGET


async def test_every_document_resolves_by_all_three_reference_forms(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 6 — agents cite inconsistently and a lookup miss wastes a whole turn."""
    consumer = Consumer(await dropped_in(settings_factory, tmp_path, monkeypatch))

    for record in consumer.records:
        for ref in (record["id"], record["pmid"], record["filename"], record["path"]):
            assert consumer.read(ref) is record, f"{ref} did not resolve"


async def test_a_line_parser_and_a_yaml_parser_read_the_same_frontmatter(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 7 — a block two readers parse differently means two different things.

    The nested keys are the whole difficulty: `generated`, `verified` and `sources` are
    nested in the spec, so they cannot be flattened, and a consumer parses line by line,
    so they cannot be indented across lines. Flow style on one line is the only shape
    that is valid YAML to one reader and one opaque string to the other.
    """
    consumer = Consumer(await dropped_in(settings_factory, tmp_path, monkeypatch))

    for record in consumer.records:
        path = consumer.root / record["path"]
        text = path.read_text(encoding="utf-8")
        block = text[4 : text.index("\n---", 3)]
        loaded = yaml.safe_load(block)

        for key, raw in record["raw"].items():
            if key in NESTED_KEYS:
                assert raw[0] in "{[" and "\n" not in raw, f"{key} is not flow style"
                assert yaml.safe_load(raw) == loaded[key]
                continue
            if raw.startswith("["):
                assert string_list(raw) == loaded[key], key
                continue
            # Rule 7's "strictly quoted": an unquoted `n` is an int to YAML and a str
            # to a line parser, which is exactly the divergence being ruled out.
            assert raw.startswith('"') and raw.endswith('"'), f"{key} is a bare scalar"
            assert scalar(raw) == loaded[key], key


async def test_the_search_surface_is_populated_for_every_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 4 — the single most common way a valid corpus under-performs.

    A document whose haystack is a bare title is not missing and does not error; it is
    simply never returned, which reads downstream as a gap in the literature.
    """
    consumer = Consumer(await dropped_in(settings_factory, tmp_path, monkeypatch))

    for record in consumer.records:
        for field_name in HAYSTACK_FIELDS:
            assert record[field_name], f"{record['path']} has no {field_name}"

    # And the surface discriminates: a shelf-specific term ranks that shelf first.
    slug = next(iter(consumer.domains))
    ranked = consumer.search(f"{slug} exposure outcome")
    top = ranked[: len(consumer.browse(slug))]
    assert all(record["domain"] == slug for _, record in top), [r["path"] for _, r in top]


async def test_only_title_and_domain_are_required_and_domain_is_the_folder(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rules 1 and 2 — a mismatch shelves a paper where nobody browsing will look."""
    consumer = Consumer(await dropped_in(settings_factory, tmp_path, monkeypatch))

    for record in consumer.records:
        assert record["title"]
        assert record["domain"] == record["path"].split("/")[0]
        assert "shelf" not in record["raw"], "the human word leaked into frontmatter"


async def test_the_reserved_names_are_never_documents(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 3 — `index.md` is regenerated, so a document there would be overwritten."""
    resources = await dropped_in(settings_factory, tmp_path, monkeypatch)
    consumer = Consumer(resources)

    assert (resources / INDEX_FILENAME).exists()
    assert all(INDEX_FILENAME not in record["path"] for record in consumer.records)
    for slug in consumer.domains:
        assert (resources / slug / INDEX_FILENAME).exists()
    # The catalog is ours, not the spec's, and lives out of the way of a `*.md` walk.
    assert CATALOG_FILENAME.startswith("_")


async def test_the_descriptor_carries_the_id_and_the_domain_titles(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 8 — optional, authoritative when present, and never a new requirement.

    Two halves: what a consumer must find in it, and that everything else it carries is
    ignorable. A descriptor richer than the spec has to attach as cleanly as a minimal
    one, or a producer is punished for recording more provenance.
    """
    resources = await dropped_in(settings_factory, tmp_path, monkeypatch)
    consumer = Consumer(resources)

    assert consumer.descriptor["id"]
    assert consumer.descriptor["domains"] == consumer.domains
    assert consumer.descriptor["documents"] == len(consumer.records)

    # Titles still resolve with the descriptor gone — it is a way to be explicit.
    (resources / DESCRIPTOR_FILENAME).unlink()
    without = Consumer(resources)
    assert without.descriptor == {}
    assert without.domains == consumer.domains
