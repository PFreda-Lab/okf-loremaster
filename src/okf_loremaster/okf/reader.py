"""Reading a bundle back off disk, without depending on the thing that wrote it.

Our own reader, deliberately, rather than importing a downstream parser: the bundle is a
contract between two programs that must be able to change independently, and a validator
that borrowed the consumer's parser could only ever prove the bundle agrees with itself.

Nothing here raises for a bad document. A malformed file is the single most useful thing
a validator can report, and a reader that threw on the first one would hide every file
after it. Read failures land in `problems` and the document is simply absent, so
`validate` reports "this file is unreadable" alongside everything else rather than
instead of it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from okf_loremaster.okf.frontmatter import FrontmatterError, load
from okf_loremaster.okf.layout import (
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    FULL_TEXT_BASIS,
    INDEX_FILENAME,
    LOG_FILENAME,
    NONE_CELL,
)

__all__ = [
    "OkfBundle",
    "OkfDocument",
    "OkfShelf",
    "body_sections",
    "fact_list",
    "markdown_table",
    "read_bundle",
]

# A `- **Label** — value` line, the shape every fact list in a body uses.
_FACT = re.compile(r"^-\s+\*\*(?P<label>[^*]+)\*\*\s+—\s+(?P<value>.*)$")

# A column break. Escaped pipes belong to the cell, which is how a title containing one
# survives the round trip.
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def body_sections(body: str) -> list[tuple[str, str]]:
    """Split a body into `(heading, text)` on its `# ` headings, in order.

    Only single-hash headings open a section. A `## ` inside one belongs to it, which is
    what lets a section carry sub-structure without inventing a sixth section.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in body.splitlines():
        if line.startswith("# "):
            if heading or buffer:
                sections.append((heading, "\n".join(buffer).strip()))
            heading = line[2:].strip()
            buffer = []
        else:
            buffer.append(line)
    if heading or buffer:
        sections.append((heading, "\n".join(buffer).strip()))
    # A preamble before the first heading is not a section.
    return [(name, text) for name, text in sections if name]


def markdown_table(text: str) -> list[dict[str, str]]:
    """Rows of the first pipe table in `text`, each keyed by its column heading.

    The exact inverse of what the emitter writes: `\\|` reads back as a literal pipe and
    the empty-cell dash reads back as `""`, so a value that made the round trip is the
    value that went in. Only the first table is returned — parsing every pipe line in a
    section would merge two tables into one nonsensical third.

    A row with the wrong number of cells is kept, not dropped. It is a malformed table,
    and reporting the columns it does have beats reporting nothing at all.
    """
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            rows.append(_cells(stripped))
        elif rows:
            break
    if len(rows) < 2 or not _is_rule(rows[1]):
        return []
    header = rows[0]
    return [dict(zip(header, cells, strict=False)) for cells in rows[2:]]


def fact_list(text: str) -> dict[str, str]:
    """The `- **Label** — value` lines of a section, as `{label: value}`.

    Later repeats of a label win, which is arbitrary but total: the emitter never writes
    one twice, so a duplicate means a hand edit and either answer is a guess.
    """
    facts: dict[str, str] = {}
    for line in text.splitlines():
        match = _FACT.match(line.strip())
        if match is not None:
            facts[match.group("label").strip()] = match.group("value").strip()
    return facts


def _cells(line: str) -> list[str]:
    parts = _UNESCAPED_PIPE.split(line)
    # A table row opens and closes with a pipe, so the split yields an empty string at
    # each end that is not a column.
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    values = [part.replace("\\|", "|").strip() for part in parts]
    return ["" if value == NONE_CELL else value for value in values]


def _is_rule(cells: list[str]) -> bool:
    return bool(cells) and all(set(cell) <= {"-", ":"} and cell for cell in cells)


@dataclass(frozen=True, slots=True)
class OkfDocument:
    """One markdown file: its frontmatter, its body, and where it was found."""

    path: Path
    # The folder that holds it, which `domain` is required to equal. Empty at the root.
    shelf: str
    fields: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def title(self) -> str:
        return str(self.fields.get("title") or "")

    @property
    def domain(self) -> str:
        return str(self.fields.get("domain") or "")

    @property
    def pmid(self) -> str:
        return str(self.fields.get("pmid") or "")

    @property
    def doc_id(self) -> str:
        """`id`, falling back to the filename stem as the spec requires."""
        return str(self.fields.get("id") or self.path.stem)

    @property
    def tags(self) -> list[str]:
        raw = self.fields.get("tags")
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return [part.strip() for part in str(raw or "").split(",") if part.strip()]

    @property
    def n(self) -> int | None:
        """The sample size as a number, or `None`. Frontmatter quotes it as a string."""
        try:
            return int(str(self.fields.get("n", "")).strip())
        except ValueError:
            return None

    @property
    def full_text(self) -> bool:
        return str(self.fields.get("text_basis") or "").strip().lower() == FULL_TEXT_BASIS

    @property
    def export_safe(self) -> bool:
        """Whether the document says it may be redistributed.

        Compared as a string, and that is the whole point of it being a property. The
        writer quotes every flat scalar, so the value on disk is `"true"` or `"false"` —
        and `"false"` is truthy in Python. Anything that reads the field and tests it for
        truth redistributes the entire bundle while reporting it as filtered. A real bool
        is accepted too, for a bundle some other tool wrote.
        """
        raw = self.fields.get("export_safe")
        if isinstance(raw, bool):
            return raw
        return str(raw if raw is not None else "").strip().lower() == "true"

    def sections(self) -> list[tuple[str, str]]:
        return body_sections(self.body)

    def section(self, heading: str) -> str | None:
        for name, text in self.sections():
            if name == heading:
                return text
        return None


@dataclass(frozen=True, slots=True)
class OkfShelf:
    slug: str
    path: Path
    index: OkfDocument | None = None
    documents: tuple[OkfDocument, ...] = ()

    @property
    def title(self) -> str:
        if self.index is None:
            return ""
        return str(self.index.fields.get("domain_title") or self.index.title or "")


@dataclass(frozen=True, slots=True)
class OkfBundle:
    path: Path
    index: OkfDocument | None = None
    shelves: tuple[OkfShelf, ...] = ()
    catalog: tuple[dict[str, Any], ...] = ()
    descriptor: dict[str, Any] = field(default_factory=dict)
    charter: dict[str, Any] = field(default_factory=dict)
    # Files that exist but could not be read, as `(path, why)`. Never silently skipped.
    problems: tuple[tuple[Path, str], ...] = ()

    def documents(self) -> Iterator[OkfDocument]:
        for shelf in self.shelves:
            yield from shelf.documents

    @property
    def document_count(self) -> int:
        return sum(len(shelf.documents) for shelf in self.shelves)

    @property
    def has_catalog(self) -> bool:
        return (self.path / CATALOG_FILENAME).exists()

    @property
    def has_descriptor(self) -> bool:
        return (self.path / DESCRIPTOR_FILENAME).exists()

    @property
    def has_log(self) -> bool:
        return (self.path / LOG_FILENAME).exists()

    @property
    def has_charter(self) -> bool:
        return (self.path / CHARTER_FILENAME).exists()


def read_bundle(path: Path) -> OkfBundle:
    """Read every markdown file, the catalog, the descriptor and the charter."""
    if not path.exists():
        raise FileNotFoundError(f"no bundle at {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is a file, not a bundle directory")

    problems: list[tuple[Path, str]] = []
    index = _read_document(path / INDEX_FILENAME, shelf="", problems=problems)

    shelves: list[OkfShelf] = []
    for directory in sorted(p for p in path.iterdir() if p.is_dir()):
        if directory.name.startswith(".") or directory.name.startswith("_"):
            continue
        shelves.append(_read_shelf(directory, problems))

    catalog, catalog_problems = _read_catalog(path / CATALOG_FILENAME)
    problems.extend(catalog_problems)

    return OkfBundle(
        path=path,
        index=index,
        shelves=tuple(shelves),
        catalog=tuple(catalog),
        descriptor=_read_yaml(path / DESCRIPTOR_FILENAME, problems),
        charter=_read_yaml(path / CHARTER_FILENAME, problems),
        problems=tuple(problems),
    )


def _read_shelf(directory: Path, problems: list[tuple[Path, str]]) -> OkfShelf:
    documents: list[OkfDocument] = []
    for candidate in sorted(directory.glob("*.md")):
        if candidate.name == INDEX_FILENAME:
            continue
        document = _read_document(candidate, shelf=directory.name, problems=problems)
        if document is not None:
            documents.append(document)
    return OkfShelf(
        slug=directory.name,
        path=directory,
        index=_read_document(directory / INDEX_FILENAME, shelf=directory.name, problems=problems),
        documents=tuple(documents),
    )


def _read_document(
    path: Path, *, shelf: str, problems: list[tuple[Path, str]]
) -> OkfDocument | None:
    if not path.exists():
        return None
    try:
        fields, body = load(path.read_text(encoding="utf-8"))
    except (FrontmatterError, OSError, UnicodeDecodeError) as exc:
        problems.append((path, str(exc)))
        return None
    return OkfDocument(path=path, shelf=shelf, fields=fields, body=body)


def _read_catalog(path: Path) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    if not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    problems: list[tuple[Path, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], [(path, str(exc))]
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append((path, f"line {number} is not valid JSON: {exc}"))
            continue
        if isinstance(loaded, dict):
            rows.append(loaded)
        else:
            problems.append((path, f"line {number} is not a JSON object"))
    return rows, problems


def _read_yaml(path: Path, problems: list[tuple[Path, str]]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        problems.append((path, str(exc)))
        return {}
    return loaded if isinstance(loaded, dict) else {}
