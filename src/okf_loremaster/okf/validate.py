"""Checking a finished bundle against the contract, from the outside.

Run in-graph after `emit_okf` and again by `okf-loremaster validate <bundle>`, over the
same code path, because the two questions are the same question: does what is on disk
satisfy what a downstream reader is promised?

The checks divide cleanly, and the division is the design:

**Errors are things that break a consumer.** A missing `domain`, a `domain` that does
not equal its folder, a document that will not parse, a section that moved. AFCE
resolves a document three ways and shelves it by folder; every one of these makes a
document silently unreachable or misfiled rather than visibly broken, which is why they
are checked here rather than trusted to the writer.

**Warnings are things that make a bundle worse without making it wrong.** A shelf that
retained nothing, a document with no tags — AFCE's retrieval haystack is title, tags and
journal, so an untagged document is findable only by its title. And the vocabulary
aggregate: an extraction that wanted a key the charter never listed is not an error, it
is evidence the charter was written before anyone had read a paper. One such key is
noise; the same key across more than `UNMAPPED_SHARE` of the corpus is a finding, and it
comes with the exact command that fixes it.

Every frontmatter block is checked through *both* our line-parser and `yaml.safe_load`,
and the two must agree. That is the only check that actually proves the flow-style
discipline held: a block that reads one way for a spec consumer and another way for a
line-parser is a bundle that means two different things depending on who opens it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from okf_loremaster.okf.frontmatter import FrontmatterError, parse, split
from okf_loremaster.okf.layout import (
    BODY_SECTIONS,
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    DISTANCES,
    INDEX_FILENAME,
    LOG_FILENAME,
    RESERVED_FILENAMES,
    vector_store_path,
)
from okf_loremaster.okf.reader import OkfBundle, OkfDocument, read_bundle

__all__ = [
    "UNMAPPED_SHARE",
    "BundleReport",
    "Finding",
    "Severity",
    "validate_bundle",
]

# The share of documents that must reach for the same unlisted vocabulary key before it
# stops being one extraction's idea and starts being a hole in the charter.
UNMAPPED_SHARE = 0.15

# Embedding model names that are not a local checkpoint. A downstream consumer verifies
# the embedder on attach and rejects a remote one outright, so a bundle built with one
# would fail at the far end rather than here — which is the worst place to find out.
# Provider prefixes, not model names: `org/name` is how a local hub checkpoint is spelled.
REMOTE_EMBEDDERS = (
    "http://",
    "https://",
    "openai/",
    "azure/",
    "anthropic/",
    "bedrock/",
    "cohere/",
    "gemini/",
    "jina/",
    "mistral/",
    "vertex_ai/",
    "voyage/",
)

# A markdown link target. Only relative ones are checked; a bundle is meant to be
# readable offline, so a link into it that does not resolve is a dead end.
_LINK = re.compile(r"\]\(([^)\s]+)\)")

# What a frontmatter value may start with. AFCE's parser accepts quoted flat scalars and
# string lists; OKF's nested blocks arrive as flow maps. Anything else is a bare scalar,
# which reads as an unquoted string at one end and as a typed value at the other.
_ALLOWED_VALUE_STARTS = ('"', "[", "{")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    severity: Severity
    message: str
    path: Path | None = None

    def line(self, *, relative_to: Path | None = None) -> str:
        if self.path is None:
            return self.message
        shown = self.path
        if relative_to is not None:
            with suppress(ValueError):  # a path outside the bundle is shown in full
                shown = self.path.relative_to(relative_to)
        return f"{shown}: {self.message}"


@dataclass(frozen=True, slots=True)
class BundleReport:
    path: Path
    documents: int = 0
    shelves: int = 0
    findings: tuple[Finding, ...] = ()

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """Whether the bundle passes the hard gate. Warnings do not fail it."""
        return not self.errors

    def summary(self) -> str:
        state = "passes" if self.ok else f"fails with {len(self.errors)} error(s)"
        detail = f"{self.documents} document(s) across {self.shelves} shelf/shelves"
        note = f", {len(self.warnings)} warning(s)" if self.warnings else ""
        return f"{detail}; {state}{note}"

    def lines(self) -> list[str]:
        return [f.line(relative_to=self.path) for f in self.findings]


def validate_bundle(path: Path, *, embed_model: str = "") -> BundleReport:
    """Read a bundle and check it. Never raises for a bad bundle — only a missing one."""
    bundle = read_bundle(path)
    findings: list[Finding] = []

    for bad_path, why in bundle.problems:
        findings.append(Finding(Severity.ERROR, f"could not be read: {why}", bad_path))

    _check_root(bundle, findings)
    _check_shelves(bundle, findings)
    for document in bundle.documents():
        _check_document(document, findings)
    _check_catalog(bundle, findings)
    _check_links(bundle, findings)
    _check_ids(bundle, findings)
    _check_vocabulary(bundle, findings)
    _check_embedder(embed_model, findings)
    _check_vector_index(bundle, findings)

    return BundleReport(
        path=path,
        documents=bundle.document_count,
        shelves=len(bundle.shelves),
        findings=tuple(findings),
    )


# --- the bundle as a whole --------------------------------------------------


def _check_root(bundle: OkfBundle, findings: list[Finding]) -> None:
    if bundle.index is None:
        findings.append(
            Finding(
                Severity.ERROR,
                f"no {INDEX_FILENAME} at the bundle root — a consumer reads the domains "
                f"from it",
                bundle.path,
            )
        )
    if not bundle.shelves:
        findings.append(
            Finding(Severity.ERROR, "no shelf directories — the bundle holds nothing", bundle.path)
        )
    if not bundle.has_catalog:
        findings.append(Finding(Severity.ERROR, f"no {CATALOG_FILENAME}", bundle.path))
    for filename, what in (
        (DESCRIPTOR_FILENAME, "a consumer detects the bundle without it, but records nothing"),
        (LOG_FILENAME, "the build history is not recoverable from the bundle alone"),
        (CHARTER_FILENAME, "what the bundle was built for is not recoverable from it"),
    ):
        if not (bundle.path / filename).exists():
            findings.append(Finding(Severity.WARNING, f"no {filename} — {what}", bundle.path))

    if bundle.index is not None and "domain" in bundle.index.fields:
        findings.append(
            Finding(
                Severity.ERROR,
                "the root index carries a `domain` key, but it sits in no domain folder",
                bundle.index.path,
            )
        )


def _check_shelves(bundle: OkfBundle, findings: list[Finding]) -> None:
    for shelf in bundle.shelves:
        if shelf.index is None:
            findings.append(
                Finding(Severity.ERROR, f"shelf has no {INDEX_FILENAME}", shelf.path)
            )
        elif shelf.index.domain != shelf.slug:
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"index declares domain {shelf.index.domain!r} but sits in {shelf.slug!r}",
                    shelf.index.path,
                )
            )
        if not shelf.documents:
            findings.append(
                Finding(
                    Severity.WARNING,
                    "shelf retained no papers — present and empty, which is a claim about "
                    "the literature rather than a missing folder",
                    shelf.path,
                )
            )
        stray = sorted(
            p.name
            for p in shelf.path.glob("*.md")
            if p.name in RESERVED_FILENAMES and p.name != INDEX_FILENAME
        )
        for name in stray:
            findings.append(
                Finding(Severity.ERROR, f"{name} is reserved and cannot be a document", shelf.path)
            )


# --- one document -----------------------------------------------------------


def _check_document(document: OkfDocument, findings: list[Finding]) -> None:
    def fail(message: str) -> None:
        findings.append(Finding(Severity.ERROR, message, document.path))

    def warn(message: str) -> None:
        findings.append(Finding(Severity.WARNING, message, document.path))

    if not document.title:
        fail("no `title` — it is required, and it is most of the search surface")
    if not document.domain:
        fail("no `domain` — it is required, and it is how the document is shelved")
    elif document.domain != document.shelf:
        fail(f"declares domain {document.domain!r} but sits in {document.shelf!r}")

    # The naming trap. "Shelf" is the human word; the key is `domain`, and a file
    # carrying both would be shelved by whichever one the reader happened to prefer.
    if "shelf" in document.fields:
        fail("carries a `shelf` key — the frontmatter key is `domain`")

    if not document.tags:
        warn(
            "no `tags` — retrieval matches over title, tags and journal, so this "
            "document is findable by its title alone"
        )

    _check_frontmatter_text(document, findings)
    _check_sections(document, findings)


def _check_frontmatter_text(document: OkfDocument, findings: list[Finding]) -> None:
    """Both parsers must see the same document, and every scalar must be quoted."""
    try:
        block, _ = split(document.path.read_text(encoding="utf-8"))
    except (FrontmatterError, OSError, UnicodeDecodeError) as exc:
        findings.append(Finding(Severity.ERROR, f"frontmatter unreadable: {exc}", document.path))
        return

    for number, line in enumerate(block.splitlines(), start=1):
        if not line.strip():
            continue
        _key, _, raw = line.partition(":")
        value = raw.strip()
        if value and not value.startswith(_ALLOWED_VALUE_STARTS):
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"line {number} holds a bare scalar ({value!r}); flat values are "
                    f"double-quoted and nested ones are YAML flow style",
                    document.path,
                )
            )

    try:
        strict = parse(block)
    except FrontmatterError as exc:
        findings.append(Finding(Severity.ERROR, str(exc), document.path))
        return
    try:
        loaded = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        findings.append(
            Finding(Severity.ERROR, f"frontmatter is not valid YAML: {exc}", document.path)
        )
        return
    if not isinstance(loaded, Mapping):
        findings.append(Finding(Severity.ERROR, "frontmatter is not a YAML mapping", document.path))
        return

    relaxed = {key: ("" if value is None else value) for key, value in loaded.items()}
    if relaxed != strict:
        differing = sorted(
            key
            for key in set(relaxed) | set(strict)
            if relaxed.get(key, object()) != strict.get(key, object())
        )
        findings.append(
            Finding(
                Severity.ERROR,
                "the line-parser and the YAML parser disagree about "
                f"{', '.join(differing)} — the block means two different things "
                "depending on who reads it",
                document.path,
            )
        )


def _check_sections(document: OkfDocument, findings: list[Finding]) -> None:
    headings = [name for name, _ in document.sections()]
    if headings != list(BODY_SECTIONS):
        findings.append(
            Finding(
                Severity.ERROR,
                f"body sections are {headings or ['none']}, expected "
                f"{list(BODY_SECTIONS)} in that order",
                document.path,
            )
        )
        return
    for name, text in document.sections():
        if not text.strip():
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"section `# {name}` is empty — an absent finding and an unwritten "
                    f"section are different claims",
                    document.path,
                )
            )


# --- catalog, links, identity -----------------------------------------------


def _check_catalog(bundle: OkfBundle, findings: list[Finding]) -> None:
    if not bundle.has_catalog:
        return
    catalog_path = bundle.path / CATALOG_FILENAME
    listed = {str(row.get("file") or "") for row in bundle.catalog}
    listed.discard("")
    present = {f"{doc.shelf}/{doc.filename}" for doc in bundle.documents()}

    for missing in sorted(present - listed):
        findings.append(
            Finding(Severity.ERROR, f"{missing} has no row in the catalog", catalog_path)
        )
    for orphan in sorted(listed - present):
        findings.append(
            Finding(Severity.ERROR, f"catalog names {orphan}, which is not in the bundle",
                    catalog_path)
        )


def _check_links(bundle: OkfBundle, findings: list[Finding]) -> None:
    """Every relative link inside the bundle must resolve."""
    documents = [bundle.index, *(shelf.index for shelf in bundle.shelves), *bundle.documents()]
    for document in documents:
        if document is None:
            continue
        for target in _LINK.findall(document.body):
            cleaned = target.split("#", 1)[0].strip()
            if not cleaned or "://" in cleaned or cleaned.startswith("mailto:"):
                continue
            base = bundle.path if cleaned.startswith("/") else document.path.parent
            resolved = (base / cleaned.lstrip("/")).resolve()
            if not resolved.exists():
                findings.append(
                    Finding(Severity.ERROR, f"link to {cleaned} does not resolve", document.path)
                )


def _check_ids(bundle: OkfBundle, findings: list[Finding]) -> None:
    """A handle that names two documents resolves to whichever one is found first."""
    seen: dict[str, Path] = {}
    for document in bundle.documents():
        handle = document.doc_id
        if handle in seen:
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"id {handle!r} is already used by {seen[handle].name}",
                    document.path,
                )
            )
            continue
        seen[handle] = document.path


# --- advisory ---------------------------------------------------------------


def _check_vocabulary(bundle: OkfBundle, findings: list[Finding]) -> None:
    """Aggregate what extractions wanted to record and the charter never allowed.

    Reported here rather than per paper because one extraction reaching for an unlisted
    key is a judgment call and a fifth of them reaching for the same one is a hole in the
    charter — and the difference is only visible once.
    """
    total = len(bundle.catalog)
    if not total:
        return
    counts: dict[str, int] = {}
    for row in bundle.catalog:
        unmapped = row.get("unmapped_vocab")
        if not isinstance(unmapped, Mapping):
            continue
        for key in unmapped:
            counts[str(key)] = counts.get(str(key), 0) + 1
    if not counts:
        return

    listed = _charter_vocabularies(bundle)
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        share = count / total
        message = (
            f"{count} of {total} document(s) recorded coding hints under {key!r}, which "
            f"the charter did not list — they are not in frontmatter"
        )
        if share >= UNMAPPED_SHARE:
            message += f". Rerun with:\n    {_rerun_command(bundle, listed, key)}"
        findings.append(Finding(Severity.WARNING, message, bundle.path / CATALOG_FILENAME))


def _charter_vocabularies(bundle: OkfBundle) -> list[str]:
    raw = bundle.charter.get("vocabularies")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _rerun_command(bundle: OkfBundle, listed: Iterable[str], key: str) -> str:
    vocabularies = [*listed, key]
    charter = bundle.path / CHARTER_FILENAME
    where = charter if charter.exists() else Path(CHARTER_FILENAME)
    return f"okf-loremaster build --charter {where} --vocab {','.join(vocabularies)}"


def _check_vector_index(bundle: OkfBundle, findings: list[Finding]) -> None:
    """The sidecar index, if there is one, and the bundle's pointer at it.

    All warnings, never errors. The index is derived and rebuildable in one command, so a
    broken one is not a reason to fail a bundle that is otherwise correct — but a dangling
    pointer or an undeclared distance metric is silent at the far end, which is exactly
    the kind of thing this file exists to say out loud.
    """
    store = vector_store_path(bundle.path)
    pointer = bundle.descriptor.get("vectors")
    pointed = isinstance(pointer, Mapping)

    if pointed and not store.is_dir():
        findings.append(
            Finding(
                Severity.WARNING,
                f"{DESCRIPTOR_FILENAME} points at a vector index at {store.name}, which is "
                f"not there — rebuild it with `okf-loremaster index {bundle.path}` or drop "
                f"the `vectors` key",
                bundle.path / DESCRIPTOR_FILENAME,
            )
        )
    if not store.is_dir():
        return

    if not pointed:
        findings.append(
            Finding(
                Severity.WARNING,
                f"a vector index exists at {store.name} but {DESCRIPTOR_FILENAME} does not "
                f"point at it, so a consumer that reads the bundle will not find it",
                bundle.path / DESCRIPTOR_FILENAME,
            )
        )

    descriptor_path = store / DESCRIPTOR_FILENAME
    if not descriptor_path.exists():
        findings.append(
            Finding(
                Severity.WARNING,
                f"no {DESCRIPTOR_FILENAME} — a prebuilt vector store cannot be attached "
                f"without one, because nothing else declares its embedding model",
                store,
            )
        )
        return
    try:
        loaded = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        findings.append(Finding(Severity.WARNING, f"unreadable: {exc}", descriptor_path))
        return
    if not isinstance(loaded, Mapping):
        findings.append(Finding(Severity.WARNING, "is not a YAML mapping", descriptor_path))
        return

    if not str(loaded.get("embedding_model") or ""):
        findings.append(
            Finding(
                Severity.WARNING,
                "declares no `embedding_model` — a consumer verifies the embedder on "
                "attach and cannot with nothing to check",
                descriptor_path,
            )
        )
    if not str(loaded.get("embedding_revision") or ""):
        findings.append(
            Finding(
                Severity.WARNING,
                "declares no `embedding_revision`, so the checkpoint that produced these "
                "vectors is not pinned and a rebuild may not reproduce them",
                descriptor_path,
            )
        )
    if str(loaded.get("distance") or "") not in DISTANCES:
        findings.append(
            Finding(
                Severity.WARNING,
                f"declares no usable `distance` (one of {', '.join(DISTANCES)}) — a "
                f"consumer that guesses wrong gets results in a different order and no "
                f"error at all",
                descriptor_path,
            )
        )


def _check_embedder(embed_model: str, findings: list[Finding]) -> None:
    name = embed_model.strip().lower()
    if name and name.startswith(REMOTE_EMBEDDERS):
        findings.append(
            Finding(
                Severity.WARNING,
                f"the configured embedding model ({embed_model}) looks remote. A "
                f"downstream consumer verifies the embedder on attach and rejects "
                f"anything that is not local, so the vector index would be refused.",
            )
        )
