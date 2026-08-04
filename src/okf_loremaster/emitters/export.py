"""Copying a bundle out, optionally down to what may be redistributed.

Reads a finished bundle off disk and writes a second one, rather than re-rendering from a
run: an export is a thing you do to a bundle someone handed you, possibly months after the
run that produced it and possibly on a machine with no API key. `read_bundle` is the only
input.

Five decisions, all of them the kind that is wrong quietly:

**A retained document is copied byte for byte.** Not re-rendered from its parsed fields.
The body holds verbatim quotes from the source paper, and a round trip through a renderer
is exactly where "reproduced exactly as published" stops being true. The indexes, the
catalog and the descriptor *are* rewritten, because they describe a set that changed.

**`export_safe` is compared as a string.** Frontmatter quotes every scalar, so the key
reads back as `"false"` — which is truthy. Testing it for truth would copy the whole
bundle and report it as filtered, which is the failure that matters here: it does not
error, it silently redistributes.

**The recorded license is checked against the flag, and disagreement excludes.** The flag
is derived from the license by this tool, so the two can only disagree in a bundle that
was edited. On a question of what may be given to someone else, the export takes the
conservative side and says so in a warning.

**The copy gets its own descriptor `id`.** A consumer keys on it — AFCE calls it the
resource id — so an export sharing the source's id is two different corpora claiming to
be the same one. The `vectors` pointer is dropped for the same reason it is not copied:
the store still embeds every document, including the ones the filter just removed.

**An emptied shelf keeps its directory and its index**, saying that nothing on it was
redistributable. Same reason the emitter keeps a shelf that retained no papers: an absent
shelf and an empty one are different claims, and the taxonomy still reads the same.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from okf_loremaster.emitters.okf import SHELF_COLUMNS, SHELF_PREDICTORS
from okf_loremaster.okf.frontmatter import render
from okf_loremaster.okf.layout import (
    BODY_SECTIONS,
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    INDEX_FILENAME,
    LOG_FILENAME,
    NONE_CELL,
    PREDICTOR_COLUMNS,
    ROOT_INDEX_TYPE,
    SHELF_INDEX_TYPE,
)
from okf_loremaster.okf.markdown import facts, inline, table_row, table_rule
from okf_loremaster.okf.reader import OkfBundle, OkfDocument, OkfShelf, markdown_table, read_bundle
from okf_loremaster.schemas.common import is_export_safe

__all__ = ["ExportResult", "export_bundle"]

# The frontmatter keys this module reads by name. The emitter writes both.
_LICENSE = "license"
_DESIGN = "study_design"

_PREDICTORS_SECTION = BODY_SECTIONS[1]
_PREDICTOR_NAME = PREDICTOR_COLUMNS[1]

_ROOT_COLUMNS = ("shelf", "title", "papers", "full text", "abstract only", "scope")


@dataclass(frozen=True, slots=True)
class ExportResult:
    """What `export_bundle` wrote, for the CLI to report."""

    path: Path
    documents: int
    shelves: int
    permissive_only: bool
    # `(file, license)` for each document the filter left behind, so the caller can say
    # what was not copied rather than only how much.
    omitted: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def omitted_count(self) -> int:
        return len(self.omitted)


def export_bundle(
    source: Path, destination: Path, *, permissive_only: bool = False
) -> ExportResult:
    """Write a copy of `source` at `destination`, filtered if asked.

    Refuses rather than merges: a destination that already holds files, or one nested in
    the source, stops the export before anything is written.
    """
    bundle = read_bundle(source)
    _refuse_bad_destination(source, destination)

    warnings = [f"{path.name} could not be read and was not copied: {why}"
                for path, why in bundle.problems]

    kept: dict[str, list[OkfDocument]] = {}
    omitted: list[tuple[str, str]] = []
    for shelf in bundle.shelves:
        keep: list[OkfDocument] = []
        for document in shelf.documents:
            if not permissive_only:
                keep.append(document)
                continue
            license_text = str(document.fields.get(_LICENSE) or "")
            flagged, derived = document.export_safe, is_export_safe(license_text)
            if flagged != derived:
                warnings.append(
                    f"{shelf.slug}/{document.filename} records export_safe="
                    f"{str(flagged).lower()} but its license ({license_text or 'none'}) reads "
                    f"as {str(derived).lower()}; it was left behind"
                )
            if flagged and derived:
                keep.append(document)
            else:
                omitted.append((f"{shelf.slug}/{document.filename}", license_text or "none"))
        kept[shelf.slug] = keep

    total = sum(len(items) for items in kept.values())
    if not total:
        raise ValueError(
            f"nothing to export from {source}: "
            + (
                "no document carries a license that permits redistribution"
                if permissive_only
                else "the bundle holds no documents"
            )
        )

    scopes = _scopes(bundle)
    destination.mkdir(parents=True, exist_ok=True)
    for shelf in bundle.shelves:
        directory = destination / shelf.slug
        directory.mkdir(parents=True, exist_ok=True)
        for document in kept[shelf.slug]:
            # Byte for byte: the body holds quotes reproduced exactly as published.
            shutil.copyfile(document.path, directory / document.filename)
        (directory / INDEX_FILENAME).write_text(
            _shelf_index(shelf, kept[shelf.slug], scope=scopes.get(shelf.slug, "")),
            encoding="utf-8",
        )

    (destination / INDEX_FILENAME).write_text(
        _root_index(
            bundle, kept, scopes=scopes, permissive_only=permissive_only, omitted=len(omitted)
        ),
        encoding="utf-8",
    )
    (destination / CATALOG_FILENAME).write_text(_catalog(bundle, kept), encoding="utf-8")
    (destination / DESCRIPTOR_FILENAME).write_text(
        _descriptor(bundle, kept, permissive_only=permissive_only, omitted=len(omitted)),
        encoding="utf-8",
    )
    for name in (LOG_FILENAME, CHARTER_FILENAME):
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)

    return ExportResult(
        path=destination,
        documents=total,
        shelves=len(bundle.shelves),
        permissive_only=permissive_only,
        omitted=tuple(omitted),
        warnings=tuple(warnings),
    )


def _refuse_bad_destination(source: Path, destination: Path) -> None:
    """Everything that has to be true before a single file is written."""
    here, there = source.resolve(), destination.resolve()
    if here == there:
        raise ValueError(f"the destination is the bundle itself: {destination}")
    if there.is_relative_to(here):
        raise ValueError(
            f"{destination} is inside {source}; an export written into its own source "
            f"would be read as a shelf of it"
        )
    if here.is_relative_to(there):
        raise ValueError(f"{destination} contains {source}; pick a directory of its own")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"{destination} is a file, not a directory")
    if destination.is_dir() and any(destination.iterdir()):
        raise ValueError(
            f"{destination} is not empty; export writes a whole bundle and will not merge "
            f"into one that already holds files"
        )


# --- the rewritten indexes ---------------------------------------------------


def _shelf_index(shelf: OkfShelf, documents: Sequence[OkfDocument], *, scope: str) -> str:
    """The source shelf index, re-rendered for the documents that survived."""
    title = shelf.title or shelf.slug
    fields: dict[str, Any] = {
        "type": SHELF_INDEX_TYPE,
        "title": title,
        "description": scope or f"{len(documents)} paper(s) on {title}.",
        "domain": shelf.slug,
        "domain_title": title,
    }

    body = [f"# {title}", ""]
    if scope:
        body += [scope, ""]
    if not documents:
        body += [
            f"No paper on this shelf carries a license that permits redistribution. The "
            f"shelf is kept so the taxonomy still reads the same; the source bundle holds "
            f"{len(shelf.documents)} paper(s) here.",
            "",
            f"[← bundle index](../{INDEX_FILENAME})",
            "",
        ]
        return render(fields) + "\n" + "\n".join(body)

    full = sum(1 for document in documents if document.full_text)
    body += [
        f"{len(documents)} paper(s) — {full} read from full text, "
        f"{len(documents) - full} from the abstract only.",
        "",
        table_row(SHELF_COLUMNS),
        table_rule(len(SHELF_COLUMNS)),
    ]
    for document in documents:
        body.append(
            table_row(
                (
                    document.pmid,
                    # Escaping is `table_row`'s job; escaping here as well would turn a
                    # pipe in a title into a backslash and a column break.
                    f"[{document.title}](./{document.filename})",
                    str(document.fields.get(_DESIGN) or ""),
                    _n_cell(document),
                    _key_predictors(document),
                )
            )
        )
    body += ["", f"[← bundle index](../{INDEX_FILENAME})", ""]
    return render(fields) + "\n" + "\n".join(body)


def _root_index(
    bundle: OkfBundle,
    kept: dict[str, list[OkfDocument]],
    *,
    scopes: dict[str, str],
    permissive_only: bool,
    omitted: int,
) -> str:
    total = sum(len(items) for items in kept.values())
    source_index = bundle.index
    title = (source_index.title if source_index is not None else "") or bundle.path.name
    filter_line = (
        "documents whose recorded license permits redistribution"
        if permissive_only
        else "every document in the source bundle"
    )
    fields: dict[str, Any] = {
        "type": ROOT_INDEX_TYPE,
        "title": title,
        "description": (
            f"{total} paper(s) across {len(kept)} shelf/shelves — an export of "
            f"{bundle.path.name} holding {filter_line}."
        ),
    }

    body = [f"# {title}", ""]
    preamble = _preamble(source_index)
    if preamble:
        body += [preamble, ""]
    body += [
        f"> An export of a bundle, not a build of one. It holds {filter_line}; "
        f"`{LOG_FILENAME}` describes the run that produced the source.",
        "",
        "## Shelves",
        "",
        table_row(_ROOT_COLUMNS),
        table_rule(len(_ROOT_COLUMNS)),
    ]
    for shelf in bundle.shelves:
        documents = kept[shelf.slug]
        full = sum(1 for document in documents if document.full_text)
        body.append(
            table_row(
                (
                    f"[{shelf.slug}]({shelf.slug}/{INDEX_FILENAME})",
                    shelf.title,
                    str(len(documents)),
                    str(full),
                    str(len(documents) - full),
                    scopes.get(shelf.slug, ""),
                )
            )
        )

    body += ["", "## Export", ""]
    body += facts(
        [
            ("Source", str(bundle.descriptor.get("id") or bundle.path.name)),
            ("Filter", "permissive licenses only" if permissive_only else "none"),
            ("Documents", f"{total} of {bundle.document_count}"),
            (
                "Omitted",
                f"{omitted} under a license that does not permit redistribution"
                if omitted
                else "",
            ),
            ("Tool", "okf-loremaster export"),
        ]
    )

    # Only files that were actually copied: a link to a missing one is a validator error.
    files = [(LOG_FILENAME, "how the source bundle was built"),
             (CHARTER_FILENAME, "the charter the source was built from")]
    body += ["", "## Files", ""]
    body += [f"- [{name}]({name}) — {why}" for name, why in files
             if (bundle.path / name).exists()]
    body += [
        f"- `{CATALOG_FILENAME}` — one JSON row per document",
        f"- [{DESCRIPTOR_FILENAME}]({DESCRIPTOR_FILENAME}) — what a consumer reads on attach",
        "",
    ]
    return render(fields) + "\n" + "\n".join(body)


def _catalog(bundle: OkfBundle, kept: dict[str, list[OkfDocument]]) -> str:
    """The source catalog, filtered to the retained files, in shelf order.

    Filtered rather than rebuilt: the catalog carries `unmapped_vocab`, which is
    deliberately not in frontmatter and so cannot be recovered from the documents.
    """
    by_file = {str(row.get("file") or ""): row for row in bundle.catalog}
    lines: list[str] = []
    for slug, documents in kept.items():
        for document in documents:
            key = f"{slug}/{document.filename}"
            row = by_file.get(key)
            if row is None:
                # A document with no catalog row: emit what the file itself carries so
                # the catalog still describes every file, which `validate` requires.
                row = {
                    "pmid": document.pmid,
                    "title": document.title,
                    "domain": slug,
                    "file": key,
                    "description": str(document.fields.get("description") or ""),
                    "design": str(document.fields.get(_DESIGN) or ""),
                    "n": document.n,
                    "tags": document.tags,
                }
            lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return "".join(lines)


def _descriptor(
    bundle: OkfBundle,
    kept: dict[str, list[OkfDocument]],
    *,
    permissive_only: bool,
    omitted: int,
) -> str:
    source = bundle.descriptor
    source_id = str(source.get("id") or bundle.path.name)
    total = sum(len(items) for items in kept.values())
    payload: dict[str, Any] = {
        "kind": "okf",
        # Its own id: a consumer keys on this, and two corpora claiming one id is a
        # collision it has no way to notice.
        "id": f"{source_id}-{'permissive' if permissive_only else 'export'}",
        "name": str(source.get("name") or bundle.path.name),
        "description": (
            f"{total} of {bundle.document_count} papers from {source_id}, "
            + ("filtered to licenses that permit redistribution." if permissive_only
               else "copied without a license filter.")
        ),
        "index": INDEX_FILENAME,
        "catalog": CATALOG_FILENAME,
    }
    for name, key in ((LOG_FILENAME, "log"), (CHARTER_FILENAME, "charter")):
        if (bundle.path / name).exists():
            payload[key] = name
    payload["domains"] = {shelf.slug: shelf.title or shelf.slug for shelf in bundle.shelves}
    payload["documents"] = total
    for key in ("tool", "tool_version", "charter_digest", "built_on", "stale_after",
                "verified_by"):
        if source.get(key):
            payload[key] = source[key]
    payload["derived_from"] = source_id
    payload["export_filter"] = "permissive-only" if permissive_only else "none"
    payload["omitted"] = omitted
    # No `vectors` pointer: the store is not copied, and it embeds every document in the
    # source — including the ones the filter just removed.
    return str(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))


# --- reading one document ----------------------------------------------------


def _n_cell(document: OkfDocument) -> str:
    return NONE_CELL if document.n is None else f"{document.n:,}"


def _key_predictors(document: OkfDocument) -> str:
    """The shelf table's last column, read back out of the document's own table."""
    rows = markdown_table(document.section(_PREDICTORS_SECTION) or "")
    names = [row.get(_PREDICTOR_NAME, "") for row in rows]
    names = [name for name in names if name]
    shown = names[:SHELF_PREDICTORS]
    if not shown:
        return NONE_CELL
    extra = len(names) - len(shown)
    return ", ".join(shown) + (f", +{extra} more" if extra > 0 else "")


def _scopes(bundle: OkfBundle) -> dict[str, str]:
    """Each shelf's scope, from the charter where there is one.

    The charter rather than the shelf index because a shelf with no scope gets a
    generated description — "N paper(s) on X" — and N is the source's count, which is the
    one number a filtered export must not repeat.
    """
    taxonomy = bundle.charter.get("shelf_taxonomy")
    if isinstance(taxonomy, list):
        return {
            str(entry.get("slug") or ""): inline(str(entry.get("scope") or ""))
            for entry in taxonomy
            if isinstance(entry, dict)
        }
    return {
        shelf.slug: inline(str(shelf.index.fields.get("description") or ""))
        for shelf in bundle.shelves
        if shelf.index is not None
    }


def _preamble(index: OkfDocument | None) -> str:
    """The prose line under the source root index's title — the original prompt."""
    if index is None:
        return ""
    for line in index.body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "|", "-", ">")):
            return inline(stripped)
    return ""
