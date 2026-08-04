"""OKF frontmatter: the single writer and the single reader.

The discipline this module exists to enforce is **one key per line, YAML flow style for
anything nested**, and it is enforced here rather than asked for at the call sites
because there is exactly one way to get it wrong and it is invisible in a diff.

Why the discipline exists at all. OKF v0.2 defines `generated`, `verified` and `sources`
as nested structures and derives its trust tiers from `verified` specifically, so
flattening them to `generated_by` / `generated_at` would forfeit conformance. But
writing them as indented block YAML breaks naive line-parsers — the downstream reader
skips lines without a colon and promotes indented keys to bogus top-level ones, so a
`sources:` list becomes a top-level `id` overwriting the document's own. Flow style on a
single line satisfies both readers at once: valid YAML for a spec consumer, one line per
field for grep and for anything that reads line by line.

That is why `parse` is deliberately *stricter* than `yaml.safe_load`. It is a
line-parser, and it refuses what a line-parser would misread — an indented continuation,
a line with no key, a duplicate key. `validate` then checks every emitted block through
both this parser and `yaml.safe_load` and requires them to agree, which is the only way
to know that the two audiences are seeing the same document.

Scalars are written double-quoted with their whitespace collapsed — numbers and booleans
included, for the reason given on `_scalar`. A newline inside a value would end the line
and therefore the key, so collapsing is not cosmetic — it is what makes "one key per
line" true rather than merely intended.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import yaml

__all__ = [
    "FENCE",
    "FIELD_ORDER",
    "FrontmatterError",
    "load",
    "parse",
    "render",
    "split",
    "stamp",
]

FENCE = "---"

# Preferred key order. Identity first, then provenance, then the nested blocks that have
# to be read as YAML. Keys not named here follow in the order the caller supplied them,
# so this is a preference rather than a whitelist — an index file's `domain_title` does
# not need a line in this tuple to be emitted.
FIELD_ORDER = (
    "type",
    "title",
    "description",
    "resource",
    "domain",
    "domain_title",
    "id",
    "pmid",
    "journal",
    "authors",
    "published",
    "tags",
    "study_design",
    "n",
    "text_basis",
    "license",
    "export_safe",
    "generated",
    "verified",
    "sources",
)

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class FrontmatterError(ValueError):
    """A frontmatter block this reader will not accept.

    Raised rather than repaired. A block that does not round-trip is a bug in whatever
    wrote it, and quietly reading it anyway would let the bug reach a bundle.
    """


# --- writing ----------------------------------------------------------------


def render(fields: Mapping[str, Any]) -> str:
    """Render a frontmatter block, fences included, one key per line.

    A key whose value is `None`, an empty string, or an empty collection is omitted:
    only `type` is required by the spec, so every other line has to be carrying
    something. `False` and `0` are values, not emptiness, and are written.
    """
    lines = [FENCE]
    for key in _ordered(fields):
        value = fields[key]
        if _is_empty(value):
            continue
        if not _KEY.match(key):
            raise FrontmatterError(f"frontmatter key {key!r} is not a plain identifier")
        lines.append(f"{key}: {_flow(value)}")
    lines.append(FENCE)
    return "\n".join(lines) + "\n"


def stamp(moment: datetime) -> str:
    """A timestamp as OKF writes them: UTC, second resolution, trailing `Z`.

    Rendered rather than left to `isoformat()`, which emits `+00:00` for an aware
    datetime and nothing at all for a naive one — two spellings of the same instant and
    one spelling of an unknown one.
    """
    from datetime import UTC

    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ordered(fields: Mapping[str, Any]) -> list[str]:
    known = [key for key in FIELD_ORDER if key in fields]
    rest = [key for key in fields if key not in FIELD_ORDER]
    return known + rest


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False


def _flow(value: Any) -> str:
    """One value, in YAML flow style, on one line."""
    if isinstance(value, Mapping):
        inner = ", ".join(f"{key}: {_flow(item)}" for key, item in value.items())
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow(item) for item in value) + "]"
    return _scalar(value)


def _scalar(value: Any) -> str:
    """Every flat scalar, double-quoted — numbers and booleans included.

    Not a stylistic preference. The downstream contract is "strictly quoted flat scalars
    and string lists", against a dependency-free parser that hands back strings anyway;
    an unquoted `n: 1454` is typed for a YAML consumer and a string for a line-parser,
    which is the one thing the flow-style discipline exists to rule out. Quoted, both
    readers get `"1454"` and the two agree.
    """
    if isinstance(value, bool):
        return '"true"' if value else '"false"'
    if isinstance(value, int):
        return _quote(str(value))
    if isinstance(value, float):
        return _quote(repr(value))
    if isinstance(value, datetime):
        return _quote(stamp(value))
    if isinstance(value, date):
        return _quote(value.isoformat())
    return _quote(str(value))


def _quote(text: str) -> str:
    """A double-quoted YAML scalar that cannot break the line it is on."""
    collapsed = " ".join(_CONTROL.sub(" ", text).split())
    escaped = collapsed.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# --- reading ----------------------------------------------------------------


def split(text: str) -> tuple[str, str]:
    """Split a document into (frontmatter block without fences, body).

    A document with no opening fence raises: an OKF file without frontmatter has no
    `domain`, so treating it as a body-only document would produce a document that
    cannot be filed.
    """
    lines = text.splitlines()
    start = 0
    # A leading byte-order mark or blank line is a copy-paste artifact, not a document
    # without frontmatter.
    while start < len(lines) and not lines[start].strip().lstrip("﻿"):
        start += 1
    if start >= len(lines) or lines[start].strip().lstrip("﻿") != FENCE:
        raise FrontmatterError("document does not open with a --- frontmatter fence")

    for index in range(start + 1, len(lines)):
        if lines[index].strip() == FENCE:
            block = "\n".join(lines[start + 1 : index])
            body = "\n".join(lines[index + 1 :])
            return block, body.lstrip("\n")
    raise FrontmatterError("frontmatter block is never closed by a --- fence")


def parse(block: str) -> dict[str, Any]:
    """Parse a frontmatter block the way a line-parser has to.

    Every non-blank line must be `key: <value>` with the key at column zero and the
    value complete on that line. Anything else is what the flow-style discipline exists
    to prevent, so it is an error here rather than a silently different document.
    """
    fields: dict[str, Any] = {}
    for number, line in enumerate(block.splitlines(), start=1):
        if not line.strip():
            continue
        if line[0].isspace():
            raise FrontmatterError(
                f"line {number} is indented; frontmatter is one key per line, with "
                f"nested values in YAML flow style: {line.strip()!r}"
            )
        key, separator, raw = line.partition(":")
        if not separator:
            raise FrontmatterError(f"line {number} has no key: {line.strip()!r}")
        key = key.strip()
        if not _KEY.match(key):
            raise FrontmatterError(f"line {number} has an unusable key: {key!r}")
        if key in fields:
            raise FrontmatterError(f"line {number} repeats the key {key!r}")
        fields[key] = _value(raw, number)
    return fields


def _value(raw: str, number: int) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise FrontmatterError(f"line {number} is not valid YAML: {text!r}") from exc
    return "" if loaded is None else loaded


def load(text: str) -> tuple[dict[str, Any], str]:
    """A whole document as (fields, body)."""
    block, body = split(text)
    return parse(block), body
