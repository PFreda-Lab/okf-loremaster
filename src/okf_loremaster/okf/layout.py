"""The shape of a bundle, stated once.

Every filename, every reserved heading, every frontmatter key that carries meaning to a
downstream reader lives here, because the writer and the reader must not be able to
disagree about them. `emitters/okf.py` builds a bundle from these names and
`okf/validate.py` checks one against the same names; a rename that touched only one side
would produce a bundle that validates against nothing.

The five body headings are ordered, and the order is part of the contract. An agent
reading forty of these files at once relies on the same question being in the same place
in each — and `# Null or non-significant findings` sitting fourth from the end is what
makes "this was tested and did not hold" as findable as "this held".
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "BODY_SECTIONS",
    "CATALOG_FILENAME",
    "CHARTER_FILENAME",
    "DESCRIPTOR_FILENAME",
    "DISTANCES",
    "DOCUMENT_TYPE",
    "FULL_TEXT_BASIS",
    "INDEX_FILENAME",
    "LOG_FILENAME",
    "NONE_CELL",
    "PREDICTOR_COLUMNS",
    "QUOTE_LEAD",
    "RESERVED_FILENAMES",
    "ROOT_INDEX_TYPE",
    "SHELF_INDEX_TYPE",
    "UNVERIFIED_CELL",
    "VECTOR_SUFFIX",
    "vector_store_path",
]

INDEX_FILENAME = "index.md"
LOG_FILENAME = "log.md"
CATALOG_FILENAME = "_catalog.jsonl"
DESCRIPTOR_FILENAME = "resource_descriptor.yaml"
CHARTER_FILENAME = "charter.yaml"

# Files at the root of a bundle or a shelf that are ours to regenerate, never documents.
RESERVED_FILENAMES = frozenset(
    {INDEX_FILENAME, LOG_FILENAME, CATALOG_FILENAME, DESCRIPTOR_FILENAME, CHARTER_FILENAME}
)

# The `type` frontmatter key. Free text as far as the spec is concerned, but the only
# field the spec actually requires, so it says what kind of thing the reader is holding.
DOCUMENT_TYPE = "Literature Evidence"
ROOT_INDEX_TYPE = "Bundle Index"
SHELF_INDEX_TYPE = "Shelf Index"

# `# ` headings, in this order, in every concept file.
BODY_SECTIONS = (
    "Bottom line",
    "Predictors reported",
    "Null or non-significant findings",
    "Vocabulary hints",
    "Caveats",
)

# The predictor table's columns, in order. Here rather than in the emitter because the
# vector index parses this table back out of a finished bundle: a column renamed on the
# writing side alone would produce chunks whose metadata is silently empty.
PREDICTOR_COLUMNS = (
    "#",
    "Predictor",
    "Operationalization",
    "Timing",
    "Outcome",
    "Type",
    "Effect",
    "p",
    "Direction",
    "Confidence",
)

# Introduces the numbered verbatim quotes under a table, keyed to its `#` column. Shared
# for the same reason as the columns: it is how a reader finds the quotes again.
QUOTE_LEAD = "Quoted from the paper, by row:"

# An empty table cell. A dash rather than blank so a row's shape survives a reader that
# collapses whitespace, and so "nothing here" is visibly deliberate.
NONE_CELL = "—"

# What `text_basis` reads as for a paper read from full text. The underscore is the trap:
# the prose everywhere says "full text", the value on disk does not, and a reader that
# compares against the prose reports an entire corpus as abstract-only without erroring.
# `TextBasis.FULL_TEXT` is the writing side of the same string; a test holds them equal.
FULL_TEXT_BASIS = "full_text"

# What a magnitude that numeric verification removed renders as. Deliberately not
# `NONE_CELL`: "the paper reported no effect size" and "the effect size it reported is
# not in the text we read" are different claims, and collapsing them into one blank cell
# is exactly the confusion the verification pass exists to prevent.
UNVERIFIED_CELL = "unverified"

# The derived vector index sits *beside* the bundle, never inside it.
VECTOR_SUFFIX = ".chroma"

# The distance metrics a consumer knows how to honor. Here rather than in the emitter
# because the validator checks the same list: a store built with one metric and queried
# as another returns a different order and no error, so both sides must agree on which
# names are even sayable.
DISTANCES = ("cosine", "l2", "ip")


def vector_store_path(bundle: Path) -> Path:
    """`<bundle>.chroma`, a sibling of the bundle directory.

    A sibling rather than a subdirectory for three reasons: `read_bundle` treats every
    directory at the root as a shelf, so an index inside would validate as a shelf with
    no papers; the store is binary and would otherwise be copied by anything that copies
    a bundle; and it is derived, so deleting it costs nothing but deleting a shelf costs
    a rebuild.
    """
    resolved = bundle if bundle.name else bundle.resolve()
    return resolved.parent / f"{resolved.name}{VECTOR_SUFFIX}"
