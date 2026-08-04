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
    "OKF_DIRNAME",
    "PREDICTOR_COLUMNS",
    "QUOTE_LEAD",
    "RESERVED_FILENAMES",
    "ROOT_INDEX_TYPE",
    "TOPIC_INDEX_TYPE",
    "UNVERIFIED_CELL",
    "VECTORS_DIRNAME",
    "okf_bundle_path",
    "vector_store_path",
]

INDEX_FILENAME = "index.md"
LOG_FILENAME = "log.md"
CATALOG_FILENAME = "_catalog.jsonl"
DESCRIPTOR_FILENAME = "resource_descriptor.yaml"
CHARTER_FILENAME = "charter.yaml"

# Files at the root of a bundle or a topic that are ours to regenerate, never documents.
RESERVED_FILENAMES = frozenset(
    {INDEX_FILENAME, LOG_FILENAME, CATALOG_FILENAME, DESCRIPTOR_FILENAME, CHARTER_FILENAME}
)

# The `type` frontmatter key. Free text as far as the spec is concerned, but the only
# field the spec actually requires, so it says what kind of thing the reader is holding.
DOCUMENT_TYPE = "Literature Evidence"
ROOT_INDEX_TYPE = "Bundle Index"
TOPIC_INDEX_TYPE = "Topic Index"

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
    # Two different questions, side by side on purpose. `Confidence` is whether the row
    # was read correctly; `Strength` is how much weight the study behind it can carry. A
    # well-read row from a small unadjusted survey is `high` and `limited`, and a reader
    # who sees only one of the two columns draws the wrong conclusion from either.
    "Strength",
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

# A run writes one folder holding two resources, so moving the deliverable is one copy.
# The vector store is a *sibling* of the OKF corpus, never inside it: `read_bundle`
# treats every directory at the bundle root as a topic, so a store nested there would
# validate as a topic with no papers.
OKF_DIRNAME = "okf"
VECTORS_DIRNAME = "vectors"

# The distance metrics a consumer knows how to honor. Here rather than in the emitter
# because the validator checks the same list: a store built with one metric and queried
# as another returns a different order and no error, so both sides must agree on which
# names are even sayable.
DISTANCES = ("cosine", "l2", "ip")


def okf_bundle_path(run: Path) -> Path:
    """`<run>/okf` — the OKF corpus inside a run's output folder."""
    resolved = run if run.name else run.resolve()
    return resolved / OKF_DIRNAME


def vector_store_path(bundle: Path) -> Path:
    """`<run>/vectors`, given the OKF corpus at `<run>/okf`.

    Takes the corpus rather than the run folder because every caller holds the corpus:
    the emitter walks it, the validator reads it, and the overview reports on it. A
    sibling rather than a subdirectory because `read_bundle` treats every directory at
    the corpus root as a topic, and because the store is derived — deleting it costs an
    embedding pass, deleting a topic costs a whole run.
    """
    resolved = bundle if bundle.name else bundle.resolve()
    return resolved.parent / VECTORS_DIRNAME
