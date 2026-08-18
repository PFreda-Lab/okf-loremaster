"""The shape of a bundle, stated once.

Every filename, every reserved heading, every frontmatter key that carries meaning to a
downstream reader lives here, because the writer and the reader must not be able to
disagree about them. `emitters/okf.py` builds a bundle from these names and
`okf/validate.py` checks one against the same names; a rename that touched only one side
would produce a bundle that validates against nothing.

The body headings are ordered, and the order is part of the contract. An agent reading
forty of these files at once relies on the same question being in the same place in each
— and `# Null or non-significant findings` sitting third from the end is what makes
"this was tested and did not hold" as findable as "this held".

Every heading is also a **named constant**, and nothing may index into `BODY_SECTIONS`
positionally. The vector emitter did exactly that, and inserting `# Abstract` second
would have silently redefined its `PREDICTORS_SECTION` as the abstract — a whole index
built from the wrong half of every document, with no error anywhere.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ABSTRACT_SECTION",
    "BODY_SECTIONS",
    "BOTTOM_LINE_SECTION",
    "CATALOG_FILENAME",
    "CAVEATS_SECTION",
    "CHARTER_FILENAME",
    "DESCRIPTOR_FILENAME",
    "DISTANCES",
    "DOCUMENT_TYPE",
    "FULL_TEXT_BASIS",
    "INDEX_FILENAME",
    "INTERACTIONS_SECTION",
    "INTERACTION_COLUMNS",
    "INTERACTION_SEPARATOR",
    "LOG_FILENAME",
    "NONE_CELL",
    "NULL_FINDINGS_SECTION",
    "OKF_DIRNAME",
    "PREDICTORS_FILENAME",
    "PREDICTORS_SECTION",
    "PREDICTOR_COLUMNS",
    "PREDICTOR_INDEX_TYPE",
    "QUOTE_LEAD",
    "REQUIRED_BODY_SECTIONS",
    "RESERVED_FILENAMES",
    "ROOT_INDEX_TYPE",
    "SEARCH_FILENAME",
    "SEARCH_STRATEGY_TYPE",
    "SITE_COLUMNS",
    "TOPIC_INDEX_TYPE",
    "UNVERIFIED_CELL",
    "VECTORS_DIRNAME",
    "VOCABULARY_SECTION",
    "okf_bundle_path",
    "vector_store_path",
]

INDEX_FILENAME = "index.md"
LOG_FILENAME = "log.md"
CATALOG_FILENAME = "_catalog.jsonl"
DESCRIPTOR_FILENAME = "resource_descriptor.yaml"
CHARTER_FILENAME = "charter.yaml"
PREDICTORS_FILENAME = "predictors.md"
# The search, written out to be repeated by hand. Separate from `log.md`, which reports
# the same queries in two lines each as part of a run's forensics; this one explains what
# every field means and what will and will not reproduce, because a methods section and a
# build log are read by different people looking for different things.
SEARCH_FILENAME = "search.md"

# Files at the root of a bundle or a topic that are ours to regenerate, never documents.
RESERVED_FILENAMES = frozenset(
    {
        INDEX_FILENAME,
        LOG_FILENAME,
        CATALOG_FILENAME,
        DESCRIPTOR_FILENAME,
        CHARTER_FILENAME,
        PREDICTORS_FILENAME,
        SEARCH_FILENAME,
    }
)

# The `type` frontmatter key. Free text as far as the spec is concerned, but the only
# field the spec actually requires, so it says what kind of thing the reader is holding.
DOCUMENT_TYPE = "Literature Evidence"
ROOT_INDEX_TYPE = "Bundle Index"
TOPIC_INDEX_TYPE = "Topic Index"
# `predictors.md`. Carries no `domain`, and cannot: a document's `domain` must equal the
# folder it sits in, and this one sits at the corpus root beside the folders. That is what
# keeps a file which cuts across every topic from being read as a paper in none of them.
PREDICTOR_INDEX_TYPE = "Predictor Index"
# `search.md`. Carries no `domain` for the same reason `predictors.md` does not.
SEARCH_STRATEGY_TYPE = "Search Strategy"

# `# ` headings, one constant each. Never `BODY_SECTIONS[n]`.
BOTTOM_LINE_SECTION = "Bottom line"
ABSTRACT_SECTION = "Abstract"
PREDICTORS_SECTION = "Predictors reported"
INTERACTIONS_SECTION = "Interactions"
NULL_FINDINGS_SECTION = "Null or non-significant findings"
VOCABULARY_SECTION = "Vocabulary hints"
CAVEATS_SECTION = "Caveats"

# The headings, in this order, in every concept file this version writes.
#
# `# Abstract` sits second because that is where it was asked for, and the placement is
# worth defending rather than apologizing for: an agent that has read the two-line bottom
# line and wants the author's own framing before the tables gets it in the next section,
# and one that wants the structured evidence scrolls past a block it can recognize by its
# heading. It is publisher text copied verbatim — see `emitters.okf._abstract` for what
# that means for redistribution.
#
# `# Interactions` sits directly under the table it expands, because every row in it is
# keyed to a `#` in that table and a reader who has to hunt for the key will not use it.
BODY_SECTIONS = (
    BOTTOM_LINE_SECTION,
    ABSTRACT_SECTION,
    PREDICTORS_SECTION,
    INTERACTIONS_SECTION,
    NULL_FINDINGS_SECTION,
    VOCABULARY_SECTION,
    CAVEATS_SECTION,
)

# The headings a document must carry whenever it was written. Every bundle ever emitted
# has these five, so the validator can require them without failing a corpus built before
# the other two existed — and it must not fail one, because a bundle is a deliverable that
# outlives the tool version that wrote it. The two headings outside this tuple are checked
# for position and emptiness when present and never for presence.
REQUIRED_BODY_SECTIONS = (
    BOTTOM_LINE_SECTION,
    PREDICTORS_SECTION,
    NULL_FINDINGS_SECTION,
    VOCABULARY_SECTION,
    CAVEATS_SECTION,
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
    # Names only, and last. A pointer at `# Interactions`, not a summary of it: the type
    # and the coefficient are what a reader needs to act, they do not fit in a cell beside
    # ten other columns, and a table that answers the question is a table nobody scrolls
    # past. Appended rather than inserted so anything reading these columns by position
    # keeps working.
    "Interacts with",
)

# `# Interactions`, where the pointer becomes the finding. `#` is the predictor table's
# row number and the only thing joining the two, which is why it leads here as it does
# there. One line per interaction rather than one per row: a predictor with three of them
# is making three claims, and a merged cell makes them one.
INTERACTION_COLUMNS = ("#", "Predictor", "Interacts with", "Type", "Magnitude", "Evidence")

# Separates the names in the predictor table's `Interacts with` cell. A semicolon rather
# than a comma because a variable's own name routinely holds one.
INTERACTION_SEPARATOR = "; "

# `predictors.md`, where every row is a pointer rather than a finding. `paper` and `row`
# come first and together they are the address: the document to open, and the `#` value to
# find once it is open. Everything after them exists to decide whether to make that trip.
SITE_COLUMNS = (
    "paper",
    "row",
    "topic",
    "as measured",
    "direction",
    "effect",
    "strength",
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
