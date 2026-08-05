"""The Open Knowledge Format itself: its shape, its frontmatter, and how to check it.

Deliberately separate from `emitters/`. This package knows what a bundle *is*; the
emitter knows how to turn a run into one. Keeping the reader and the validator here —
and importing neither from the writer — is what lets `okf-loremaster validate` be
meaningful against a bundle this tool did not produce.
"""

from __future__ import annotations

from okf_loremaster.okf.frontmatter import (
    FIELD_ORDER,
    FrontmatterError,
    load,
    parse,
    render,
    split,
    stamp,
)
from okf_loremaster.okf.layout import (
    BODY_SECTIONS,
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    DOCUMENT_TYPE,
    FULL_TEXT_BASIS,
    INDEX_FILENAME,
    LOG_FILENAME,
    NONE_CELL,
    PREDICTOR_INDEX_TYPE,
    PREDICTORS_FILENAME,
    RESERVED_FILENAMES,
    ROOT_INDEX_TYPE,
    SEARCH_FILENAME,
    SEARCH_STRATEGY_TYPE,
    TOPIC_INDEX_TYPE,
    UNVERIFIED_CELL,
)
from okf_loremaster.okf.markdown import cell, facts, inline, table_row, table_rule
from okf_loremaster.okf.reader import (
    OkfBundle,
    OkfDocument,
    OkfTopic,
    body_sections,
    fact_list,
    markdown_table,
    read_bundle,
)
from okf_loremaster.okf.validate import (
    BundleReport,
    Finding,
    Severity,
    validate_bundle,
)

__all__ = [
    "BODY_SECTIONS",
    "CATALOG_FILENAME",
    "CHARTER_FILENAME",
    "DESCRIPTOR_FILENAME",
    "DOCUMENT_TYPE",
    "FIELD_ORDER",
    "FULL_TEXT_BASIS",
    "INDEX_FILENAME",
    "LOG_FILENAME",
    "NONE_CELL",
    "PREDICTORS_FILENAME",
    "PREDICTOR_INDEX_TYPE",
    "RESERVED_FILENAMES",
    "ROOT_INDEX_TYPE",
    "SEARCH_FILENAME",
    "SEARCH_STRATEGY_TYPE",
    "TOPIC_INDEX_TYPE",
    "UNVERIFIED_CELL",
    "BundleReport",
    "Finding",
    "FrontmatterError",
    "OkfBundle",
    "OkfDocument",
    "OkfTopic",
    "Severity",
    "body_sections",
    "cell",
    "fact_list",
    "facts",
    "inline",
    "load",
    "markdown_table",
    "parse",
    "read_bundle",
    "render",
    "split",
    "stamp",
    "table_row",
    "table_rule",
    "validate_bundle",
]
