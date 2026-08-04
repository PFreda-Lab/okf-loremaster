"""Turning a finished run into files on disk.

`okf.py` writes the bundle; `vectors.py` derives the Chroma index from it — by walking
the finished bundle, never by extracting a second time; `export.py` writes a filtered copy
of one, also by walking it. That ordering is the reason these live together and apart from
`okf/`: an emitter depends on the format, the format does not depend on an emitter. Only
`okf.py` needs a run to have happened, which is why the other two work on a bundle that
arrived from somewhere else.

`vectors` is importable without the `[vectors]` extra installed — chromadb and
sentence-transformers are imported inside the two places that touch them, not at module
scope. That is what lets the graph wire `index_vectors` in unconditionally and lets the
CLI fail with "install the extra" rather than an ImportError traceback.
"""

from __future__ import annotations

from okf_loremaster.emitters.export import ExportResult, export_bundle
from okf_loremaster.emitters.okf import (
    BundleWrite,
    body_for,
    catalog_row,
    document_for,
    frontmatter_for,
    log_markdown,
    write_bundle,
)

__all__ = [
    "BundleWrite",
    "ExportResult",
    "body_for",
    "catalog_row",
    "document_for",
    "export_bundle",
    "frontmatter_for",
    "log_markdown",
    "write_bundle",
]
