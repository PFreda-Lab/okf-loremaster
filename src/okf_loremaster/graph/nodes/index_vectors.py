"""The index_vectors node: derive a vector store from the finished bundle. No model call.

Last in the graph, and after `validate` rather than before it. The order is deliberate:
the index is built by reading the bundle back off disk, so it must be built from the same
files the validator judged — and a bundle that failed the gate still gets an index,
because the errors that gate reports are things a person fixes in a file, not reasons to
withhold a derived artifact.

Skipped entirely under `--finalize okf`. There is no default embedder: `deps.embedder` is
`None` unless something upstream built one, so a run that never asked to index cannot
accidentally download a model.

Failures here are warnings, not exceptions. The bundle is the deliverable and it is
already written; an index that could not be built is a rebuildable inconvenience, and
crashing the run at the last node would be a worse answer than saying so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from okf_loremaster.emitters.vectors import build_index
from okf_loremaster.graph.state import Deps, RunState, span

__all__ = ["index_vectors_node"]

NODE = "index_vectors"


async def index_vectors_node(state: RunState, deps: Deps) -> dict[str, Any]:
    if deps.embedder is None:
        # Not an error and not worth an event: the graph runs this node on every run,
        # and "you did not ask for an index" is not news.
        return {}

    location = state.get("bundle")
    if not location:
        raise RuntimeError("index_vectors reached without a bundle — the graph is wired wrong")

    path = Path(location)
    warnings = list(state.get("warnings") or [])

    with span(deps, NODE) as report:
        deps.progress(NODE, f"embedding with {deps.embedder.model_id}")
        try:
            result = await build_index(
                path,
                embedder=deps.embedder,
                on_progress=lambda done, total: deps.progress(
                    NODE, f"embedded {done}/{total} chunks", current=done, total=total
                ),
            )
        except Exception as exc:  # every failure here leaves the bundle itself intact
            note = f"the vector index could not be built: {exc}"
            warnings.append(note)
            deps.warn(NODE, note)
            report["summary"] = "no index"
            return {"vector_index": "", "warnings": warnings}

        for note in result.warnings:
            warnings.append(note)
            deps.warn(NODE, note)
        report["summary"] = result.summary()

    manifest = state.get("manifest")
    update: dict[str, Any] = {
        "vector_index": str(result.path) if result.chunks else "",
        "vector_chunks": result.chunks,
        "warnings": warnings,
    }
    if manifest is not None and result.chunks:
        # The manifest promises the *resolved* model and revision, which is why it is
        # filled in here rather than at emit time: until something has actually been
        # embedded, nobody knows which checkpoint answered.
        update["manifest"] = manifest.model_copy(
            update={
                "embed_model": result.embed_model,
                "embed_revision": result.embed_revision,
            }
        )
    return update
