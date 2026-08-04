"""The validate node: read the bundle back off disk and check it. No model call.

Reading it back rather than checking the objects that produced it is the whole point. A
run that validated its own in-memory records would prove that the pipeline agrees with
itself and nothing about the files a downstream agent will open — and every defect this
gate exists to catch lives in the gap between the two.

Failures are reported, never raised. A bundle that fails the gate is still on disk and
still worth looking at; the run's exit code carries the verdict, and the errors are named
so the fix is obvious. That is also why the node runs last rather than gating the write:
you cannot inspect a bundle that was never written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.okf.validate import validate_bundle

__all__ = ["validate_node"]

NODE = "validate"

# Errors named in the run's warning list. The rest are in the report the CLI prints; a
# structurally broken bundle can produce one per document, and a warning list two hundred
# lines long is a warning list nobody reads.
MAX_REPORTED = 10


async def validate_node(state: RunState, deps: Deps) -> dict[str, Any]:
    location = state.get("bundle")
    if not location:
        raise RuntimeError("validate reached without a bundle — the graph is wired wrong")

    path = Path(location)
    warnings = list(state.get("warnings") or [])

    with span(deps, NODE) as report:
        result = validate_bundle(path, embed_model=deps.settings.embed_model)
        errors = [finding.line(relative_to=path) for finding in result.errors]

        for finding in result.warnings:
            note = finding.line(relative_to=path)
            warnings.append(note)
            deps.warn(NODE, note)

        for note in errors[:MAX_REPORTED]:
            deps.warn(NODE, f"invalid: {note}")
        if len(errors) > MAX_REPORTED:
            deps.warn(NODE, f"and {len(errors) - MAX_REPORTED} further validation error(s)")
        if errors:
            warnings.append(
                f"the bundle failed validation with {len(errors)} error(s) — "
                f"run `okf-loremaster validate {path}` for the full report"
            )

        report["summary"] = result.summary()

    return {
        "validated": result.ok,
        "validation_errors": errors,
        "warnings": warnings,
    }
