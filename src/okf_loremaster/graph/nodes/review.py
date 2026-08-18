"""The review node: offer the records for sign-off, and stamp them if it is given.

Bound through `Deps.reviewer` rather than added as a third graph interrupt. The two
existing interrupts sit at the boundaries where resuming saves real money — before
screening thousands of abstracts, before reading hundreds of papers. `reconcile` to
`emit_okf` is neither expensive nor slow, and wrapping it in checkpoint-and-resume
machinery would buy nothing and cost a resume path nobody would exercise.

Declining is not a failure. The node returns unchanged records, the emitter writes them
without a `verified` block, and the bundle sits at OKF's `unverified` tier — which is
what it is. The only thing a decline loses is a claim we were never entitled to make.
"""

from __future__ import annotations

from typing import Any

from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.schemas import ConceptRecord, Verification

__all__ = ["review_node"]

NODE = "review"


async def review_node(state: RunState, deps: Deps) -> dict[str, Any]:
    reviewer = deps.reviewer
    records: list[ConceptRecord] = list(state.get("records") or [])
    if reviewer is None:
        # No `--review`. Not a skipped step and not a decline — nothing was asked.
        return {}

    warnings = list(state.get("warnings") or [])
    with span(deps, NODE) as report:
        signoff = await reviewer.sign_off(
            records,
            topics=dict(state.get("topics") or {}),
            verification=state.get("verification"),
            warnings=warnings,
        )
        if not signoff.approved or signoff.at is None:
            note = (
                "sign-off was not given, so the bundle is emitted with no `verified` "
                "block — the OKF `unverified` tier" + (f" ({signoff.note})" if signoff.note else "")
            )
            warnings.append(note)
            deps.warn(NODE, note)
            report["summary"] = "not signed off; emitting unverified"
            return {"warnings": warnings}

        attestation = Verification(by=signoff.by, at=signoff.at)
        stamped = [record.model_copy(update={"verified": [attestation]}) for record in records]
        report["summary"] = f"{len(stamped)} document(s) signed off by {signoff.by}"

    deps.progress(NODE, f"signed off by {signoff.by}")
    return {"records": stamped, "verified_by": signoff.by}
