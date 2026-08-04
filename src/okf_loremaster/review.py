"""Human sign-off, and what a signature is allowed to mean.

OKF derives its trust tiers from `verified`. A document with no `verified` block is
`unverified`, which is the honest tier for a machine extraction — so the only thing that
may write one is a person who looked. That makes this a small module with one rule in
it, stated three ways:

**A declined sign-off still emits the bundle.** Nothing is discarded; the files are
simply written at the tier they have actually earned. Refusing to emit would punish the
reviewer for reading carefully.

**`--review` cannot be combined with `--dry-run` or `--json`.** A dry run writes no
bundle and a machine-readable stream has nobody to ask; signing under either would write
`by: "human:<id>"` naming a person who never saw the bundle. That is not a weaker
attestation, it is a false one, and the CLI refuses the combination rather than
degrading it. It combines freely with an autonomous run: reading a finished bundle and
steering the search that produced it are separate decisions.

**The signer is named, not assumed.** `OKF_LOREMASTER_REVIEWER_ID` when set, the OS
login otherwise. A signature that says only "a human" identifies nobody.

`Reviewer` is a protocol so the graph can be driven by a console, by the TUI, by a test,
or by nothing at all without knowing which — the same shape as `ui.pauses.Pause`, and
for the same reason: this is a decision surface, and a node must not contain one. It is
async for the same reason too: the TUI answers by awaiting a modal screen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from okf_loremaster.config import Settings
from okf_loremaster.schemas import ConceptRecord, VerificationSummary

__all__ = ["NoReview", "Reviewer", "Signoff", "signer_id"]

# The `by` prefix OKF uses for a person, as opposed to a process. `generated.by` names a
# model; `verified.by` has to name someone who can be asked about it.
HUMAN_PREFIX = "human:"


def signer_id(settings: Settings) -> str:
    """Who a sign-off is attributed to, as `human:<id>`.

    Reads config first so a shared or headless machine can say who is actually at the
    keyboard; falls back to the OS login. `getpass` rather than `os.environ` because
    config is the only module allowed to read the environment.
    """
    import getpass

    name = settings.reviewer_id.strip()
    if not name:
        try:
            name = getpass.getuser()
        except Exception:  # pragma: no cover - no login name on this platform
            name = "unknown"
    return f"{HUMAN_PREFIX}{name}"


@dataclass(frozen=True, slots=True)
class Signoff:
    """The outcome of one review."""

    approved: bool
    by: str = ""
    at: datetime | None = None
    note: str = ""

    @classmethod
    def granted(cls, by: str, *, note: str = "") -> Signoff:
        return cls(approved=True, by=by, at=datetime.now(UTC), note=note)

    @classmethod
    def declined(cls, note: str = "") -> Signoff:
        return cls(approved=False, note=note)


class Reviewer(Protocol):
    """What the review node needs from a sign-off surface."""

    async def sign_off(
        self,
        records: Sequence[ConceptRecord],
        *,
        topics: dict[str, list[str]],
        verification: VerificationSummary | None,
        warnings: Sequence[str],
    ) -> Signoff: ...


class NoReview:
    """Declines without asking or printing. The default, and what tests use.

    Declining rather than approving is the point: no one looked, so there is nothing to
    attest to, and the bundle goes out at the tier it earned.
    """

    async def sign_off(
        self,
        records: Sequence[ConceptRecord],
        *,
        topics: dict[str, list[str]],
        verification: VerificationSummary | None,
        warnings: Sequence[str],
    ) -> Signoff:
        return Signoff.declined("no reviewer was attached")
