"""Screening and curation verdicts.

Two model calls with very different jobs. Screening is FAST, one paper at a time,
answering "is this about the charter's question at all" — thousands of cheap judgments,
so the schema is small and every field earns its tokens. Curation is MID and sees a
whole shelf at once, answering "which of these belong here, and is this shelf full,
thin, or empty" — the one place a run can notice it searched badly.
"""

from __future__ import annotations

from pydantic import Field

from okf_loremaster.schemas.common import Confidence, Model

__all__ = [
    "CurationDecision",
    "CurationResult",
    "ScreenVerdict",
    "ShelfCuration",
    "ShelfGap",
]


class ScreenVerdict(Model):
    """One FAST judgment on one title-and-abstract.

    `relevance` exists alongside `include` because the include/exclude line has to move
    at curation time. If a shelf comes up short, the run needs a ranked queue of the
    papers it nearly kept rather than a re-screen; if the pool overflows, it needs a
    principled way to cut. A bare boolean supports neither.
    """

    pmid: str
    include: bool
    # 0 unrelated · 1 tangential · 2 relevant · 3 directly on point.
    relevance: int = Field(default=0, ge=0, le=3)
    # The shelf the screener thinks this belongs on. Advisory: curation decides, and
    # the screener has seen the taxonomy but not the shelf's other papers.
    shelf: str = ""
    # One clause. Long reasons at FAST volume cost more than they inform.
    reason: str = Field(default="", max_length=240)
    confidence: Confidence = Confidence.MEDIUM

    @property
    def borderline(self) -> bool:
        """Excluded but arguable, or included on weak grounds.

        These are the papers curation reconsiders first when a shelf is under its
        floor, which is cheaper and better-informed than another search round.
        """
        return (not self.include and self.relevance >= 2) or (
            self.include and self.confidence is Confidence.LOW
        )


class CurationDecision(Model):
    """Where one paper lands, decided with the whole shelf in view."""

    pmid: str
    keep: bool
    # The shelf that considered this paper, written in by the curate node rather than
    # by the model — set on rejections too, because that is what tells the floor
    # backfill which shelf a rejected paper is the nearest miss for.
    shelf: str = ""
    rationale: str = Field(default="", max_length=240)


class ShelfCuration(Model):
    """One MID call's answer about one shelf.

    Scoped to a single shelf because the call is: asking the model to name the shelf on
    every decision would spend output tokens echoing a value we already hold and could
    receive wrong. The node writes `CurationDecision.shelf` in afterward.
    """

    decisions: list[CurationDecision] = Field(default_factory=list)
    # What the shelf still lacks, in the curator's own words. This is the seed for the
    # conditional re-query edge, which is why it is prompted for as search concepts
    # rather than as a complaint.
    missing: str = Field(default="", max_length=300)


class ShelfGap(Model):
    """A shelf that curation could not fill, and what it would take to fill it."""

    shelf: str
    kept: int
    floor: int
    # What the shelf is missing, in the curator's words — this becomes the seed for the
    # conditional re-query edge rather than simply re-running the same searches.
    missing: str = ""

    @property
    def shortfall(self) -> int:
        return max(0, self.floor - self.kept)


class CurationResult(Model):
    decisions: list[CurationDecision] = Field(default_factory=list)
    gaps: list[ShelfGap] = Field(default_factory=list)

    @property
    def kept(self) -> list[CurationDecision]:
        return [d for d in self.decisions if d.keep]

    def by_shelf(self) -> dict[str, list[str]]:
        """PMIDs per shelf, kept only, in decision order."""
        out: dict[str, list[str]] = {}
        for decision in self.kept:
            out.setdefault(decision.shelf, []).append(decision.pmid)
        return out

    @property
    def needs_more_search(self) -> bool:
        return any(gap.shortfall > 0 for gap in self.gaps)
