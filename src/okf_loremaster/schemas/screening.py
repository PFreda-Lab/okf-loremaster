"""Screening and curation verdicts.

Two model calls with very different jobs. Screening is FAST, one paper at a time,
answering "is this about the charter's question at all" — thousands of cheap judgments,
so the schema is small and every field earns its tokens. Curation is balanced-tier and sees a
whole topic at once, answering "which of these belong here, and is this topic full,
thin, or empty" — the one place a run can notice it searched badly.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from okf_loremaster.schemas.common import Confidence, Model, prose

__all__ = [
    "CurationDecision",
    "CurationResult",
    "ScreenVerdict",
    "TopicCuration",
    "TopicGap",
]


class ScreenVerdict(Model):
    """One FAST judgment on one title-and-abstract.

    `relevance` exists alongside `include` because the include/exclude line has to move
    at curation time. If a topic comes up short, the run needs a ranked queue of the
    papers it nearly kept rather than a re-screen; if the pool overflows, it needs a
    principled way to cut. A bare boolean supports neither.
    """

    pmid: str
    include: bool
    # 0 unrelated · 1 tangential · 2 relevant · 3 directly on point.
    relevance: int = Field(default=0, ge=0, le=3)
    # The topic the screener thinks this belongs on. Advisory: curation decides, and
    # the screener has seen the taxonomy but not the topic's other papers.
    topic: str = ""
    # One clause. Long reasons at FAST volume cost more than they inform. Trimmed
    # rather than rejected: a screening call carries a whole batch, so one verdict
    # that ran a clause long would fail the parse for every paper beside it.
    reason: Annotated[str, prose(240)] = ""
    confidence: Confidence = Confidence.MEDIUM

    @property
    def borderline(self) -> bool:
        """Excluded but arguable, or included on weak grounds.

        These are the papers curation reconsiders first when a topic is under its
        floor, which is cheaper and better-informed than another search round.
        """
        return (not self.include and self.relevance >= 2) or (
            self.include and self.confidence is Confidence.LOW
        )


class CurationDecision(Model):
    """Where one paper lands, decided with the whole topic in view."""

    pmid: str
    keep: bool
    # The topic that considered this paper, written in by the curate node rather than
    # by the model — set on rejections too, because that is what tells the floor
    # backfill which topic a rejected paper is the nearest miss for.
    topic: str = ""
    rationale: Annotated[str, prose(240)] = ""


class TopicCuration(Model):
    """One balanced-tier call's answer about one topic.

    Scoped to a single topic because the call is: asking the model to name the topic on
    every decision would spend output tokens echoing a value we already hold and could
    receive wrong. The node writes `CurationDecision.topic` in afterward.
    """

    decisions: list[CurationDecision] = Field(default_factory=list)
    # What the topic still lacks, in the curator's own words. This is the seed for the
    # conditional re-query edge, which is why it is prompted for as search concepts
    # rather than as a complaint.
    missing: Annotated[str, prose(300)] = ""


class TopicGap(Model):
    """A topic that curation could not fill, and what it would take to fill it."""

    topic: str
    kept: int
    floor: int
    # What the topic is missing, in the curator's words — this becomes the seed for the
    # conditional re-query edge rather than simply re-running the same searches.
    missing: str = ""

    @property
    def shortfall(self) -> int:
        return max(0, self.floor - self.kept)


class CurationResult(Model):
    decisions: list[CurationDecision] = Field(default_factory=list)
    gaps: list[TopicGap] = Field(default_factory=list)

    @property
    def kept(self) -> list[CurationDecision]:
        return [d for d in self.decisions if d.keep]

    def by_topic(self) -> dict[str, list[str]]:
        """PMIDs per topic, kept only, in decision order."""
        out: dict[str, list[str]] = {}
        for decision in self.kept:
            out.setdefault(decision.topic, []).append(decision.pmid)
        return out

    @property
    def needs_more_search(self) -> bool:
        return any(gap.shortfall > 0 for gap in self.gaps)
