"""What recurs across the corpus, and where each occurrence can be read.

Every object here is a pointer or a bag of pointers. A `PredictorSite` is an address —
a document and the `#` value of one row inside it — and nothing in this module holds a
finding that is not also written in a paper's own file. That is deliberate and it is the
whole design: an index that summarizes the corpus well enough to replace it is an index
that quietly becomes the corpus, and then the evidence, the quotes and the provenance
that justify the bundle stop being read at all.

So the counts here are navigational. `papers` says how many documents to open, not how
established a relationship is, because a curated corpus counts its own curation — MMR
diversification and the charter's per-topic floors decide how many times a predictor can
appear, long before the literature gets a say.

Grouped by predictor **and outcome together**, never by predictor alone. One paper can
report the same exposure against six different outcomes in six directions; collapsed on
the predictor that reads as a contested finding, and split by outcome it reads as what it
is — six coherent ones.
"""

from __future__ import annotations

from collections import Counter

from pydantic import Field

from okf_loremaster.schemas.common import Direction, Model
from okf_loremaster.schemas.strength import RowStrength

__all__ = ["OutcomeGroup", "PredictorGroup", "PredictorSite", "RecurrenceIndex"]


class PredictorSite(Model):
    """One row of one paper's predictor table, as an address into the bundle."""

    pmid: str
    # The topic folder the paper sits in, which is also its `domain` frontmatter key.
    domain: str = ""
    # `<domain>/<pmid>_<Author>.md`, relative to the corpus root — where `predictors.md`
    # also sits, so the string is a working relative link with nothing prepended.
    file: str = ""
    # 1-based, matching the `#` column of `# Predictors reported`. The second half of the
    # address: without it a reader lands on a paper with a dozen rows and no idea which.
    row: int = 1

    # The paper's own words, kept verbatim. The clustering that put this row in a group
    # normalized them; showing the normalized form instead would hide what was merged.
    predictor: str = ""
    outcome: str = ""
    operationalization: str = ""

    direction: Direction = Direction.UNCLEAR
    # Rendered by the emitter from the same rule the document table uses, so a magnitude
    # that verification removed cannot reappear here.
    effect: str = ""
    strength: RowStrength | None = None
    year: int | None = None


class OutcomeGroup(Model):
    """Every row that relates one predictor to one outcome."""

    outcome: str
    # Each distinct way the papers wrote this outcome, most common first. The audit trail
    # for the clustering: a merge nobody can see is a merge nobody can dispute.
    surface_forms: list[str] = Field(default_factory=list)
    sites: list[PredictorSite] = Field(default_factory=list)

    @property
    def papers(self) -> int:
        return len({site.pmid for site in self.sites})

    @property
    def directions(self) -> list[tuple[Direction, int]]:
        """How the rows fall by sign, commonest first, ties in enum order.

        Reported instead of a bare row count because the count answers a question nobody
        asked. "Three papers" is compatible with three agreements and with two-against-one,
        and the difference is the only thing worth knowing before opening them.
        """
        counts = Counter(site.direction for site in self.sites)
        order = list(Direction)
        return sorted(counts.items(), key=lambda item: (-item[1], order.index(item[0])))

    @property
    def contested(self) -> bool:
        """Whether the rows disagree about the sign.

        Only a genuine opposition counts: an increase beside a decrease. A null result
        beside an effect is a difference in power or population at least as often as a
        disagreement, and flagging it would put a warning on most of the corpus.
        """
        signs = {site.direction for site in self.sites}
        return Direction.INCREASES in signs and Direction.DECREASES in signs


class PredictorGroup(Model):
    """One predictor, and every outcome the corpus relates it to."""

    predictor: str
    surface_forms: list[str] = Field(default_factory=list)
    outcomes: list[OutcomeGroup] = Field(default_factory=list)

    @property
    def sites(self) -> list[PredictorSite]:
        return [site for outcome in self.outcomes for site in outcome.sites]

    @property
    def rows(self) -> int:
        return len(self.sites)

    @property
    def papers(self) -> int:
        return len({site.pmid for site in self.sites})

    @property
    def topics(self) -> list[str]:
        """The topic folders this predictor turns up in, in corpus order.

        A predictor that recurs inside one topic is a well-studied relationship; one that
        recurs across three is a construct the taxonomy cuts through, and those are the
        ones worth an engineer's attention.
        """
        seen: dict[str, None] = {}
        for site in self.sites:
            if site.domain:
                seen.setdefault(site.domain, None)
        return list(seen)

    @property
    def contested(self) -> bool:
        return any(outcome.contested for outcome in self.outcomes)


class RecurrenceIndex(Model):
    """The whole predictor index: what recurred, and how much did not."""

    groups: list[PredictorGroup] = Field(default_factory=list)
    # Predictors reported by exactly one paper. A number rather than a list: they did not
    # recur, which is what this file is about, and they are already in the topic indexes.
    # Counted so that "what this file leaves out" is stated rather than merely true.
    once: int = 0
    rows: int = 0
    papers: int = 0

    @property
    def predictors(self) -> int:
        return len(self.groups)
