"""Scoring, diversification, and the per-topic quota. All code, no judgment.

Ranking decides which candidates reach the screener, and screening is the single
largest cost in a run. It is deterministic on purpose: the same corpus and the same
charter must produce the same pool, or nothing downstream is reproducible.

Two things happen beyond sorting by relevance, and both fix the same failure. Pure
relevance rank returns the *same paper* several times over — a well-covered topic
produces a cluster of near-identical reviews — and it lets one prolific topic crowd out
the rest, because a topic whose queries match ten thousand papers fills the pool before
a topic whose queries match two hundred gets a look in.

- **MMR** trades a little relevance for coverage, so a cluster contributes its best
  member rather than its first six.
- **The per-topic quota** reserves capacity for every topic before the pool is filled.

No paper has been assigned a topic at this point — screening does that. What is
available is which *query* found it, and each planned query names the topic it was
written for. That is the affinity used here, and it is enough: the point is to stop one
topic's searches monopolizing the pool, not to pre-empt the screener's judgment.

`selection_diff` exists so the effect is visible rather than asserted. `--dry-run`
prints it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import Field

from okf_loremaster.schemas import Candidate, Model, ScoredCandidate

__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_WEIGHTS",
    "UNASSIGNED",
    "SelectionComparison",
    "Weights",
    "mmr_order",
    "quota_select",
    "relevance",
    "score_all",
    "selection_diff",
    "similarity",
    "text_tokens",
    "tokens",
    "topic_affinity",
]

# Relevance versus diversity. 0.7 keeps rank dominant while still breaking up clusters;
# at 1.0 MMR degenerates to the pure ordering and the whole pass becomes a no-op.
DEFAULT_LAMBDA = 0.7

# Publication types that are *about* the literature rather than in it. Not a judgment
# about any specialty or condition — these are PubMed's own structural categories.
_NON_RESEARCH_TYPES = frozenset(
    {
        "Comment",
        "Editorial",
        "Letter",
        "News",
        "Newspaper Article",
        "Published Erratum",
        "Biography",
        "Historical Article",
        "Patient Education Handout",
    }
)

# Citation counts are meaningless for a paper this new: it is not uncited, it is
# unread. Such a paper gets the neutral score rather than the floor.
_TOO_NEW_YEARS = 2
# Citation count at which the log-scaled component saturates.
_CITATION_CEILING = 200.0
# RCR is field- and time-normalized with 1.0 the NIH-funded average, so 3.0 is already
# well into the top of any field.
_RCR_CEILING = 3.0

_TOKEN = re.compile(r"[a-z0-9]+")
_TOKEN_STOPWORDS = frozenset(
    """
    a an and are as at be by for from in into is it its of on or the this to with
    """.split()  # noqa: SIM905  — a wrapped word block reads better than a quoted list
)


@dataclass(frozen=True, slots=True)
class Weights:
    """Relative pull of each ranking signal. Sums to 1.0.

    Constants rather than configuration: a knob per signal invites tuning the ranking
    against one project's corpus, which is exactly what would stop it generalizing.
    """

    agreement: float = 0.25
    position: float = 0.30
    recency: float = 0.15
    citation: float = 0.15
    abstract: float = 0.10
    article: float = 0.05


DEFAULT_WEIGHTS = Weights()


# --- relevance --------------------------------------------------------------


def _agreement(candidate: Candidate) -> float:
    """How many independent queries found it. Convergence is evidence."""
    return min(1.0, max(0, len(candidate.found_by) - 1) / 2.0)


def _position(candidate: Candidate) -> float:
    """Best position reached in any query, decayed rather than cut off."""
    return 1.0 / (1.0 + candidate.best_rank / 10.0)


def _recency(candidate: Candidate, *, now_year: int, min_year: int | None) -> float:
    if candidate.year is None:
        # Unknown is not old. A missing date is a metadata gap, not a signal.
        return 0.5
    floor = min_year if min_year is not None else now_year - 20
    if now_year <= floor:
        return 1.0
    return min(1.0, max(0.0, (candidate.year - floor) / (now_year - floor)))


def _citation(candidate: Candidate, *, now_year: int) -> float:
    if candidate.rcr is not None:
        return min(1.0, max(0.0, candidate.rcr / _RCR_CEILING))
    if candidate.year is not None and now_year - candidate.year <= _TOO_NEW_YEARS:
        return 0.5
    return min(1.0, math.log1p(candidate.citation_count) / math.log1p(_CITATION_CEILING))


def _article(candidate: Candidate) -> float:
    """Whether this is primary literature at all."""
    if _NON_RESEARCH_TYPES & set(candidate.publication_types):
        return 0.0
    if candidate.source_type != "journal":
        return 0.3
    return 1.0 if candidate.is_research_article else 0.7


def relevance(
    candidate: Candidate,
    *,
    now_year: int,
    min_year: int | None = None,
    weights: Weights = DEFAULT_WEIGHTS,
) -> tuple[float, dict[str, float]]:
    """Score in [0, 1], plus the named contributions that sum to it.

    The components come back with the total so a ranking can be explained. An
    unexplainable ordering is one nobody can debug when a paper that obviously belongs
    fails to make the pool.
    """
    parts = {
        "agreement": weights.agreement * _agreement(candidate),
        "position": weights.position * _position(candidate),
        "recency": weights.recency * _recency(candidate, now_year=now_year, min_year=min_year),
        "citation": weights.citation * _citation(candidate, now_year=now_year),
        "abstract": weights.abstract * (1.0 if candidate.has_abstract else 0.0),
        "article": weights.article * _article(candidate),
    }
    return sum(parts.values()), parts


def score_all(
    candidates: Sequence[Candidate],
    *,
    now_year: int,
    min_year: int | None = None,
    weights: Weights = DEFAULT_WEIGHTS,
) -> list[ScoredCandidate]:
    """Score and sort, highest first. Ties break on PMID, so the order is total."""
    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        score, parts = relevance(candidate, now_year=now_year, min_year=min_year, weights=weights)
        scored.append(ScoredCandidate(candidate=candidate, score=score, components=parts))

    scored.sort(key=lambda s: (-s.score, s.pmid))
    for index, item in enumerate(scored):
        item.position = index
    return scored


# --- diversification --------------------------------------------------------


def text_tokens(text: str) -> frozenset[str]:
    """Lowercased content words of any string.

    Shared so that everything comparing two pieces of text in this package tokenizes
    them identically — a candidate against a candidate here, a candidate against a
    topic's own description in the curate node.
    """
    return frozenset(
        t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _TOKEN_STOPWORDS
    )


def tokens(candidate: Candidate) -> frozenset[str]:
    """The bag of words MMR compares two candidates on.

    Title plus MeSH plus keywords, and deliberately not the abstract: two papers on the
    same finding share a topic vocabulary throughout their abstracts whatever their
    differences, so including it pushes every pairwise similarity toward the same
    middling number and MMR stops discriminating.
    """
    return text_tokens(" ".join([candidate.title, *candidate.mesh_terms, *candidate.keywords]))


def similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap. Lexical on purpose — embeddings are a step 8 dependency.

    Using them here would mean loading a transformer to rank candidates that may never
    be screened, and paying that cost on every `--dry-run`.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def mmr_order(
    scored: Sequence[ScoredCandidate],
    *,
    limit: int,
    lambda_: float = DEFAULT_LAMBDA,
) -> list[ScoredCandidate]:
    """Maximal marginal relevance: take the best paper that is not already covered.

    Each candidate carries its running maximum similarity to the selected set, updated
    with each pick. That is what keeps the pass linear in selections rather than
    quadratic in them.
    """
    pool = list(scored)
    if limit <= 0 or not pool:
        return []

    vectors = {item.pmid: tokens(item.candidate) for item in pool}
    closest: dict[str, float] = dict.fromkeys(vectors, 0.0)

    def marginal(item: ScoredCandidate) -> tuple[float, int, str]:
        value = lambda_ * item.score - (1.0 - lambda_) * closest[item.pmid]
        # Ties break on the pure ordering, then on PMID, so the result never depends
        # on the order candidates happened to arrive in.
        return value, -item.position, item.pmid

    selected: list[ScoredCandidate] = []
    remaining = pool
    while remaining and len(selected) < limit:
        best = max(remaining, key=marginal)
        selected.append(best)
        remaining = [item for item in remaining if item.pmid != best.pmid]
        chosen = vectors[best.pmid]
        for item in remaining:
            overlap = similarity(vectors[item.pmid], chosen)
            if overlap > closest[item.pmid]:
                closest[item.pmid] = overlap
    return selected


# --- per-topic quota --------------------------------------------------------


def topic_affinity(candidate: Candidate, query_topic: Mapping[str, str]) -> str:
    """Which topic's queries found this paper. `""` when none of them targeted a topic.

    A paper found by several topics' queries is attributed to the first in the
    charter's own order, so the mapping is total and stable rather than dependent on
    which search happened to finish first.
    """
    found = {query_topic.get(term, "") for term in candidate.found_by}
    for topic in query_topic.values():
        if topic and topic in found:
            return topic
    return ""


def quota_select(
    scored: Sequence[ScoredCandidate],
    *,
    query_topic: Mapping[str, str],
    pool_size: int,
    lambda_: float = DEFAULT_LAMBDA,
) -> list[ScoredCandidate]:
    """Fill the pool topic by topic, then top it up from whatever is left over.

    Unused quota is not wasted: a topic whose queries returned twenty papers releases
    the rest of its share back to the pool. The reservation protects a thin topic from
    being crowded out; it does not hold capacity empty on its behalf.
    """
    if pool_size <= 0 or not scored:
        return []

    groups: dict[str, list[ScoredCandidate]] = {}
    for item in scored:
        groups.setdefault(topic_affinity(item.candidate, query_topic), []).append(item)

    quota = max(1, math.ceil(pool_size / len(groups)))
    taken: list[ScoredCandidate] = []
    for _, members in sorted(groups.items()):
        taken.extend(mmr_order(members, limit=quota, lambda_=lambda_))

    if len(taken) < pool_size:
        chosen = {item.pmid for item in taken}
        leftovers = [item for item in scored if item.pmid not in chosen]
        taken.extend(mmr_order(leftovers, limit=pool_size - len(taken), lambda_=lambda_))

    taken.sort(key=lambda s: (-s.score, s.pmid))
    return taken[:pool_size]


# --- making the effect visible ----------------------------------------------

# What a paper with no topic-targeted query behind it is called in the printed table.
UNASSIGNED = "(unassigned)"


class SelectionComparison(Model):
    """Pure relevance rank against MMR plus quota, as counts a person can read.

    Carried in run state and printed by `--dry-run`. Without it, "diversification is
    on" is a claim about the code rather than an observation about this corpus.
    """

    pure: list[str] = Field(default_factory=list)
    diversified: list[str] = Field(default_factory=list)
    pure_by_topic: dict[str, int] = Field(default_factory=dict)
    diversified_by_topic: dict[str, int] = Field(default_factory=dict)

    @property
    def added(self) -> list[str]:
        """In the diversified pool, absent from the pure one."""
        pure = set(self.pure)
        return [pmid for pmid in self.diversified if pmid not in pure]

    @property
    def dropped(self) -> list[str]:
        diversified = set(self.diversified)
        return [pmid for pmid in self.pure if pmid not in diversified]

    @property
    def changed(self) -> int:
        return len(self.added)

    @property
    def topics_helped(self) -> list[str]:
        """Topics holding more of the pool than pure relevance would have left them."""
        return sorted(
            topic
            for topic, count in self.diversified_by_topic.items()
            if count > self.pure_by_topic.get(topic, 0)
        )

    def summary(self) -> str:
        if not self.pure:
            return "nothing to rank"
        if self.changed == 0:
            return (
                f"diversification changed nothing: {len(self.pure)} candidate(s) is at or "
                "below the pool size, so every one was retained either way"
            )
        share = self.changed / len(self.diversified) * 100 if self.diversified else 0.0
        helped = ", ".join(self.topics_helped) or "none"
        return (
            f"{self.changed} of {len(self.diversified)} ({share:.0f}%) papers are in the "
            f"pool only because of MMR and the per-topic quota; topics gaining share: {helped}"
        )


def selection_diff(
    pure: Sequence[ScoredCandidate],
    diversified: Sequence[ScoredCandidate],
    *,
    query_topic: Mapping[str, str],
) -> SelectionComparison:
    """Compare the two selections over the same scored pool."""

    def by_topic(items: Sequence[ScoredCandidate]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            topic = topic_affinity(item.candidate, query_topic) or UNASSIGNED
            counts[topic] = counts.get(topic, 0) + 1
        return dict(sorted(counts.items()))

    return SelectionComparison(
        pure=[item.pmid for item in pure],
        diversified=[item.pmid for item in diversified],
        pure_by_topic=by_topic(pure),
        diversified_by_topic=by_topic(diversified),
    )
