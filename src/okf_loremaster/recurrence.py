"""What turns up more than once, and where. All code, no judgment, no model call.

A finished bundle answers "what does this paper say" well and "which papers say the same
thing" not at all — the answer is spread over a hundred files and nobody reads a hundred
files to find it. This module gathers every predictor row in the corpus, clusters the ones
that are the same relationship written differently, and hands back an index of addresses.

Three decisions are load-bearing, and each of them is a thing this deliberately does not
do:

**It does not rank.** No score combining strength with frequency, because the two answer
different questions — "how good is this study" and "how much has this been studied" — and
multiplying them produces a number that answers neither. Worse, frequency in a curated
corpus is a measurement of the curation: MMR diversification and the charter's per-topic
floors decide how often a predictor may appear before the literature is consulted.

**It does not summarize.** Every row it emits is a pointer at a document and a row number
inside it. An index that can be read instead of the corpus is one that will be, and then
the quotes, the operationalizations and the provenance that justify the whole bundle stop
being opened.

**It does not group by predictor alone.** A predictor is grouped with its outcome, because
one paper reporting one exposure against six outcomes in six directions is six coherent
findings, and collapsing them onto the exposure would print it as a contradiction.

Clustering is lexical and deliberately timid. Merging two forms that are not the same
relationship invents an agreement or a disagreement no paper claimed, and it does so
invisibly; failing to merge two that are leaves two adjacent entries a reader can see and
join for themselves. So the merge rule is exact normalized match, then one conservative
pass over phrases that differ only by a qualifier — and never over a qualifier that flips
the meaning, which is what `_POLARITY` is for. Every merge prints the surface forms it
absorbed, so any of them can be disputed.

Embeddings would cluster better and are not used: `--finalize okf` never builds them, so
the common path would silently get the worse clustering, and reaching for them here would
drag the `[vectors]` extra into a file every run writes.

Nothing here names a condition, a specialty or a cohort. `_POLARITY` is a property of
English rather than of a literature — `high`, `former` and `reduced` flip a meaning in
any field, which is the test `CLAUDE.md` sets for a constant that may live in `src/`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from okf_loremaster.ranking import text_tokens
from okf_loremaster.schemas import ConceptRecord, PredictorRow
from okf_loremaster.schemas.recurrence import (
    OutcomeGroup,
    PredictorGroup,
    PredictorSite,
    RecurrenceIndex,
)

__all__ = ["MIN_PAPERS", "index_predictors", "surface_key"]

# How many distinct papers a predictor needs before it gets an entry. Two, because that is
# the smallest number that means "again" — the file is about recurrence, and one paper
# reporting something is already fully described by that paper's own document.
MIN_PAPERS = 2

# Parenthetical spans, which in this literature are almost always an abbreviation the
# paper defined for itself. "Chronic sleep restriction (CSR)" and "chronic sleep
# restriction" are one phrase, and keeping the initialism makes them two.
_PARENTHETICAL = re.compile(r"\([^)]*\)")

# Words whose presence changes what a phrase claims rather than merely narrowing it. A
# qualifier outside this set — "chronic", "maternal", "weekly" — describes the same
# relationship more precisely, so a phrase carrying it may be absorbed into the phrase
# without it. One of these, and the two phrases are opposite ends of the same axis, and
# merging them would print a U-shaped relationship as a contradiction.
#
# English, not medicine: these flip a meaning in any literature, which is why they can be
# a constant here when a disease name cannot.
_POLARITY = frozenset(
    """
    abnormal absent adequate better best current decreased decreasing deficient
    early earlier elevated excess excessive fast female few fewer former frequent
    good greater heavy high higher highest impaired inadequate increased increasing
    infrequent insufficient irregular large late later less lesser light long longer
    low lower lowest male mild moderate more never new normal old older poor poorer
    positive negative present previous prior rapid raised reduced regular severe short
    shorter slow small sufficient worse worst young younger
    """.split()  # noqa: SIM905  — a wrapped word block reads better than a quoted list
)

# Plural forms these endings produce are not plurals. Stripping the `s` from `status`,
# `analysis` or `stress` would make three words that no longer match themselves.
_NOT_A_PLURAL = ("ss", "us", "is", "os", "as")


def surface_key(text: str) -> frozenset[str]:
    """The token set two phrases are compared on.

    Parentheses dropped, tokenized the way everything else in this package tokenizes
    (`ranking.text_tokens`, so stopwords and one- and two-character fragments go), then
    singularized. A set rather than a sequence: "duration of sleep" and "sleep duration"
    are the same phrase, and word order is not evidence that they are not.
    """
    stripped = _PARENTHETICAL.sub(" ", text)
    return frozenset(_singular(token) for token in text_tokens(stripped))


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(_NOT_A_PLURAL):
        return token[:-1]
    return token


def _absorbs(host: frozenset[str], guest: frozenset[str]) -> bool:
    """Whether `guest` is `host` plus qualifiers that do not change the claim.

    Containment rather than overlap, because that is the shape the near-duplicates
    actually take: a paper writes the phrase, and the next paper writes the phrase with a
    word in front of it. Overlap would also merge two phrases that share most of their
    words while differing on the one that matters.
    """
    if not host or not guest or host == guest:
        return False
    if not (host < guest or guest < host):
        return False
    return not (host ^ guest) & _POLARITY


@dataclass(slots=True)
class _Cluster:
    """One accumulating group of surface forms that mean the same thing."""

    key: frozenset[str]
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def label(self) -> str:
        """The commonest surface form, then the shortest, then alphabetical.

        Commonest because it is how the corpus mostly says it; the rest of the ordering
        exists so that two runs over the same records agree on the heading.
        """
        return min(self.counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[0]

    @property
    def forms(self) -> list[str]:
        return [
            form
            for form, _ in sorted(
                self.counts.items(), key=lambda item: (-item[1], len(item[0]), item[0])
            )
        ]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _cluster(phrases: Iterable[str]) -> dict[str, _Cluster]:
    """Map every surface form to the cluster it belongs to.

    Two passes. The first is exact match on the normalized key and is where nearly every
    merge happens. The second walks the clusters commonest-first and lets an established
    one absorb a later, rarer one — established meaning already accepted in this pass, so
    absorption is always tested against a representative that the corpus actually wrote,
    never against a token set that grew by accretion. Without that, `sleep duration`
    absorbing `sleep duration in hours` would let the result reach on to absorb something
    the original phrase shares nothing with.
    """
    seeds: dict[frozenset[str], _Cluster] = {}
    for phrase in phrases:
        text = " ".join(phrase.split())
        if not text:
            continue
        key = surface_key(text)
        # A phrase whose tokens all normalize away — an initialism, a bare number — falls
        # back to its own text, so it still clusters with itself and with nothing else.
        cluster = seeds.setdefault(key or frozenset({text.lower()}), _Cluster(key=key))
        cluster.counts[text] += 1

    ordered = sorted(seeds.values(), key=lambda c: (-c.total, c.label))
    accepted: list[_Cluster] = []
    for cluster in ordered:
        host = next((c for c in accepted if _absorbs(c.key, cluster.key)), None)
        if host is None:
            accepted.append(cluster)
            continue
        host.counts.update(cluster.counts)

    return {form: cluster for cluster in accepted for form in cluster.counts}


def _site(record: ConceptRecord, row: PredictorRow, number: int, effect: str) -> PredictorSite:
    strengths = record.strength.rows if record.strength is not None else []
    return PredictorSite(
        pmid=record.pmid,
        domain=record.domain,
        file=f"{record.domain}/{record.filename}",
        row=number,
        predictor=" ".join(row.predictor.split()),
        outcome=" ".join(row.outcome.split()),
        operationalization=row.operationalization,
        direction=row.direction,
        effect=effect,
        strength=strengths[number - 1] if number <= len(strengths) else None,
        year=record.year,
    )


def index_predictors(
    records: Sequence[ConceptRecord], *, effect_of: Callable[[PredictorRow], str]
) -> RecurrenceIndex:
    """Every predictor row in the corpus, clustered into what recurs.

    `effect_of` is passed in rather than imported so this module does not depend on the
    emitter that will render it: the rule for what a magnitude may print as belongs to the
    writer, and it is not a rule this file should be able to get differently.
    """
    sites: list[PredictorSite] = []
    for record in records:
        for number, row in enumerate(record.extraction.predictors, start=1):
            sites.append(_site(record, row, number, effect_of(row)))

    by_predictor = _cluster(site.predictor for site in sites)
    grouped: dict[str, list[PredictorSite]] = {}
    for site in sites:
        grouped.setdefault(by_predictor[site.predictor].label, []).append(site)

    groups: list[PredictorGroup] = []
    once = 0
    for label, members in grouped.items():
        if len({site.pmid for site in members}) < MIN_PAPERS:
            once += 1
            continue
        groups.append(
            PredictorGroup(
                predictor=label,
                surface_forms=by_predictor[members[0].predictor].forms,
                outcomes=_outcomes(members),
            )
        )

    # Papers first, then rows, then the label, so the ordering is total and a rerun over
    # the same records writes the same file. Papers rather than rows leads because two
    # papers agreeing is a different kind of evidence from one paper repeating itself.
    groups.sort(key=lambda g: (-g.papers, -g.rows, g.predictor))
    return RecurrenceIndex(
        groups=groups,
        once=once,
        rows=len(sites),
        papers=len({site.pmid for site in sites}),
    )


def _outcomes(sites: Sequence[PredictorSite]) -> list[OutcomeGroup]:
    """Split one predictor's rows by the outcome each was measured against.

    Clustered within the predictor rather than across the corpus: the same word means
    different things next to different exposures, and a corpus-wide outcome vocabulary
    would merge on the word instead of on the relationship.
    """
    # An unnamed outcome is its own bucket rather than being folded in with the named
    # ones. "This predictor, against something the extraction did not record" is a real
    # state of the corpus, and attaching those rows to whichever outcome happened to be
    # commonest would be an invention.
    labeled = [site for site in sites if site.outcome]
    by_outcome = _cluster(site.outcome for site in labeled)

    buckets: dict[str, list[PredictorSite]] = {}
    for site in labeled:
        buckets.setdefault(by_outcome[site.outcome].label, []).append(site)
    for site in sites:
        if not site.outcome:
            buckets.setdefault("", []).append(site)

    groups = [
        OutcomeGroup(
            outcome=label,
            surface_forms=by_outcome[members[0].outcome].forms if label else [],
            sites=sorted(members, key=lambda s: (-(s.year or 0), s.pmid, s.row)),
        )
        for label, members in buckets.items()
    ]
    groups.sort(key=lambda g: (-g.papers, -len(g.sites), g.outcome == "", g.outcome))
    return groups
