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
pass over phrases that differ by exactly one qualifier — never more than one, and never
one that flips the meaning, which is what `_NOT_A_QUALIFIER` is for. Every merge prints the
surface forms it absorbed, so any of them can be disputed, and the heading over them is
never allowed to assert something only some of them say.

Both halves of that rule are narrower than they first were, and both were narrowed by
the same failure on a real corpus: a normalization that discarded too much handed the
containment pass keys short enough to contain almost anything, and it absorbed three
unrelated phrases before anything noticed. Normalization that deletes is the dangerous
kind — what it deletes, no later guard can protect.

Embeddings would cluster better and are not used: `--finalize okf` never builds them, so
the common path would silently get the worse clustering, and reaching for them here would
drag the `[vectors]` extra into a file every run writes.

Nothing here names a condition, a specialty or a cohort. The word lists are properties of
English and of research design rather than of a literature — `high`, `former` and `reduced`
flip a meaning in any field, and `treatment`, `prevention` and `cessation` turn an exposure
into an intervention in any field, which is the test `CLAUDE.md` sets for a constant that
may live in `src/`.
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

# A parenthetical span, and the abbreviations inside one. Only the abbreviation is
# dropped, and only inside the parentheses.
#
# Dropping the whole span was the first attempt and it was wrong in the worst available
# direction. Papers put two very different things in parentheses: an initialism they
# defined for themselves — "Chronic sleep restriction (CSR)" — and the cutoff that *is*
# the variable — "Sleep duration (short, <=6h/d)". Discarding both collapsed the second
# kind onto a bare `{sleep, duration}` key shared with "SD of sleep duration" and
# "Sleep duration (>=9h vs 7-9h)", which merged short sleep, long sleep and a
# variability measure into one entry labeled for whichever was commonest. The qualifier
# never reached `_POLARITY`, because it had already been deleted.
#
# So the initialism goes and the words stay. An initialism is an all-caps run: it may
# carry digits and hyphens (`HOMA-IR`, `REM`), and a normally capitalized word does not
# match, because the character after its first letter is lowercase.
_PARENTHETICAL = re.compile(r"\(([^)]*)\)")
_ABBREVIATION = re.compile(r"\b[A-Z][A-Z0-9-]*\b")

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

# Words that name something done *to* the variable rather than a property *of* it. A
# corpus reported "postoperative pain" as raising the risk of its outcome and
# "postoperative pain treatment" as lowering it; one differing token, no polarity word in
# sight, and the two merged under a single heading that then carried both signs. Treating
# a thing is not a narrower reading of having it — it is the opposite intervention on it —
# so these are refused for the same reason `_POLARITY` is, one level up: polarity flips
# where a variable sits on its axis, these flip whether the row is about the variable at
# all.
#
# English and research design, not medicine. "Treatment", "prevention" and "cessation"
# turn an exposure into an intervention in any literature, which is the test `CLAUDE.md`
# sets for a constant that may live here.
_INTERVENTION = frozenset(
    """
    adherence administration avoidance cessation control discontinuation initiation
    intervention management prevention preventive prophylaxis screening supplementation
    therapy treated treatment withdrawal
    """.split()  # noqa: SIM905  — a wrapped word block reads better than a quoted list
)

# One differing token drawn from either set is not a qualifier, and the merge is refused.
_NOT_A_QUALIFIER = _POLARITY | _INTERVENTION

# Plural forms these endings produce are not plurals. Stripping the `s` from `status`,
# `analysis` or `stress` would make three words that no longer match themselves.
_NOT_A_PLURAL = ("ss", "us", "is", "os", "as")

# A threshold written into a predictor's name: a comparator and a number, with whatever
# unit follows. Used only to decide a heading, never to build a key — a cutoff is content
# and `surface_key` keeps it.
_THRESHOLD = re.compile(r"[<>≤≥=~]*\s*\d+(?:\.\d+)?\s*[\w/%]*")
_EMPTIED = re.compile(r"\(\s*\)")
_DANGLING = re.compile(r"[\s,;:]+([)\]])")


def surface_key(text: str) -> frozenset[str]:
    """The token set two phrases are compared on.

    Self-defined abbreviations dropped, tokenized the way everything else in this
    package tokenizes (`ranking.text_tokens`, so stopwords and one- and two-character
    fragments go), then singularized. A set rather than a sequence: "duration of sleep"
    and "sleep duration" are the same phrase, and word order is not evidence otherwise.

    The one departure from `text_tokens` is that a two-letter initialism is put back. Its
    length rule exists for a retrieval index, where a two-character fragment is noise; a
    predictor name is three or four words long, and in "SD of sleep duration" the two
    characters carry the entire construct. Dropping them left it indistinguishable from
    "sleep duration" itself, so a variability measure grouped with the thing it varies.
    """
    opened = _unabbreviated(text)
    tokens = {_singular(token) for token in text_tokens(opened)}
    return frozenset(tokens | {word.lower() for word in _ABBREVIATION.findall(opened) if word[1:]})


def _unabbreviated(text: str) -> str:
    """Parentheses opened, and only the initialisms inside them removed."""
    return _PARENTHETICAL.sub(lambda m: f" {_ABBREVIATION.sub(' ', m.group(1))} ", text)


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(_NOT_A_PLURAL):
        return token[:-1]
    return token


def _absorbs(host: frozenset[str], guest: frozenset[str]) -> bool:
    """Whether `guest` is `host` plus *one* qualifier that does not change the claim.

    Containment rather than overlap, because that is the shape the near-duplicates
    actually take: a paper writes the phrase, and the next paper writes the phrase with a
    word in front of it. Overlap would also merge two phrases that share most of their
    words while differing on the one that matters.

    One word, not several. A short key is a subset of a great many longer ones —
    `{sleep, duration}` contains itself in "apnoea duration during REM sleep" and in
    "sleep fragmentation without reduction in sleep duration" — and each extra word is
    another chance that the two phrases stopped being about the same thing. One word is
    what "the same phrase with a qualifier" actually means, and the cases that need more
    than one are better left as two adjacent entries a reader can join.

    Three kinds of single word are still refused. A polarity word points the phrase the
    other way, and an intervention word makes it about acting on the variable rather than
    having it — `_NOT_A_QUALIFIER` is both. And an initialism is not a qualifier at all:
    "SD of sleep duration" is a statistic computed over the variable rather than a narrower
    reading of it, and the same holds for whatever a given field abbreviates. Only an
    initialism can leave a token this short in a key — `text_tokens` drops every other
    two-character fragment — so the length is a reliable test for one without naming any.
    """
    if not host or not guest or host == guest:
        return False
    if not (host < guest or guest < host):
        return False
    difference = host ^ guest
    if len(difference) > 1:
        return False
    if difference & _NOT_A_QUALIFIER:
        return False
    return all(len(token) > 2 for token in difference)


def _numbers(text: str) -> frozenset[str]:
    """Every number a surface form carries. Two forms agree when these match."""
    return frozenset(re.findall(r"\d+(?:\.\d+)?", text))


def _without_thresholds(text: str) -> str:
    """The same phrase with its cutoffs removed, and the punctuation they leave tidied.

    Only ever reached when no member of the group was written without one, so the choice
    is between a heading that names the group and a heading that names one of its rows.
    Falls back to the original if there is nothing left, which is the case for a predictor
    whose whole name is a number.
    """
    stripped = _DANGLING.sub(r"\1", _EMPTIED.sub(" ", _THRESHOLD.sub(" ", text)))
    return " ".join(stripped.split()).strip(" ,;:-.") or text


@dataclass(slots=True)
class _Cluster:
    """One accumulating group of surface forms that mean the same thing."""

    key: frozenset[str]
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def label(self) -> str:
        """The commonest surface form, then the shortest, then alphabetical — but never
        one asserting a threshold the group does not agree on.

        Commonest because it is how the corpus mostly says it; the rest of the ordering
        exists so that two runs over the same records agree on the heading.

        The threshold rule was added after two corpora in a row printed a heading that was
        false for something underneath it: `Comorbidity index score (≥1)` over a group that
        also held `Charlson Comorbidity Index (CCI) score ≥8`, and before that a `≥3` over
        forms carrying no cutoff at all. A heading is read as the name of the group, so a
        cutoff in it is a claim about every row. Merging papers that dichotomized the same
        variable at different points is right — they are studying the same thing — but the
        name of the result cannot be one of the cutoffs. So a form without one wins if the
        corpus wrote any, and if every form carries one the cutoff comes off the heading.
        `surface_forms` still prints all of them, which is where the detail belongs.
        """
        ranked = sorted(self.counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        forms = [form for form, _ in ranked]
        if len({_numbers(form) for form in forms}) == 1:
            return forms[0]
        return next((form for form in forms if not _numbers(form)), _without_thresholds(forms[0]))

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
