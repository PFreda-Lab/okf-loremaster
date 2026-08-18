"""How much weight a paper's evidence can carry. All code, no judgment.

Three things are already called something else and none of them is this:

- `Confidence` is whether the *extraction* read a row correctly. Numeric verification
  downgrades it when a number is not in the source text. It says nothing about the study.
- OKF derives **trust tiers** from the `verified` block, which is about human sign-off.
- `ranking` scores whether a paper is worth *retrieving*, from citations and query
  agreement — signals about a paper's reception, not its methods.

This is the fourth axis: given that we read it correctly, and whoever signed it off, how
good is the study. A well-read row from a 40-person cross-sectional survey should score
`confidence: high` and `strength: limited`, and before this module a bundle had no way
to say so.

**Computed, never asked for.** The ingredients are extracted; the arithmetic is here.
That is the "agents only for judgment" invariant, and three practical consequences
follow from it: the score is reproducible, the weights can change without re-reading a
paper, and every score can be decomposed into the named contributions that produced it.
A model asked for `strength: 0.7` gives a number nobody can check or recompute.

**Run it after verification, never before.** A row whose effect size was deleted for not
appearing in the source text has lost the interval that its precision score reads. Scored
first, it would keep a precision it can no longer support.

Nothing here names a condition, a specialty, or a cohort. The design hierarchy is a
property of study methodology, and the one genuinely field-relative input — how big a
study has to be before it is a big study — is read from the charter, because there is no
answer to it that holds across literatures.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from okf_loremaster.schemas import (
    Charter,
    Extraction,
    PredictorRow,
    StrengthGrade,
    StudyDesign,
    TextBasis,
)
from okf_loremaster.schemas.strength import PaperStrength, RowStrength

__all__ = [
    "DEFAULT_PAPER_WEIGHTS",
    "DEFAULT_ROW_WEIGHTS",
    "PaperWeights",
    "RowWeights",
    "grade_for",
    "score_extraction",
]

# The value a component takes when there is nothing behind it. Neutral rather than zero,
# for the reason `ranking._citation` uses the same number: an unavailable signal should
# be a no-op, not a penalty. A paper is not weak because its abstract omitted a design;
# it is unmeasured, and `unmeasured` on the result says which components those were.
_UNMEASURED = 0.5

# Score bands. Wide on purpose — the inputs are a design category, an order of magnitude
# and two booleans, which does not support finer resolution than three names.
_STRONG_AT = 0.70
_MODERATE_AT = 0.50


@dataclass(frozen=True, slots=True)
class PaperWeights:
    """Relative pull of each paper-level signal. Sums to 1.0.

    Constants rather than configuration, on the same grounds as `ranking.Weights`: a knob
    per signal invites tuning against one project's corpus, which is what would stop the
    scoring generalizing to the next one.
    """

    design: float = 0.35
    size: float = 0.25
    adjustment: float = 0.25
    # How much of the paper we actually read. Not a property of the study, but a bound on
    # what can be claimed about it: an abstract does not print a covariate set, so a
    # strong reading and a weak one are indistinguishable from one. Lowest weight,
    # because the components it limits already register as unmeasured on their own.
    basis: float = 0.15


@dataclass(frozen=True, slots=True)
class RowWeights:
    """Relative pull of each row-level signal. Sums to 1.0."""

    # The study the row came out of. Half, because a row cannot be better evidence than
    # the design that produced it, and it cannot be much worse either.
    paper: float = 0.50
    precision: float = 0.30
    adjusted: float = 0.20


DEFAULT_PAPER_WEIGHTS = PaperWeights()
DEFAULT_ROW_WEIGHTS = RowWeights()

# The conventional evidence hierarchy. Methodology, not subject matter: these same ranks
# hold for any condition, which is why they can live in `src/` when a disease name cannot.
_DESIGN_SCORE: dict[StudyDesign, float] = {
    StudyDesign.SYSTEMATIC_REVIEW: 1.00,
    StudyDesign.RANDOMIZED_TRIAL: 1.00,
    StudyDesign.PROSPECTIVE_COHORT: 0.80,
    StudyDesign.RETROSPECTIVE_COHORT: 0.60,
    StudyDesign.CASE_CONTROL: 0.50,
    StudyDesign.CROSS_SECTIONAL: 0.35,
    # A model fit to data reports what a model does, which is evidence about the world
    # only to the degree the model is right about it.
    StudyDesign.MODELING: 0.35,
    StudyDesign.CASE_SERIES: 0.20,
    StudyDesign.NARRATIVE_REVIEW: 0.15,
}

# Measures whose null value is 1 and whose intervals are multiplicative, so their width
# is only meaningful on a log scale: an odds ratio of 2.0 spanning 1.0-4.0 is as wide as
# one of 0.5 spanning 0.25-1.0, and subtracting the bounds says otherwise. Matched on
# whole words so that "mean difference" does not match on "or".
_RATIO_MEASURE = re.compile(
    r"\b(or|hr|rr|irr|sir|smr|odds ratio|hazard ratio|risk ratio|rate ratio|"
    r"prevalence ratio|incidence ratio|ratio)\b",
    re.IGNORECASE,
)

# A ratio interval spanning a factor of ten carries no useful magnitude. Ten is the
# ceiling rather than the cliff: the score falls linearly in log10 up to it.
_RATIO_SPAN_CEILING = 1.0
# For additive measures, interval width as a multiple of the estimate itself. Four means
# an interval twice as wide as the point estimate on either side scores zero.
_ABSOLUTE_SPAN_CEILING = 4.0


def grade_for(score: float, *, measured: bool = True) -> StrengthGrade:
    """Band a score, or decline to.

    `measured=False` is the case where every component came back unmeasured — the score
    lands at the neutral value by arithmetic rather than by evidence, and reporting that
    as `moderate` would dress up an absence as a finding.
    """
    if not measured:
        return StrengthGrade.UNGRADED
    if score >= _STRONG_AT:
        return StrengthGrade.STRONG
    if score >= _MODERATE_AT:
        return StrengthGrade.MODERATE
    return StrengthGrade.LIMITED


# --- paper level ------------------------------------------------------------


def _design(extraction: Extraction) -> float | None:
    if extraction.design is StudyDesign.UNCLEAR:
        return None
    return _DESIGN_SCORE.get(extraction.design)


def _size(extraction: Extraction, charter: Charter | None) -> float | None:
    """Sample size against the scale this literature works on.

    Log-scaled between the charter's two anchors, so `typical` lands at the neutral 0.5
    and `large` at 1.0. Logarithmic because sample size matters by order of magnitude:
    400 to 800 is the same step as 4,000 to 8,000, and a linear scale would compress
    every study below the largest one into the same score.

    Returns `None` — unmeasured — when the paper gave no size or the charter gave no
    scale. A default scale here would be a constant that is right for one literature and
    quietly wrong for every other, which is the thing this package does not do.
    """
    n = extraction.n
    if n is None or n <= 0 or charter is None:
        return None
    typical, large = charter.sample_size_typical, charter.sample_size_large
    if typical is None or large is None or large <= typical:
        return None
    span = math.log(large) - math.log(typical)
    position = (math.log(n) - math.log(typical)) / span
    return min(1.0, max(0.0, 0.5 + 0.5 * position))


def _adjustment(extraction: Extraction) -> float | None:
    """What share of this paper's estimates held other variables constant.

    Read off the rows, because adjustment is a property of an estimate: papers routinely
    print an unadjusted and an adjusted column, and the second is a different claim.

    Where no row says either way, a stated covariate set is still evidence that the paper
    adjusted something — scored below a row that says so outright, since it does not
    establish that *these* estimates are the adjusted ones.
    """
    known = [row.adjusted for row in extraction.predictors if row.adjusted is not None]
    if known:
        return sum(1.0 for value in known if value) / len(known)
    if extraction.adjusted_for:
        return 0.75
    return None


def _basis(basis: TextBasis) -> float:
    """Always measured: we always know which of the two we read."""
    return 1.0 if basis is TextBasis.FULL_TEXT else 0.4


# --- row level --------------------------------------------------------------


def _precision(row: PredictorRow) -> float | None:
    """How tightly the effect is pinned down, from the interval around it.

    Magnitude is deliberately not a signal. A large effect from a small study is the
    classic artifact — underpowered studies survive publication by overstating, so an
    impressive number with a wide interval is weaker evidence, not stronger. Width is
    what separates the two.

    No interval means unmeasured, and the p-value is not used as a stand-in. A null
    result with a tight interval around no effect is precise evidence of nothing, and a
    p-value cannot tell that apart from a study too small to have found anything.
    """
    low, high = row.ci_low, row.ci_high
    if low is None or high is None or high < low:
        return None
    if _RATIO_MEASURE.search(row.effect_measure) and low > 0:
        span = math.log10(high / low)
        return min(1.0, max(0.0, 1.0 - span / _RATIO_SPAN_CEILING))
    effect = row.effect
    if effect is None or effect == 0.0:
        return None
    relative = (high - low) / abs(effect)
    return min(1.0, max(0.0, 1.0 - relative / _ABSOLUTE_SPAN_CEILING))


def _row_adjusted(row: PredictorRow) -> float | None:
    if row.adjusted is None:
        return None
    # Not zero. An unadjusted estimate is still a measured relationship, and a paper that
    # says plainly that it did not adjust has told the truth about a real result.
    return 1.0 if row.adjusted else 0.2


# --- assembly ---------------------------------------------------------------


def _combine(
    signals: Sequence[tuple[str, float | None, float]],
) -> tuple[float, dict[str, float], list[str]]:
    """Weighted sum over signals, with the unmeasured ones held at neutral.

    Unmeasured components keep their weight rather than having it redistributed. The
    alternative — renormalizing over what is known — makes a paper scored on one signal
    look as authoritative as one scored on four, and hides that behind an identical
    number. Holding them neutral pulls a score toward the middle, which is where a claim
    resting on one measurement belongs.
    """
    parts: dict[str, float] = {}
    unmeasured: list[str] = []
    total = 0.0
    for name, value, weight in signals:
        if value is None:
            unmeasured.append(name)
            value = _UNMEASURED
        contribution = weight * value
        parts[name] = round(contribution, 4)
        total += contribution
    return min(1.0, max(0.0, total)), parts, unmeasured


def score_extraction(
    extraction: Extraction,
    *,
    charter: Charter | None = None,
    basis: TextBasis = TextBasis.ABSTRACT,
    paper_weights: PaperWeights = DEFAULT_PAPER_WEIGHTS,
    row_weights: RowWeights = DEFAULT_ROW_WEIGHTS,
) -> PaperStrength:
    """Score one paper and every row in it.

    The paper score feeds the row scores, so a row cannot outrun the study it came from.
    Rows come back positionally parallel to `extraction.predictors`; call this after the
    length budgets have run, so the two lists cannot disagree about how many rows exist.
    """
    score, parts, unmeasured = _combine(
        [
            ("design", _design(extraction), paper_weights.design),
            ("size", _size(extraction, charter), paper_weights.size),
            ("adjustment", _adjustment(extraction), paper_weights.adjustment),
            ("basis", _basis(basis), paper_weights.basis),
        ]
    )
    paper = PaperStrength(
        score=round(score, 4),
        # `basis` is always measured, so a paper is never wholly ungraded. The check is
        # kept anyway: it is the honest condition, and it stops a future weight of zero
        # on `basis` from silently making every paper look graded.
        grade=grade_for(score, measured=len(unmeasured) < 4),
        parts=parts,
        unmeasured=unmeasured,
    )
    paper.rows = [
        _score_row(row, paper_score=score, weights=row_weights) for row in extraction.predictors
    ]
    return paper


def _score_row(row: PredictorRow, *, paper_score: float, weights: RowWeights) -> RowStrength:
    score, parts, unmeasured = _combine(
        [
            ("paper", paper_score, weights.paper),
            ("precision", _precision(row), weights.precision),
            ("adjusted", _row_adjusted(row), weights.adjusted),
        ]
    )
    return RowStrength(
        score=round(score, 4),
        grade=grade_for(score, measured=len(unmeasured) < 3),
        parts=parts,
        unmeasured=unmeasured,
    )
