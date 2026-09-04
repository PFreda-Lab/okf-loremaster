"""Types shared across the schema package.

Every model here inherits `Model`, whose one deliberate setting is `extra="ignore"`.
Most of these types are the target of a structured model response, and a model that
volunteers one extra key should not cost the whole extraction. Missing and mistyped
fields still fail loudly; only unexpected ones are dropped.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StringConstraints,
    model_validator,
)

from okf_loremaster.basis import TextBasisPolicy
from okf_loremaster.schemas.limits import truncate_chars

__all__ = [
    "PERMISSIVE_LICENSES",
    "Confidence",
    "Direction",
    "EvidenceType",
    "InteractionMagnitude",
    "InteractionType",
    "Model",
    "Slug",
    "StrengthGrade",
    "StudyDesign",
    "TextBasis",
    "TextBasisPolicy",
    "band_interaction",
    "filename_token",
    "is_export_safe",
    "prose",
    "slugify",
]

# Topic slugs and the `domain` frontmatter key. Lowercase, digits, single hyphens,
# no leading or trailing hyphen — because the slug is also a directory name and must
# survive a case-insensitive filesystem, a URL, and a YAML scalar unquoted.
SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"

Slug = Annotated[str, StringConstraints(pattern=SLUG_PATTERN, min_length=2, max_length=64)]


def prose(limit: int) -> AfterValidator:
    """A model-written prose field, trimmed at `limit` characters rather than rejected.

    Use for any text a model writes that nothing downstream parses — a rationale, a
    scope line, a screener's note. `max_length` on one of those makes a sentence a few
    characters too long a `ValidationError`, and a `ValidationError` on a model reply is
    a **failed node**: the repair round trip re-asks and, unless the prompt states the
    number, the model writes long again. That killed a live run over a topic scope
    eleven characters over.

    `limits.py` has said *truncate and warn, never reject* from the beginning, on the
    reasoning that an over-long reply is a good reply that ran on and re-asking pays for
    a whole second call to fix a formatting problem. This is that rule, as a type.
    `truncate_chars` leaves an ellipsis, so a trimmed line says so where a reader sees it.

    A hard cap still belongs on anything read by something other than a person — `Slug`
    above becomes a directory name and the `domain` key in every file under it, and a
    truncated one is a broken bundle rather than a shortened sentence.
    """
    return AfterValidator(lambda value: truncate_chars(value, limit)[0])


_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9]+")

# A JSON escape that outlived the decoder. A model writing `\\u2265` where it meant
# `≥` produces, after `json.loads`, the six characters `\`, `u`, `2`, `2`, `6`, `5`
# rather than `≥` — valid JSON the whole way, and wrong from the moment it is read.
_STRAY_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")

# Unicode categories a bundle must never contain: controls, format characters,
# surrogates, private use, unassigned, and the two separators that end a line. An escaped
# newline decoded in place would split a markdown table row in half, so these are left as
# they came and reported by the validator rather than quietly turned into structure.
_UNWRITABLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})


def _decode_stray_escapes(text: str) -> str:
    """Turn a `\\uXXXX` that survived JSON decoding back into the character it names.

    Only above ASCII, and only for a character that can be written. An escape for an
    ASCII character is never necessary in a JSON reply, and decoding one could introduce
    a quote, a backslash or a pipe into text that a markdown table or a JSONL row is
    about to be built from — so those are left alone deliberately, and the bundle
    validator errors on anything still escaped when the writing is done.
    """
    if "\\u" not in text:
        return text

    def one(match: re.Match[str]) -> str:
        char = chr(int(match.group(1), 16))
        if ord(char) < 0x80 or unicodedata.category(char) in _UNWRITABLE_CATEGORIES:
            return match.group(0)
        return char

    return _STRAY_ESCAPE.sub(one, text)


def _repaired(value: Any) -> Any:
    """`value` with every string in it decoded. Keys are left alone — they are ASCII."""
    if isinstance(value, str):
        return _decode_stray_escapes(value)
    if isinstance(value, dict):
        return {key: _repaired(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_repaired(item) for item in value]
    return value


class Model(BaseModel):
    """Base for every schema in this package."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        # Enum members serialize as their string value, so a dumped model is plain
        # JSON/YAML that a downstream reader can parse without importing us.
        use_enum_values=False,
        validate_default=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _decode_escapes(cls, data: Any) -> Any:
        """Undo a JSON escape a model wrote twice, before anything is built from it.

        Here rather than in `parse.py` because a reply is not the only way one of these
        gets in. The same value comes back out of the extraction cache, and out of a
        `charter.yaml` a person edited, and each of those is a door `parse.py` does not
        stand in — so a cache written before this existed would replay the defect on the
        next resume, and only the model boundary was ever guarded.

        Observed on 12 of 1025 cached extractions (1.2%), against 486 that carried the
        same characters correctly, so this is a model that occasionally double-escapes
        rather than a writer that always does. Every emitter downstream was already
        writing UTF-8 faithfully; what they were handed was already the escape text.

        It matters more than it looks. A consumer that scans outbound text for runs of
        seven or more digits — the shape of a patient identifier, which nothing else
        distinguishes — reads `\\u2265100,000` as `2265100` and refuses the bundle. The
        correct `≥100,000` carries no such run. That is a stopped run, from prose.
        """
        return _repaired(data) if isinstance(data, dict | list) else data


class Confidence(StrEnum):
    """How much weight a single extracted claim can carry.

    Ordered, because numeric verification downgrades rather than discards: a claim
    whose number cannot be found in the source text is still evidence that the paper
    discussed the construct, just not evidence of the magnitude.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1}[self]

    @property
    def downgraded(self) -> Confidence:
        """One step lower, saturating at LOW."""
        return {
            Confidence.HIGH: Confidence.MEDIUM,
            Confidence.MEDIUM: Confidence.LOW,
            Confidence.LOW: Confidence.LOW,
        }[self]


class StudyDesign(StrEnum):
    """How the study was built, as one of the standard designs.

    Distinct from `Extraction.study_design`, which keeps the paper's own words. This is
    the normalized category, and it exists because the words cannot be scored: "a
    retrospective review of prospectively collected registry data" and "chart review"
    describe the same design and share no vocabulary.

    The ordering is the conventional evidence hierarchy, which is a property of study
    methodology rather than of any condition — the same reason MeSH and `[tiab]` belong
    in this package while a disease name does not. `UNCLEAR` is a real answer and the
    right one whenever a paper does not say; it scores as unmeasured rather than as bad.
    """

    SYSTEMATIC_REVIEW = "systematic_review"
    RANDOMIZED_TRIAL = "randomized_trial"
    PROSPECTIVE_COHORT = "prospective_cohort"
    RETROSPECTIVE_COHORT = "retrospective_cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_SERIES = "case_series"
    MODELING = "modeling"
    NARRATIVE_REVIEW = "narrative_review"
    UNCLEAR = "unclear"

    @property
    def label(self) -> str:
        """Short form for a markdown table or a frontmatter line."""
        return self.value.replace("_", " ")


class StrengthGrade(StrEnum):
    """A banded evidence-strength score, for reading rather than for sorting.

    Deliberately not called confidence or trust, both of which are taken and mean other
    things here. `Confidence` is whether the extraction read a row correctly; OKF derives
    its own *trust tiers* from the `verified` block, which is about human sign-off. This
    third thing is about the study: how much weight its design, size and analysis can
    carry, whoever read it and whoever signed it off.

    The score is the number to sort on. The grade is what a reader skims, and it is
    banded because a bundle that ranks paper 0.61 above paper 0.60 is claiming a
    precision the inputs do not have.
    """

    STRONG = "strong"
    MODERATE = "moderate"
    LIMITED = "limited"
    UNGRADED = "ungraded"


class EvidenceType(StrEnum):
    """What kind of claim a predictor row is making.

    Without this, "X predicts Y" and "changing X changes Y" land in the same table and
    read as the same claim. A meaningful share of any retained corpus is trials, so the
    distinction is not an edge case — it is the difference between a feature worth
    engineering and an intervention that cannot be one.

    `OUTCOME_DEFINITION` is the third kind on purpose: papers that define or validate
    how the outcome itself is measured carry no predictor at all, and would otherwise
    be forced into a shape that misrepresents them.
    """

    OBSERVATIONAL_ASSOCIATION = "observational_association"
    RANDOMIZED_INTERVENTION = "randomized_intervention"
    OUTCOME_DEFINITION = "outcome_definition"

    @property
    def label(self) -> str:
        """Short form for a markdown table column."""
        return {
            EvidenceType.OBSERVATIONAL_ASSOCIATION: "association",
            EvidenceType.RANDOMIZED_INTERVENTION: "intervention",
            EvidenceType.OUTCOME_DEFINITION: "outcome def.",
        }[self]


class Direction(StrEnum):
    """Sign of the reported relationship, independent of the effect measure.

    An odds ratio below 1 and a negative coefficient mean the same thing and are easy
    to read backwards, so the direction is recorded rather than inferred at render time.
    """

    INCREASES = "increases"
    DECREASES = "decreases"
    NONE = "none"
    UNCLEAR = "unclear"


class InteractionType(StrEnum):
    """How one variable in a study acts on another. A closed, methodological list.

    Closed because a downstream agent has to be able to branch on it, and free text
    would give it a synonym set to normalize before it could. Methodological because
    that is the level this package is allowed to name: correlation, mutual exclusivity,
    effect modification, confounding, mediation and derivation are properties of study
    design, not of any condition — the same reason `StudyDesign` belongs here and a
    disease name does not.

    Six relationships, each with the label for its mirror. Two are symmetric and are
    their own inverse; the other four are directional, and a mirrored entry that kept
    the forward label would reverse the claim. Reciprocity is computed in
    `interactions.py` rather than asked of the model — a model asked to state both
    halves states them inconsistently, and the second half is free to derive.
    """

    CORRELATED = "correlated"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    MODIFIES = "modifies"
    MODIFIED_BY = "modified_by"
    CONFOUNDS = "confounds"
    CONFOUNDED_BY = "confounded_by"
    MEDIATES = "mediates"
    MEDIATED_BY = "mediated_by"
    DERIVED_FROM = "derived_from"
    DERIVES = "derives"

    @property
    def label(self) -> str:
        """Short form for a markdown table column."""
        return self.value.replace("_", " ")

    @property
    def inverse(self) -> InteractionType:
        """The same relationship, stated from the other variable's side."""
        return _INVERSE_INTERACTIONS[self]

    @property
    def structural(self) -> bool:
        """Whether this relationship holds by construction rather than by measurement.

        Two variables that cannot both be present, or one computed from the other, are
        related as strongly as anything can be and there is no coefficient to band. The
        magnitude for those is `structural`, which says more than any number would.
        """
        return self in _STRUCTURAL_INTERACTIONS


_INVERSE_INTERACTIONS: dict[InteractionType, InteractionType] = {
    InteractionType.CORRELATED: InteractionType.CORRELATED,
    InteractionType.MUTUALLY_EXCLUSIVE: InteractionType.MUTUALLY_EXCLUSIVE,
    InteractionType.MODIFIES: InteractionType.MODIFIED_BY,
    InteractionType.MODIFIED_BY: InteractionType.MODIFIES,
    InteractionType.CONFOUNDS: InteractionType.CONFOUNDED_BY,
    InteractionType.CONFOUNDED_BY: InteractionType.CONFOUNDS,
    InteractionType.MEDIATES: InteractionType.MEDIATED_BY,
    InteractionType.MEDIATED_BY: InteractionType.MEDIATES,
    InteractionType.DERIVED_FROM: InteractionType.DERIVES,
    InteractionType.DERIVES: InteractionType.DERIVED_FROM,
}

_STRUCTURAL_INTERACTIONS = frozenset(
    {
        InteractionType.MUTUALLY_EXCLUSIVE,
        InteractionType.DERIVED_FROM,
        InteractionType.DERIVES,
    }
)


class InteractionMagnitude(StrEnum):
    """How much one variable acts on another. Computed, never asked for.

    A fifth word for "how much do I believe this" would be a disaster in a table that
    already carries four, so this one is deliberately not called strength: `strength` is
    study quality, `Confidence` is whether the row was read right, OKF's trust tiers are
    human sign-off, `ranking.relevance` is worth retrieving. This is the size of one
    relationship between two variables and nothing else.

    Five values, and only three of them are a band. `STRUCTURAL` is true by construction
    and has no coefficient; `STATED` is a relationship the paper asserts in prose without
    a number that can be banded. Both are more honest than inventing a band for them, and
    the verbatim measure prints beside all five either way.
    """

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    STRUCTURAL = "structural"
    STATED = "stated"


# Cutoffs by measure, as `(strong, moderate)` thresholds on the absolute value, and
# whether the scale runs the other way. Only measures whose scale means the same thing in
# every literature are listed: a correlation coefficient and a variance inflation factor
# are statistics, so they belong in this package under the same rule that admits
# `StudyDesign` and excludes a disease name.
#
# An odds ratio, a beta and a hazard ratio are deliberately absent. Their scale depends on
# the units of the variable and on the outcome's base rate, so a fixed cutoff would be
# wrong somewhere and there is no way to be right everywhere. They band as `stated`, which
# is what the paper actually told us.
#
# A p-value is absent for a different and sharper reason. A p for an interaction term is
# evidence that an interaction *exists*; it says nothing about how big it is, and a large
# study reports a vanishing p for an effect too small to matter. Banding on it would put
# `strong` next to a relationship the paper itself calls negligible.
_INTERACTION_CUTOFFS: dict[str, tuple[float, float, bool]] = {
    # Correlation and association coefficients, all on a 0-1 scale.
    "r": (0.7, 0.4, False),
    "rho": (0.7, 0.4, False),
    "rs": (0.7, 0.4, False),
    "tau": (0.7, 0.4, False),
    "phi": (0.7, 0.4, False),
    "kappa": (0.7, 0.4, False),
    "cramersv": (0.7, 0.4, False),
    "icc": (0.7, 0.4, False),
    # The squares of the same cutoffs, so a paper reporting shared variance bands where
    # the coefficient behind it would have.
    "r2": (0.49, 0.16, False),
    # Collinearity diagnostics. The conventional thresholds, and `tolerance` is 1/VIF, so
    # its scale is inverted — a *small* tolerance is a strong relationship.
    "vif": (10.0, 5.0, False),
    "tolerance": (0.1, 0.2, True),
}


def band_interaction(
    kind: InteractionType, measure: str, value: float | None
) -> InteractionMagnitude:
    """How large one interaction is, from what the paper printed. No judgment.

    Three outcomes and they are checked in this order. A relationship that holds by
    construction is `structural` whatever number sits beside it — a coefficient between
    two variables that cannot co-occur is a description of the study's design, not of its
    findings. Otherwise a recognized measure with a value bands; everything else is
    `stated`, which is not a failure but the ordinary answer for a paper that reports an
    interaction in prose.
    """
    if kind.structural:
        return InteractionMagnitude.STRUCTURAL
    cutoffs = _INTERACTION_CUTOFFS.get(_canonical_measure(measure))
    if cutoffs is None or value is None:
        return InteractionMagnitude.STATED
    strong, moderate, inverted = cutoffs
    magnitude = abs(value)
    if (magnitude <= strong) if inverted else (magnitude >= strong):
        return InteractionMagnitude.STRONG
    if (magnitude <= moderate) if inverted else (magnitude >= moderate):
        return InteractionMagnitude.MODERATE
    return InteractionMagnitude.WEAK


# Superscripts, which is how `R²` reaches us from a typeset paper.
_SUPERSCRIPTS = str.maketrans("\N{SUPERSCRIPT TWO}", "2")


def _canonical_measure(measure: str) -> str:
    """Fold a measure name onto a key in the table above, or onto nothing.

    Papers name a statistic three ways — the bare symbol, the symbol with its author
    (`Pearson r`, `Spearman rho`), and the author's name alone (`Cramer's V`) — so
    neither the last token nor the whole string is reliably the key on its own. Both are
    tried, most specific first, and a miss folds to `""`, which bands as `stated`. That
    is the right answer for an unrecognized measure and the reason nothing here guesses.
    """
    normalized = measure.lower().translate(_SUPERSCRIPTS)
    tokens = "".join(ch if ch.isalnum() else " " for ch in normalized).split()
    if not tokens:
        return ""
    for key in ("".join(tokens), tokens[-1]):
        if key in _INTERACTION_CUTOFFS:
            return key
    return ""


class TextBasis(StrEnum):
    """What the extraction actually read.

    Most of any corpus is not in the open-access subset, so abstract-only records are
    normal rather than exceptional. Recording which is which lets a downstream reader
    weight them differently instead of assuming every file had the same evidence behind
    it.
    """

    FULL_TEXT = "full_text"
    ABSTRACT = "abstract"


# Re-exported, not defined here: `cli.py` declares `--basis` and must not import
# pydantic to do it, so the policy enum lives in a stdlib-only module. It belongs in this
# namespace anyway — the run-level instruction and the per-paper observation are read
# together, and separating them by import path is how the two get confused.


# Licenses under which redistributing a derived, quoting summary is unambiguous.
# Conservative on purpose:
#   - ND forbids derivative works, and a structured extraction is a derivative.
#   - NC depends on the recipient's use, which we cannot know at build time.
#   - an empty or unrecognized string means we do not know, which is not a yes.
# This gates `okf-loremaster export --permissive-only`. It is a build-time filter,
# not legal advice.
PERMISSIVE_LICENSES = frozenset({"cc0", "cc-by", "cc by", "cc-by-sa", "cc by-sa", "public domain"})


def is_export_safe(license_text: str) -> bool:
    """Whether a bundle file derived from this license may be redistributed."""
    normalized = " ".join(license_text.lower().split())
    if not normalized:
        return False
    # BioC reports licenses as "CC BY", "CC BY-NC", "CC BY-NC-ND", "CC0", "NO-CC CODE".
    if "nc" in normalized.replace("-", " ").split() or "nd" in normalized.replace("-", " ").split():
        return False
    return normalized in PERMISSIVE_LICENSES


def slugify(text: str) -> str:
    """Fold arbitrary text into a `Slug`, for a topic folder name.

    Returns `""` for input with no alphanumeric content, which callers must handle
    rather than write a directory named `-`.
    """
    folded = _NON_ALNUM.sub("-", text.lower()).strip("-")
    return folded[:64].rstrip("-")


def filename_token(text: str) -> str:
    """Fold a name into a filename component, keeping its capitalization.

    Used for the author half of `<pmid>_<Author>.md`. Case is preserved because the
    filename is what an agent sees in a topic index and cites back at us, and
    `33745404_Ferrari-Silva.md` is legible in a way the lowercased form is not.
    Everything outside `[A-Za-z0-9]` becomes a hyphen, so the result is safe on a
    case-insensitive filesystem and inside a URL alike.
    """
    folded = _FILENAME_UNSAFE.sub("-", text).strip("-")
    return folded[:64].rstrip("-")
