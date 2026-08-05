"""Types shared across the schema package.

Every model here inherits `Model`, whose one deliberate setting is `extra="ignore"`.
Most of these types are the target of a structured model response, and a model that
volunteers one extra key should not cost the whole extraction. Missing and mistyped
fields still fail loudly; only unexpected ones are dropped.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

from okf_loremaster.schemas.limits import truncate_chars

__all__ = [
    "PERMISSIVE_LICENSES",
    "Confidence",
    "Direction",
    "EvidenceType",
    "Model",
    "Slug",
    "StrengthGrade",
    "StudyDesign",
    "TextBasis",
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


class TextBasis(StrEnum):
    """What the extraction actually read.

    Most of any corpus is not in the open-access subset, so abstract-only records are
    normal rather than exceptional. Recording which is which lets a downstream reader
    weight them differently instead of assuming every file had the same evidence behind
    it.
    """

    FULL_TEXT = "full_text"
    ABSTRACT = "abstract"


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
