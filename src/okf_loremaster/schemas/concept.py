"""One paper, as the bundle records it.

Split in two on purpose. `Extraction` is everything a model decided; `ConceptRecord`
wraps it with everything that was read verbatim from PubMed, PMC, or config. So
`record.title` came from an API and `record.extraction.bottom_line` came from a model,
and no reader has to guess which is which. It also means the emitter can write
bibliographic frontmatter without ever consulting the model's output.

Two invariants are enforced here as code rather than asked for in a prompt, because a
prompt instruction that is ignored produces a file that looks fine:

- **`null_findings` is never empty.** A validator inserts a `none reported` row. A
  missing section and a section reporting nothing are different claims, and the second
  one is evidence.
- **A code is never recorded on its own.** A `VocabularyHint` is a concept the paper
  named, and the codes it gave for that same concept hang off it. A bare code with no
  concept beside it is unusable to a reader that thinks in English, so the model
  cannot produce one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from okf_loremaster.schemas.common import (
    Confidence,
    Direction,
    EvidenceType,
    Model,
    Slug,
    StudyDesign,
    TextBasis,
    filename_token,
    is_export_safe,
)
from okf_loremaster.schemas.limits import (
    MAX_BODY_WORDS,
    MAX_BOTTOM_LINE_SENTENCES,
    MAX_CAVEAT_SENTENCES,
    MAX_DESCRIPTION_CHARS,
    MAX_NULL_FINDINGS,
    MAX_PREDICTOR_ROWS,
    MAX_TAGS,
    MAX_VOCABULARY_HINTS,
    truncate_chars,
    truncate_sentences,
    word_count,
)
from okf_loremaster.schemas.strength import PaperStrength

__all__ = [
    "NONE_REPORTED",
    "CodedAs",
    "ConceptRecord",
    "Extraction",
    "NullFinding",
    "PredictorRow",
    "SourceRef",
    "Verification",
    "VocabularyHint",
]

# The sentinel that makes "nothing null was reported" a statement rather than a gap.
NONE_REPORTED = "none reported"

PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
PMC_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
DOI_URL = "https://doi.org/{doi}"


class PredictorRow(Model):
    """One relationship the paper reports, as a row in `# Predictors reported`."""

    # Named `predictor` rather than `construct`, which is the better word for it: a
    # pydantic field called `construct` shadows the deprecated `BaseModel.construct`
    # and emits a UserWarning at import, which every CLI invocation would then print.
    predictor: str = Field(min_length=1)
    # How it was actually measured or defined in this study — the difference between a
    # concept and a feature someone can build.
    operationalization: str = ""
    # When it is observed relative to the outcome. A predictor measured after the
    # outcome window is a leak, and this field is what lets anyone notice.
    timing: str = ""
    outcome: str = ""
    evidence_type: EvidenceType = EvidenceType.OBSERVATIONAL_ASSOCIATION

    effect: float | None = None
    # "OR", "adjusted OR", "HR", "RR", "beta", "AUC", "mean difference".
    effect_measure: str = ""
    # The effect exactly as the paper wrote it, e.g. "1.82 (95% CI 1.21-2.74)".
    # Numeric verification checks this string against the source text, so it must be
    # copied rather than reformatted.
    effect_raw: str = ""
    ci_low: float | None = None
    ci_high: float | None = None
    # A string, not a float: "<0.001" and "NS" are how papers actually report this, and
    # coercing them loses the distinction between "very small" and "not given".
    p_value: str = ""
    direction: Direction = Direction.UNCLEAR
    confidence: Confidence = Confidence.MEDIUM
    # Whether *this estimate* came out of a model holding other variables constant. A
    # per-row fact, not a per-paper one: papers routinely print an unadjusted and an
    # adjusted column, and they are different claims about the same predictor.
    #
    # `None` means the paper did not say, which scores as unmeasured. False means it
    # said, and the answer was no. Collapsing the two would penalize a paper for its
    # reader's ignorance.
    adjusted: bool | None = None
    # The sentence the numbers came from, verbatim. The basis for verification and for
    # a downstream agent to quote without re-reading the paper.
    quote: str = ""

    def downgraded(self) -> Self:
        """Drop the number and lower the confidence, keeping the claim.

        What numeric verification does when it cannot find `effect_raw` in the source
        text. The predictor, timing and operationalization were still reported by the
        paper; only the magnitude is unsupported, so discarding the whole row would
        throw away good evidence to punish one bad field.
        """
        stripped = self.model_copy(deep=True)
        stripped.effect = None
        stripped.ci_low = None
        stripped.ci_high = None
        stripped.confidence = self.confidence.downgraded
        return stripped

    def without_interval(self) -> Self:
        """Drop the interval and lower the confidence, keeping the point estimate.

        The lighter half of the same idea: verification found `effect` in the source
        but not one of its bounds. Discarding a supported point estimate because the
        interval around it is unsupported would lose the more useful of the two.
        """
        stripped = self.model_copy(deep=True)
        stripped.ci_low = None
        stripped.ci_high = None
        stripped.confidence = self.confidence.downgraded
        return stripped

    @property
    def has_effect(self) -> bool:
        return self.effect is not None


class NullFinding(Model):
    """Something the paper looked for and did not find.

    Kept because it is the part of the literature that never gets written down
    anywhere else, and because a downstream agent proposing features benefits more from
    "this was tested and did not hold" than from silence.
    """

    predictor: str = Field(default=NONE_REPORTED, min_length=1)
    outcome: str = ""
    detail: str = ""
    quote: str = ""

    @property
    def is_sentinel(self) -> bool:
        return self.predictor.strip().lower() == NONE_REPORTED


class CodedAs(Model):
    """One code a paper gave, and the system it belongs to.

    `system` is lowercased and stripped of punctuation so `ICD-10`, `icd 10` and
    `icd10` are one system rather than three — a model naming the same standard three
    ways would otherwise split one column of a downstream match into thirds.

    Nothing here restricts which systems are sayable. Whatever the paper used is what
    gets written down; this package does not hold a list of approved standards.
    """

    system: str = Field(min_length=1)
    code: str = Field(min_length=1)

    @field_validator("system", mode="after")
    @classmethod
    def _canonical_system(cls, value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    @field_validator("code", mode="after")
    @classmethod
    def _tidy_code(cls, value: str) -> str:
        return value.strip()


class VocabularyHint(Model):
    """A variable the paper names, plus any codes it gave for that same variable.

    The concept leads because a reader thinks in English by preference and in codes by
    capability. Codes are optional and usually absent: most papers name a variable and
    never code it, and `codes: []` is the honest record of that, not a gap.
    """

    # The paper's own words for the variable.
    concept: str = Field(min_length=1)
    codes: list[CodedAs] = Field(default_factory=list)

    @field_validator("concept", mode="after")
    @classmethod
    def _tidy_concept(cls, value: str) -> str:
        return value.strip()

    @field_validator("codes", mode="after")
    @classmethod
    def _dedupe_codes(cls, value: list[CodedAs]) -> list[CodedAs]:
        seen: dict[tuple[str, str], CodedAs] = {}
        for entry in value:
            seen.setdefault((entry.system, entry.code), entry)
        return list(seen.values())


class Extraction(Model):
    """Everything a model decided about one paper. No bibliographic data."""

    # Two lines, for the topic index and for frontmatter. The highest-signal browse
    # field in the bundle: it is what an agent reads before deciding to open the file.
    description: str = ""
    # The finding, in at most two sentences.
    bottom_line: str = ""

    study_design: str = ""
    # The same design as one of the standard categories, so it can be scored. Kept
    # beside the free text rather than replacing it: the paper's own words are what a
    # reader wants to see, and the category is what evidence strength needs.
    design: StudyDesign = StudyDesign.UNCLEAR
    # Analytic sample size. None when the paper does not state one, which is common in
    # reviews and never worth inventing.
    n: int | None = None
    population: str = ""
    outcome_definition: str = ""
    # The covariates the paper says it adjusted for, in its own words. Recorded at the
    # paper level because that is where methods sections state it once; whether any
    # *given* estimate was adjusted is `PredictorRow.adjusted`.
    adjusted_for: list[str] = Field(default_factory=list)

    predictors: list[PredictorRow] = Field(default_factory=list)
    null_findings: list[NullFinding] = Field(default_factory=list)

    # What the paper calls its variables, and whatever codes it gave for them. Free
    # text and free systems — nothing in this package decides which are allowed.
    vocabulary_hints: list[VocabularyHint] = Field(default_factory=list)

    caveats: str = ""
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _null_findings_are_never_absent(self) -> Self:
        """The invariant, enforced where it cannot be skipped.

        Asking a prompt for this works most of the time, which is the problem: the
        failures are invisible, because an omitted section and a section that reports
        nothing render identically as an empty table.
        """
        if not self.null_findings:
            self.null_findings = [NullFinding(predictor=NONE_REPORTED)]
        return self

    @property
    def reports_null_findings(self) -> bool:
        return any(not f.is_sentinel for f in self.null_findings)

    def body_words(self) -> int:
        """Approximate word count of the rendered body, for the length budget."""
        cells = [
            text
            for row in self.predictors
            for text in (
                row.predictor,
                row.operationalization,
                row.timing,
                row.outcome,
                row.effect_raw,
                row.quote,
            )
        ]
        cells += [
            text
            for finding in self.null_findings
            for text in (finding.predictor, finding.outcome, finding.detail)
        ]
        prose = [self.bottom_line, self.population, self.outcome_definition, self.caveats]
        return sum(word_count(text) for text in prose + cells)

    def enforce_budgets(self) -> tuple[Self, list[str]]:
        """Trim to the length budgets, returning the trimmed copy and what was cut.

        Truncate and warn rather than reject: an over-long extraction is a good one
        that ran on, and re-asking pays for a whole second reading to fix a formatting
        problem.
        Dropped predictor rows are named in the warning so nothing vanishes quietly.
        """
        trimmed = self.model_copy(deep=True)
        warnings: list[str] = []

        trimmed.description, cut = truncate_chars(trimmed.description, MAX_DESCRIPTION_CHARS)
        if cut:
            warnings.append(f"description truncated to {MAX_DESCRIPTION_CHARS} characters")

        trimmed.bottom_line, cut = truncate_sentences(
            trimmed.bottom_line, MAX_BOTTOM_LINE_SENTENCES
        )
        if cut:
            warnings.append(f"bottom_line truncated to {MAX_BOTTOM_LINE_SENTENCES} sentences")

        trimmed.caveats, cut = truncate_sentences(trimmed.caveats, MAX_CAVEAT_SENTENCES)
        if cut:
            warnings.append(f"caveats truncated to {MAX_CAVEAT_SENTENCES} sentences")

        if len(trimmed.predictors) > MAX_PREDICTOR_ROWS:
            # The model's own ordering is its judgment of importance, so the tail goes
            # rather than the lowest-confidence rows wherever they sit.
            dropped = [row.predictor for row in trimmed.predictors[MAX_PREDICTOR_ROWS:]]
            trimmed.predictors = trimmed.predictors[:MAX_PREDICTOR_ROWS]
            warnings.append(
                f"dropped {len(dropped)} predictor row(s) over the limit of "
                f"{MAX_PREDICTOR_ROWS}: {', '.join(dropped)}"
            )

        # Same rule as the predictor rows, and added for the same reason: these two were
        # the last uncapped lists in the schema, so they were where a reply spent the
        # tokens it needed to finish. The tail goes, because a model lists what it thinks
        # matters first.
        if len(trimmed.null_findings) > MAX_NULL_FINDINGS:
            dropped = [row.predictor for row in trimmed.null_findings[MAX_NULL_FINDINGS:]]
            trimmed.null_findings = trimmed.null_findings[:MAX_NULL_FINDINGS]
            warnings.append(
                f"dropped {len(dropped)} null finding(s) over the limit of "
                f"{MAX_NULL_FINDINGS}: {', '.join(dropped)}"
            )

        if len(trimmed.vocabulary_hints) > MAX_VOCABULARY_HINTS:
            dropped = [hint.concept for hint in trimmed.vocabulary_hints[MAX_VOCABULARY_HINTS:]]
            trimmed.vocabulary_hints = trimmed.vocabulary_hints[:MAX_VOCABULARY_HINTS]
            warnings.append(
                f"dropped {len(dropped)} vocabulary hint(s) over the limit of "
                f"{MAX_VOCABULARY_HINTS}: {', '.join(dropped)}"
            )

        if len(trimmed.tags) > MAX_TAGS:
            trimmed.tags = trimmed.tags[:MAX_TAGS]
            warnings.append(f"tags truncated to {MAX_TAGS}")

        words = trimmed.body_words()
        if words > MAX_BODY_WORDS:
            # Not truncated further: every field is already within its own budget, so
            # the only remaining lever would be cutting a table mid-row. Flagged so a
            # persistent overrun shows up in `validate` rather than nowhere.
            warnings.append(f"body is ~{words} words, over the ~{MAX_BODY_WORDS} word guideline")

        return trimmed, warnings


class SourceRef(Model):
    """An OKF `sources` entry — where this document's content came from."""

    id: str
    resource: str = ""
    last_modified: str = ""
    usage_count: int | None = None


class Verification(Model):
    """An OKF `verified` entry. Written only by `--review` sign-off.

    Absent means the spec's `unverified` tier, which is the honest tier for machine
    extraction. A self-attestation written by the same process that did the extracting
    would discriminate nothing.
    """

    by: str
    at: datetime


class ConceptRecord(Model):
    """One concept file: verbatim provenance plus one model's reading of the paper."""

    # --- read from PubMed, never from a model ---
    pmid: str
    title: str = ""
    journal: str = ""
    journal_abbrev: str = ""
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    pmcid: str | None = None

    # --- decided by the pipeline ---
    domain: Slug
    # Verbatim from BioC's `infons.license`, never inferred. Empty for an
    # abstract-only record, where no license was ever served to us.
    license: str = ""
    text_basis: TextBasis = TextBasis.ABSTRACT

    # Evidence strength, computed by `strength.py` after verification. On the record
    # rather than inside `Extraction` because `Extraction` is what the extract node hands
    # the model as its response format: a field there is a field the model is asked to
    # fill, and a model-supplied strength score is the one thing this must never be.
    strength: PaperStrength | None = None

    # --- decided by a model ---
    extraction: Extraction

    # --- provenance ---
    generated_by: str = ""
    generated_at: datetime | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    verified: list[Verification] = Field(default_factory=list)

    @property
    def export_safe(self) -> bool:
        """Whether this file may be redistributed, computed from the license.

        Never model-supplied and never defaulted to true: an unknown license is a no.
        """
        return is_export_safe(self.license)

    @property
    def first_author_surname(self) -> str:
        if not self.authors:
            return "Anon"
        # Display form is "Surname Initials"; initials are a trailing token of capitals.
        parts = self.authors[0].split()
        if len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 4:
            parts = parts[:-1]
        return " ".join(parts) or "Anon"

    @property
    def filename(self) -> str:
        """`<pmid>_<Author>.md`, matching the layout AFCE's resolver expects."""
        author = filename_token(self.first_author_surname) or "Anon"
        return f"{self.pmid}_{author}.md"

    @property
    def resource_url(self) -> str:
        return PUBMED_URL.format(pmid=self.pmid)

    def default_sources(self) -> list[SourceRef]:
        """The `sources` entries implied by the identifiers we hold."""
        refs = [SourceRef(id=f"pmid:{self.pmid}", resource=self.resource_url)]
        if self.pmcid:
            refs.append(
                SourceRef(id=f"pmc:{self.pmcid}", resource=PMC_URL.format(pmcid=self.pmcid))
            )
        if self.doi:
            refs.append(SourceRef(id=f"doi:{self.doi}", resource=DOI_URL.format(doi=self.doi)))
        return refs
