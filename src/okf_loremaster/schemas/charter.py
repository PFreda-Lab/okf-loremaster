"""The charter: everything the run derives from the user's prompt, before any search.

This is the only place a run learns what it is about. The shelf taxonomy, the
vocabularies, the population and the outcome are all decided here by one DEEP call and
then govern every later node — so the charter is written to disk as YAML, shown to the
user at a confirmation pause, and editable by hand before the build proceeds.

`vocabularies` is the field that fails silently. It gates what every extraction is
allowed to record, it is chosen before a single paper has been read, and a key that is
missing here produces files that are merely thinner rather than obviously wrong. Three
things guard it: it is printed at the charter pause, `--vocab` overrides it, and
whatever an extraction wanted to record under an unlisted key survives in
`unmapped_vocab` for `validate` to report.

Nothing in this module names a condition, a specialty, or a coding system. Every such
value arrives at runtime from the model that read the prompt.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Self

import yaml
from pydantic import Field, field_validator, model_validator

from okf_loremaster.schemas.common import Model, Slug

__all__ = [
    "DEFAULT_SHELF_MAX",
    "DEFAULT_SHELF_MIN",
    "DEFAULT_TARGET_PAPERS",
    "MAX_SHELVES",
    "Charter",
    "Shelf",
]

# 120-250 papers is a browsability ceiling, not a recall target: the bundle exists to
# be read end to end by an agent, and past a few hundred documents that stops being
# possible. The per-shelf floor matters just as much — a shelf of three papers is a
# heading pretending to be a topic.
DEFAULT_TARGET_PAPERS = 180
DEFAULT_SHELF_MIN = 8
DEFAULT_SHELF_MAX = 40
MAX_SHELVES = 12


class Shelf(Model):
    """One folder of the bundle.

    `slug` becomes the directory name *and* the `domain` frontmatter key in every file
    inside it; downstream treats a mismatch between the two as a validation error.
    """

    slug: Slug
    title: str = Field(min_length=1)
    # One line, in the charter's own words, describing what belongs here. Shown at the
    # confirmation pause and reused as the shelf's index.md header, so it is written
    # for a reader rather than for a prompt.
    scope: str = Field(default="", max_length=300)
    # Concepts to seed query planning for this shelf. Suggestions, not a query.
    seed_terms: list[str] = Field(default_factory=list)


class Charter(Model):
    """The run's terms of reference."""

    # Verbatim, so the bundle can always answer "what was this built for".
    prompt: str
    # The prompt restated as an objective the later nodes can be judged against.
    task: str = ""
    population: str = ""
    outcome: str = ""

    inclusion: list[str] = Field(default_factory=list)
    exclusion: list[str] = Field(default_factory=list)

    shelf_taxonomy: list[Shelf] = Field(default_factory=list)
    # Runtime-keyed. Whatever coding systems the charter judges relevant, lowercased;
    # never a list this package chose.
    vocabularies: list[str] = Field(default_factory=list)

    # PubMed language codes. A filter, not a judgment about what is worth reading.
    languages: list[str] = Field(default_factory=lambda: ["eng"])
    min_year: int | None = None

    target_papers: int = Field(default=DEFAULT_TARGET_PAPERS, ge=1)
    shelf_min: int = Field(default=DEFAULT_SHELF_MIN, ge=1)
    shelf_max: int = Field(default=DEFAULT_SHELF_MAX, ge=1)

    generated_by: str = ""
    generated_at: datetime | None = None

    # --- validation --------------------------------------------------------

    @field_validator("vocabularies", mode="after")
    @classmethod
    def _normalize_vocabularies(cls, value: list[str]) -> list[str]:
        """Lowercase, strip, drop blanks, dedupe — order preserved.

        Keys are matched against extraction output, so `ICD10` and `icd10` arriving as
        two different vocabularies would split one column in half.
        """
        seen: dict[str, None] = {}
        for raw in value:
            key = raw.strip().lower()
            if key:
                seen.setdefault(key, None)
        return list(seen)

    @field_validator("languages", mode="after")
    @classmethod
    def _normalize_languages(cls, value: list[str]) -> list[str]:
        return [code.strip().lower() for code in value if code.strip()]

    @model_validator(mode="after")
    def _check_shelves(self) -> Self:
        slugs = [shelf.slug for shelf in self.shelf_taxonomy]
        duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
        if duplicates:
            raise ValueError(
                "shelf slugs must be unique — they are directory names: "
                + ", ".join(sorted(duplicates))
            )
        if self.shelf_min > self.shelf_max:
            raise ValueError(
                f"shelf_min ({self.shelf_min}) exceeds shelf_max ({self.shelf_max})"
            )
        return self

    # --- accessors ---------------------------------------------------------

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(shelf.slug for shelf in self.shelf_taxonomy)

    def shelf(self, slug: str) -> Shelf | None:
        return next((s for s in self.shelf_taxonomy if s.slug == slug), None)

    def capacity(self) -> tuple[int, int]:
        """(floor, ceiling) papers implied by the taxonomy and the per-shelf bounds."""
        count = len(self.shelf_taxonomy)
        return count * self.shelf_min, count * self.shelf_max

    def problems(self) -> list[str]:
        """Advisory complaints, for the confirmation pause.

        Advisory rather than fatal: these are judgments about a model's output that a
        human is about to look at anyway, and refusing to render a charter is a worse
        outcome than rendering one with a note attached.
        """
        issues: list[str] = []
        if not self.shelf_taxonomy:
            issues.append("no shelves — every paper would land in one undifferentiated pile")
        if len(self.shelf_taxonomy) > MAX_SHELVES:
            issues.append(
                f"{len(self.shelf_taxonomy)} shelves exceeds the browsable maximum of {MAX_SHELVES}"
            )
        if not self.vocabularies:
            issues.append(
                "no vocabularies — extractions will record no coding hints at all; "
                "pass --vocab to set them explicitly"
            )
        floor, ceiling = self.capacity()
        if self.shelf_taxonomy and self.target_papers > ceiling:
            issues.append(
                f"target_papers ({self.target_papers}) exceeds what the taxonomy can hold "
                f"({ceiling} = {len(self.shelf_taxonomy)} shelves x shelf_max {self.shelf_max})"
            )
        if self.shelf_taxonomy and self.target_papers < floor:
            issues.append(
                f"target_papers ({self.target_papers}) is below the taxonomy's floor "
                f"({floor} = {len(self.shelf_taxonomy)} shelves x shelf_min {self.shelf_min})"
            )
        return issues

    # --- serialization -----------------------------------------------------

    def to_yaml(self) -> str:
        """Round-trippable YAML, in declaration order.

        `sort_keys=False` on purpose: the file is meant to be edited by hand between
        the charter pause and the build, and alphabetical order would separate
        `shelf_min` from `shelf_max` and bury `prompt` in the middle.
        """
        payload = self.model_dump(mode="json", exclude_none=True)
        return str(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("charter file must contain a YAML mapping")
        return cls.model_validate(loaded)

    def digest(self) -> str:
        """Stable hash of the decisions, for the run manifest.

        Excludes `generated_at` and `generated_by`: two runs from the same edited
        charter should agree, and they would not if a timestamp were in the hash.
        """
        payload: dict[str, Any] = self.model_dump(
            mode="json", exclude={"generated_at", "generated_by"}
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
