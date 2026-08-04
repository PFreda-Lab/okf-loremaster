"""The charter: everything the run derives from the user's prompt, before any search.

This is the only place a run learns what it is about. The topic taxonomy, the population
and the outcome are all decided here by one reasoning-tier call and then govern every
later node — so the charter is written to disk as YAML, shown to the user at a
confirmation pause, and editable by hand before the build proceeds.

There is deliberately no list of coding systems here. An extraction records whatever
vocabulary the paper itself used, so nothing has to be guessed in advance — see
`schemas/concept.VocabularyHint`.

Nothing in this module names a condition, a specialty, or a cohort. Every such value
arrives at runtime from the model that read the prompt.
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
    "DEFAULT_TARGET_PAPERS",
    "DEFAULT_TOPIC_MAX",
    "DEFAULT_TOPIC_MIN",
    "MAX_TOPICS",
    "Charter",
    "Topic",
]

# 120-250 papers is a browsability ceiling, not a recall target: the bundle exists to
# be read end to end by an agent, and past a few hundred documents that stops being
# possible. The per-topic floor matters just as much — a topic of three papers is a
# heading pretending to be a topic.
DEFAULT_TARGET_PAPERS = 200
DEFAULT_TOPIC_MIN = 8
DEFAULT_TOPIC_MAX = 40
MAX_TOPICS = 12


class Topic(Model):
    """One folder of the bundle.

    `slug` becomes the directory name *and* the `domain` frontmatter key in every file
    inside it; downstream treats a mismatch between the two as a validation error.
    """

    slug: Slug
    title: str = Field(min_length=1)
    # One line, in the charter's own words, describing what belongs here. Shown at the
    # confirmation pause and reused as the topic's index.md header, so it is written
    # for a reader rather than for a prompt.
    scope: str = Field(default="", max_length=300)
    # Concepts to seed query planning for this topic. Suggestions, not a query.
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

    topic_taxonomy: list[Topic] = Field(default_factory=list)

    # PubMed language codes. A filter, not a judgment about what is worth reading.
    languages: list[str] = Field(default_factory=lambda: ["eng"])
    min_year: int | None = None

    target_papers: int = Field(default=DEFAULT_TARGET_PAPERS, ge=1)
    topic_min: int = Field(default=DEFAULT_TOPIC_MIN, ge=1)
    topic_max: int = Field(default=DEFAULT_TOPIC_MAX, ge=1)

    # What counts as a small and a large study *in this literature*. Evidence strength
    # scores sample size against these, and they are here rather than in `src/` because
    # there is no answer that holds across literatures: a few hundred participants is a
    # large cohort for a rare condition and a pilot for a national registry. A constant
    # would be wrong for most projects and invisible in all of them.
    #
    # Both null is a legitimate state — a hand-written charter, or a model that declined
    # to guess — and scores sample size as unmeasured rather than as poor.
    sample_size_typical: int | None = Field(default=None, ge=1)
    sample_size_large: int | None = Field(default=None, ge=1)

    generated_by: str = ""
    generated_at: datetime | None = None

    # --- validation --------------------------------------------------------

    @field_validator("languages", mode="after")
    @classmethod
    def _normalize_languages(cls, value: list[str]) -> list[str]:
        return [code.strip().lower() for code in value if code.strip()]

    @model_validator(mode="after")
    def _check_topics(self) -> Self:
        slugs = [topic.slug for topic in self.topic_taxonomy]
        duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
        if duplicates:
            raise ValueError(
                "topic slugs must be unique — they are directory names: "
                + ", ".join(sorted(duplicates))
            )
        if self.topic_min > self.topic_max:
            raise ValueError(
                f"topic_min ({self.topic_min}) exceeds topic_max ({self.topic_max})"
            )
        return self

    @model_validator(mode="after")
    def _check_sample_size_scale(self) -> Self:
        """Both ends or neither, and in the right order.

        Half a scale scores nothing, so a charter carrying one end is a silent no-op
        rather than a partial answer. An inverted pair is worse than a no-op: it would
        score every large study as small. Both are the model's or the editor's mistake
        and both are cheap to state plainly here.
        """
        typical, large = self.sample_size_typical, self.sample_size_large
        if (typical is None) != (large is None):
            raise ValueError(
                "sample_size_typical and sample_size_large go together — one without "
                "the other is not a scale, and sample size would score as unmeasured"
            )
        if typical is not None and large is not None and large <= typical:
            raise ValueError(
                f"sample_size_large ({large}) must exceed sample_size_typical ({typical})"
            )
        return self

    # --- accessors ---------------------------------------------------------

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(topic.slug for topic in self.topic_taxonomy)

    def topic(self, slug: str) -> Topic | None:
        return next((s for s in self.topic_taxonomy if s.slug == slug), None)

    def capacity(self) -> tuple[int, int]:
        """(floor, ceiling) papers implied by the taxonomy and the per-topic bounds."""
        count = len(self.topic_taxonomy)
        return count * self.topic_min, count * self.topic_max

    def problems(self) -> list[str]:
        """Advisory complaints, for the confirmation pause.

        Advisory rather than fatal: these are judgments about a model's output that a
        human is about to look at anyway, and refusing to render a charter is a worse
        outcome than rendering one with a note attached.
        """
        issues: list[str] = []
        if not self.topic_taxonomy:
            issues.append("no topics — every paper would land in one undifferentiated pile")
        if len(self.topic_taxonomy) > MAX_TOPICS:
            issues.append(
                f"{len(self.topic_taxonomy)} topics exceeds the browsable maximum of {MAX_TOPICS}"
            )
        floor, ceiling = self.capacity()
        if self.topic_taxonomy and self.target_papers > ceiling:
            issues.append(
                f"target_papers ({self.target_papers}) exceeds what the taxonomy can hold "
                f"({ceiling} = {len(self.topic_taxonomy)} topics x topic_max {self.topic_max})"
            )
        if self.topic_taxonomy and self.target_papers < floor:
            issues.append(
                f"target_papers ({self.target_papers}) is below the taxonomy's floor "
                f"({floor} = {len(self.topic_taxonomy)} topics x topic_min {self.topic_min})"
            )
        return issues

    # --- serialization -----------------------------------------------------

    def to_yaml(self) -> str:
        """Round-trippable YAML, in declaration order.

        `sort_keys=False` on purpose: the file is meant to be edited by hand between
        the charter pause and the build, and alphabetical order would separate
        `topic_min` from `topic_max` and bury `prompt` in the middle.
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
