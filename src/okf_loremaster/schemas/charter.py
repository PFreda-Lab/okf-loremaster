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
from typing import Annotated, Any, Self

import yaml
from pydantic import AliasChoices, Field, field_validator, model_validator

from okf_loremaster.schemas.common import Model, Slug, prose

__all__ = [
    "DEFAULT_MAX_TOPICS",
    "DEFAULT_TARGET_PAPERS",
    "DEFAULT_TOPIC_PAPER_MAX",
    "DEFAULT_TOPIC_PAPER_MIN",
    "MAX_TOPIC_SCOPE_CHARS",
    "PUBMED_LANGUAGES",
    "Charter",
    "Topic",
]

# 120-250 papers is a browsability ceiling, not a recall target: the bundle exists to
# be read end to end by an agent, and past a few hundred documents that stops being
# possible. The per-topic floor matters just as much — a topic of three papers is a
# heading pretending to be a topic.
#
# The middle of that band rather than the top of it. Extraction is one call per kept
# paper and sets a run's price and its wall clock alone; the default is what somebody
# gets before they have any idea what either will be, so it should not be the most
# expensive answer the band allows. Raise it with `--target-papers` once a topic is known
# to be worth 250 papers, which is a judgment nobody can make on the first run.
DEFAULT_TARGET_PAPERS = 150

# Papers *inside* one topic folder, not the number of topics. The two multiply into the
# taxonomy's floor and ceiling — see `capacity` — which is why they were renamed away
# from `topic_min`/`topic_max`: those read as bounds on the topic count.
DEFAULT_TOPIC_PAPER_MIN = 8
DEFAULT_TOPIC_PAPER_MAX = 40

# How many topic folders a bundle may have — the number the charter prompt asks for and
# the number `problems` complains above, which were two different numbers until they
# were merged here. A browsability limit like the others: a reader holding thirty
# headings in mind is reading an ontology rather than a corpus.
DEFAULT_MAX_TOPICS = 8

# One line of prose per topic, shown at the confirmation pause and reused as the topic's
# `index.md` header. A budget rather than a schema constraint — see `common.prose`.
MAX_TOPIC_SCOPE_CHARS = 300

# Every value PubMed's `[la]` field accepts, with the ISO 639-1 code and English name
# that map onto it. Three-letter ISO 639-2/B, which is not the form anyone reaches for
# first: `en`, `de` and `fr` are the codes a web page or an HTTP header uses, and they
# are the ones a model writes when nothing tells it otherwise.
#
# Getting this wrong is invisible at every layer that could catch it. `[la]` is a real
# field tag, so PubMed does not rewrite it the way it rewrites an unknown one; it just
# matches nothing, reports an empty `errorlist`, and returns `Count 0`. Since the filter
# is appended to every query in a plan, one bad code takes the entire run to zero hits —
# which is what it did, with a charter that was otherwise perfectly well drafted.
#
# `""` where a language has no two-letter form.
_LANGUAGES: tuple[tuple[str, str, str], ...] = (
    ("afr", "af", "afrikaans"),
    ("alb", "sq", "albanian"),
    ("amh", "am", "amharic"),
    ("ara", "ar", "arabic"),
    ("arm", "hy", "armenian"),
    ("aze", "az", "azerbaijani"),
    ("ben", "bn", "bengali"),
    ("bos", "bs", "bosnian"),
    ("bul", "bg", "bulgarian"),
    ("cat", "ca", "catalan"),
    ("chi", "zh", "chinese"),
    ("cze", "cs", "czech"),
    ("dan", "da", "danish"),
    ("dut", "nl", "dutch"),
    ("eng", "en", "english"),
    ("epo", "eo", "esperanto"),
    ("est", "et", "estonian"),
    ("fin", "fi", "finnish"),
    ("fre", "fr", "french"),
    ("geo", "ka", "georgian"),
    ("ger", "de", "german"),
    ("gla", "gd", "scottish gaelic"),
    ("gre", "el", "greek"),
    ("heb", "he", "hebrew"),
    ("hin", "hi", "hindi"),
    ("hrv", "hr", "croatian"),
    ("hun", "hu", "hungarian"),
    ("ice", "is", "icelandic"),
    ("ind", "id", "indonesian"),
    ("ita", "it", "italian"),
    ("jpn", "ja", "japanese"),
    ("kin", "rw", "kinyarwanda"),
    ("kor", "ko", "korean"),
    ("lat", "la", "latin"),
    ("lav", "lv", "latvian"),
    ("lit", "lt", "lithuanian"),
    ("mac", "mk", "macedonian"),
    ("mal", "ml", "malayalam"),
    ("mao", "mi", "maori"),
    ("may", "ms", "malay"),
    ("mul", "", "multiple languages"),
    ("nor", "no", "norwegian"),
    ("per", "fa", "persian"),
    ("pol", "pl", "polish"),
    ("por", "pt", "portuguese"),
    ("pus", "ps", "pushto"),
    ("rum", "ro", "romanian"),
    ("rus", "ru", "russian"),
    ("san", "sa", "sanskrit"),
    ("slo", "sk", "slovak"),
    ("slv", "sl", "slovenian"),
    ("spa", "es", "spanish"),
    ("srp", "sr", "serbian"),
    ("swe", "sv", "swedish"),
    ("tha", "th", "thai"),
    ("tur", "tr", "turkish"),
    ("ukr", "uk", "ukrainian"),
    ("und", "", "undetermined"),
    ("urd", "ur", "urdu"),
    ("vie", "vi", "vietnamese"),
    ("wel", "cy", "welsh"),
)

PUBMED_LANGUAGES = frozenset(code for code, _, _ in _LANGUAGES)

# ISO 639-2/T, the other three-letter standard. It agrees with the /B codes above for
# most languages and disagrees for these, which are exactly the languages whose /B code
# came from the English name rather than the endonym. Listed because a wrong code here
# fails the same silent way `en` did, and because a model asked for a three-letter code
# has no way to know which of the two standards a given field wants.
_TERMINOLOGY_CODES = {
    "ces": "cze",
    "cym": "wel",
    "deu": "ger",
    "ell": "gre",
    "fas": "per",
    "fra": "fre",
    "hye": "arm",
    "isl": "ice",
    "kat": "geo",
    "mkd": "mac",
    "mri": "mao",
    "msa": "may",
    "nld": "dut",
    "ron": "rum",
    "slk": "slo",
    "sqi": "alb",
    "zho": "chi",
}

_LANGUAGE_ALIASES = {
    **{iso1: code for code, iso1, _ in _LANGUAGES if iso1},
    **{name: code for code, _, name in _LANGUAGES},
    **_TERMINOLOGY_CODES,
}


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
    # Trimmed rather than rejected — see `common.prose`. This was `max_length=300`, and
    # a scope line eleven characters long killed a live run: the repair round trip
    # re-asked, the model was still never told the number, and it wrote long again.
    scope: Annotated[str, prose(MAX_TOPIC_SCOPE_CHARS)] = ""
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
    #
    # The description is the only place the model is told the format. The charter prompt
    # never asks for this field and a run that leaves it alone gets the default, but a
    # model handed the schema fills it in anyway — and what it wrote, unprompted, was the
    # two-letter code that silently zeroes every query. Cheaper to say than to repair.
    languages: list[str] = Field(
        default_factory=lambda: ["eng"],
        description=(
            "PubMed [la] codes, three letters each: eng, fre, ger, chi, jpn, spa, rus. "
            "Never the two-letter form — PubMed matches nothing on those. Omit for "
            "English only."
        ),
    )
    min_year: int | None = None

    target_papers: int = Field(default=DEFAULT_TARGET_PAPERS, ge=1)
    # `topic_min`/`topic_max` are the names these carried before they were disambiguated;
    # accepted so a charter.yaml written by an older run still loads through --charter.
    topic_paper_min: int = Field(
        default=DEFAULT_TOPIC_PAPER_MIN,
        ge=1,
        validation_alias=AliasChoices("topic_paper_min", "topic_min"),
    )
    topic_paper_max: int = Field(
        default=DEFAULT_TOPIC_PAPER_MAX,
        ge=1,
        validation_alias=AliasChoices("topic_paper_max", "topic_max"),
    )
    max_topics: int = Field(default=DEFAULT_MAX_TOPICS, ge=1)

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
        """Every entry as the code PubMed's `[la]` field actually accepts.

        Accepts what a person or a model plausibly writes — `en`, `eng`, `english`,
        `deu` — and stores the one form PubMed answers to. Deduplicated afterward, since
        two spellings of English would otherwise appear as two clauses of an OR.

        Fatal rather than advisory, unlike a topic scope line that ran long. Prose that
        overruns is a cosmetic defect in something a human is about to read; a language
        code PubMed does not know is appended to every query in the plan and takes the
        whole run to zero hits, an hour and a reasoning call after the mistake was made.
        The message is the repair hint the drafting call gets to retry against, so it is
        written as an instruction rather than a complaint.
        """
        codes: list[str] = []
        unknown: list[str] = []
        for raw in value:
            entry = raw.strip().lower()
            if not entry:
                continue
            code = entry if entry in PUBMED_LANGUAGES else _LANGUAGE_ALIASES.get(entry, "")
            if not code:
                unknown.append(raw.strip())
            elif code not in codes:
                codes.append(code)
        if unknown:
            raise ValueError(
                f"languages: PubMed has no language {', '.join(unknown)} — use its "
                "three-letter [la] codes, such as eng, fre, ger, chi, jpn, spa. PubMed "
                "accepts that field with any value and reports no error, so a code it "
                "does not know returns zero hits for every query in the run"
            )
        return codes

    @model_validator(mode="after")
    def _check_topics(self) -> Self:
        slugs = [topic.slug for topic in self.topic_taxonomy]
        duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
        if duplicates:
            raise ValueError(
                "topic slugs must be unique — they are directory names: "
                + ", ".join(sorted(duplicates))
            )
        if self.topic_paper_min > self.topic_paper_max:
            raise ValueError(
                f"topic_paper_min ({self.topic_paper_min}) exceeds "
                f"topic_paper_max ({self.topic_paper_max})"
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
        return count * self.topic_paper_min, count * self.topic_paper_max

    def problems(self) -> list[str]:
        """Advisory complaints, for the confirmation pause.

        Advisory rather than fatal: these are judgments about a model's output that a
        human is about to look at anyway, and refusing to render a charter is a worse
        outcome than rendering one with a note attached.
        """
        issues: list[str] = []
        if not self.topic_taxonomy:
            issues.append("no topics — every paper would land in one undifferentiated pile")
        if len(self.topic_taxonomy) > self.max_topics:
            issues.append(
                f"{len(self.topic_taxonomy)} topics exceeds the browsable "
                f"maximum of {self.max_topics}"
            )
        floor, ceiling = self.capacity()
        if self.topic_taxonomy and self.target_papers > ceiling:
            issues.append(
                f"target_papers ({self.target_papers}) exceeds what the taxonomy can hold "
                f"({ceiling} = {len(self.topic_taxonomy)} topics "
                f"x topic_paper_max {self.topic_paper_max})"
            )
        if self.topic_taxonomy and self.target_papers < floor:
            issues.append(
                f"target_papers ({self.target_papers}) is below the taxonomy's floor "
                f"({floor} = {len(self.topic_taxonomy)} topics "
                f"x topic_paper_min {self.topic_paper_min})"
            )
        return issues

    # --- serialization -----------------------------------------------------

    def to_yaml(self) -> str:
        """Round-trippable YAML, in declaration order.

        `sort_keys=False` on purpose: the file is meant to be edited by hand between
        the charter pause and the build, and alphabetical order would separate
        `topic_paper_min` from `topic_paper_max` and bury `prompt` in the middle.
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
