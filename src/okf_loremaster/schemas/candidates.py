"""Queries, and the papers they return, up to the point of screening.

`Candidate` is the unit of work from `search` through `rank`. It is deliberately a
different type from `clients.eutils.PubMedRecord`: a record is what one API returned,
while a candidate accumulates across queries — which searches found it, where it ranked
in each, what iCite says about it, what PubTator saw in it.

Direction of dependency is one-way. `schemas` reads from `clients`; nothing in
`clients` knows this module exists.
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import Field

from okf_loremaster.clients.eutils import PubMedRecord
from okf_loremaster.clients.icite import CitationMetrics
from okf_loremaster.clients.pubtator import AnnotatedDocument
from okf_loremaster.schemas.common import Model

__all__ = [
    "Candidate",
    "ExecutedQuery",
    "PlannedQuery",
    "QueryPlan",
    "ScoredCandidate",
]

_PUNCT = re.compile(r"[^a-z0-9 ]+")


class PlannedQuery(Model):
    """One PubMed query the planner proposes."""

    term: str = Field(min_length=1)
    # Why this query exists, in one line. Printed by --dry-run, so a plan can be
    # judged before it costs anything.
    rationale: str = ""
    # The topic this query is meant to fill, when it targets one. Free-form rather than
    # a Slug: the planner may name a topic that the charter does not have, and that
    # mismatch is worth surfacing rather than failing validation over.
    topic: str = ""


class QueryPlan(Model):
    queries: list[PlannedQuery] = Field(default_factory=list)

    @property
    def terms(self) -> tuple[str, ...]:
        return tuple(q.term for q in self.queries)


class ExecutedQuery(Model):
    """What PubMed did with a planned query.

    `translation` is recorded for every query, not just failing ones, because PubMed
    does not reject an unknown field tag — it rewrites `x[nosuchfield]` into
    `"x"[All Fields]`, returns orders of magnitude more hits, and reports an empty
    error list while doing it. A query can be malformed and successful at once, and the
    translation is the only evidence.

    `rationale`, `topic` and `search_round` are carried over from the `PlannedQuery` so
    that this record answers the whole question on its own. They are not used to run
    anything — they exist because `search.md` is written from these objects after the
    plan has gone out of scope, and a query with no account of why it was asked is a
    string a reader has to reverse-engineer.
    """

    term: str
    translation: str = ""
    count: int = 0
    retrieved: int = 0
    fields_not_found: list[str] = Field(default_factory=list)
    # Set by the search node when the translation looks nothing like the query.
    suspect: bool = False
    note: str = ""
    rationale: str = ""
    topic: str = ""
    # 1 for the opening plan, 2 for a gap round. Zero means a record written before this
    # was tracked, which reads as "unknown" rather than as a round.
    search_round: int = 0


class Candidate(Model):
    """A paper under consideration, merged across every query that returned it."""

    pmid: str
    title: str = ""
    abstract: str = ""
    journal: str = ""
    journal_abbrev: str = ""
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    pmcid: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    mesh_major: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    language: str = ""
    source_type: str = "journal"
    is_retracted: bool = False

    # Provenance across the search pass.
    found_by: list[str] = Field(default_factory=list)
    # Best (lowest) zero-based position this paper reached in any query's results.
    best_rank: int = 0

    # Enrichment, all optional: a paper too new to be scored must still be rankable.
    citation_count: int = 0
    rcr: float | None = None
    # Whether iCite actually answered for this paper. `citation_count = 0` cannot say
    # so on its own — it is equally "cited by nobody" and "never asked", and ranking has
    # to tell those apart or a service outage reads as a corpus of uncited papers.
    metrics_known: bool = False
    is_research_article: bool = True
    concepts: dict[str, list[str]] = Field(default_factory=dict)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_record(cls, record: PubMedRecord, *, found_by: str = "", rank: int = 0) -> Self:
        return cls(
            pmid=record.pmid,
            title=record.title,
            abstract=record.abstract,
            journal=record.journal,
            journal_abbrev=record.journal_abbrev,
            year=record.year,
            authors=[a.display for a in record.authors],
            doi=record.doi,
            pmcid=record.pmcid,
            publication_types=list(record.publication_types),
            mesh_terms=[m.descriptor for m in record.mesh_terms],
            mesh_major=[m.descriptor for m in record.mesh_terms if m.major],
            keywords=list(record.keywords),
            language=record.language,
            source_type=record.source_type,
            is_retracted=record.is_retracted,
            found_by=[found_by] if found_by else [],
            best_rank=rank,
        )

    def merged_with(self, other: Candidate) -> Self:
        """Combine two sightings of the same paper.

        Returns a new candidate; neither input is modified. Provenance is a union
        because "found by four independent queries" is itself a ranking signal, and the
        best rank wins because a paper's strongest showing is the fairest summary of
        how the searches saw it.
        """
        merged = self.model_copy(deep=True)
        merged.found_by = list(dict.fromkeys([*self.found_by, *other.found_by]))
        merged.best_rank = min(self.best_rank, other.best_rank)
        for field_name in ("abstract", "doi", "pmcid", "journal", "title"):
            if not getattr(merged, field_name) and getattr(other, field_name):
                setattr(merged, field_name, getattr(other, field_name))
        return merged

    def with_metrics(self, metrics: CitationMetrics) -> Self:
        updated = self.model_copy(deep=True)
        updated.citation_count = metrics.citation_count
        updated.rcr = metrics.relative_citation_ratio
        updated.is_research_article = metrics.is_research_article
        updated.metrics_known = True
        return updated

    def with_concepts(self, document: AnnotatedDocument) -> Self:
        updated = self.model_copy(deep=True)
        updated.concepts = {
            concept_type: list(document.by_type(concept_type))
            for concept_type in document.concept_types
        }
        return updated

    # --- accessors ---------------------------------------------------------

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract.strip())

    @property
    def may_have_full_text(self) -> bool:
        """A PMC id is necessary for full text, never sufficient — most are not open."""
        return self.pmcid is not None

    @property
    def normalized_title(self) -> str:
        """Lowercased, punctuation-free title, for near-duplicate detection.

        The same study reaches PubMed twice often enough to matter: a preprint and its
        journal version, a conference abstract and the paper, a corrected reprint.
        Those carry different PMIDs, so identity alone will not catch them.
        """
        return " ".join(_PUNCT.sub(" ", self.title.lower()).split())

    @property
    def screening_text(self) -> str:
        """What the FAST model sees. Title plus abstract, nothing else.

        MeSH terms are deliberately excluded: they are assigned months after
        publication, so including them would systematically favor older papers in a
        judgment that is supposed to be about content.
        """
        return f"{self.title}\n\n{self.abstract}".strip()

    def citation(self) -> str:
        """One-line human citation, for indexes and pause output."""
        lead = self.authors[0] if self.authors else "Anon"
        et_al = " et al." if len(self.authors) > 1 else ""
        venue = self.journal_abbrev or self.journal
        year = self.year or "n.d."
        return f"{lead}{et_al} {venue} {year}".strip()


class ScoredCandidate(Model):
    """A candidate with the ranker's verdict attached.

    Score and components are kept beside the candidate rather than on it so that the
    same candidate can be scored twice — pure relevance and MMR — and the two rankings
    compared. `--dry-run` prints that comparison, which is the only way to see that
    diversification did anything.
    """

    candidate: Candidate
    score: float = 0.0
    # Named contributions summing to `score`, so a ranking can be explained rather than
    # only observed.
    components: dict[str, float] = Field(default_factory=dict)
    position: int = 0

    @property
    def pmid(self) -> str:
        return self.candidate.pmid
