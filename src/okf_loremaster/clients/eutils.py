"""NCBI E-utilities: `esearch` and `efetch`.

Only these two. `esummary` returns a strict subset of what `efetch` gives us and the
pipeline never needs a title-only record, so carrying it would be an untested API
surface with no caller.

XML is parsed with the standard library's ElementTree, which does not expand internal
or external entities — it raises on an undefined one — so the usual XML entity attacks
do not apply and a third-party parser buys nothing here.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from okf_loremaster.clients._http import HttpClient
from okf_loremaster.config import Settings
from okf_loremaster.events import WarningEvent

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI's documented ceiling for a URL-encoded id list. Larger sets are chunked.
EFETCH_BATCH = 200
# esearch will not return more than this many ids in one call, whatever retmax says.
ESEARCH_MAX_RETMAX = 10_000
# Which papers a capped query returns, and so the one parameter that decides whether
# repeating a search gets the same corpus. Named rather than written inline because
# `search.md` reports it: PubMed recomputes this ranking as its index grows, and a bundle
# that does not say how its results were ordered cannot be replayed honestly.
ESEARCH_SORT = "relevance"

# PubMed marks retractions in two places and they do not always agree: the retracted
# article usually gains this publication type, but sometimes only carries a
# CommentsCorrections pointer to the notice.
_RETRACTED_TYPES = frozenset({"Retracted Publication", "Retraction of Publication"})


@dataclass(frozen=True, slots=True)
class MeshTerm:
    descriptor: str
    ui: str
    major: bool
    qualifiers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Author:
    """Surname kept separate from initials, deliberately.

    Compound surnames are ordinary — PubMed's `LastName` for PMID 33745404 is
    "Ferrari Silva" — so a display string like "Ferrari Silva B" cannot be split back
    apart on whitespace. Taking the first token yields "Ferrari", which is the wrong
    name, and it would go into a filename and a citation without ever looking wrong.
    """

    surname: str
    initials: str = ""
    collective: bool = False

    @property
    def display(self) -> str:
        return f"{self.surname} {self.initials}".strip()


@dataclass(frozen=True, slots=True)
class ESearchResult:
    """What `esearch` returned, including how PubMed actually read the query.

    `query_translation` is load-bearing, not diagnostic. PubMed silently rewrites an
    unrecognized field tag into a free-text search — `foo[nosuchfield]` becomes
    `"foo"[All Fields]` and returns thousands of plausible hits with an *empty*
    `errorlist` (verified 2026-08-03). A generated query can therefore be malformed
    and successful at the same time. The only signal available to the caller is the
    translation, so it is returned rather than discarded.
    """

    term: str
    count: int
    ids: tuple[str, ...]
    query_translation: str = ""
    fields_not_found: tuple[str, ...] = ()
    phrases_not_found: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        return self.count > len(self.ids)


@dataclass(frozen=True, slots=True)
class PubMedRecord:
    pmid: str
    title: str
    abstract: str
    journal: str
    journal_abbrev: str
    year: int | None
    pub_date_raw: str
    authors: tuple[Author, ...]
    doi: str | None
    pmcid: str | None
    publication_types: tuple[str, ...]
    mesh_terms: tuple[MeshTerm, ...]
    keywords: tuple[str, ...]
    language: str
    # "journal" or "book". PubMed indexes book chapters (GeneReviews, StatPearls)
    # alongside articles; they are reviews rather than primary evidence, so screening
    # needs to be able to tell them apart.
    source_type: str = "journal"
    retraction_refs: tuple[str, ...] = field(default=())

    @property
    def first_author_surname(self) -> str:
        """Surname of the first author, verbatim.

        May contain spaces. Slugifying it for the `<pmid>_<Author>.md` filename is the
        emitter's job, not this one's — truncating here would lose the real name.
        """
        return self.authors[0].surname if self.authors else "Anon"

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract.strip())

    @property
    def is_retracted(self) -> bool:
        return bool(
            self.retraction_refs or (_RETRACTED_TYPES & set(self.publication_types))
        )


class EUtilsClient:
    """Thin, typed wrapper. Rate limiting and caching live in `HttpClient`."""

    def __init__(self, http: HttpClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    def _common(self) -> dict[str, str]:
        # NCBI asks every caller to identify itself and throttles traffic that does
        # not. The API key is what lifts the rate ceiling.
        params = {"tool": self._settings.ncbi_tool}
        if self._settings.ncbi_email:
            params["email"] = self._settings.ncbi_email
        if self._settings.ncbi_api_key:
            params["api_key"] = self._settings.ncbi_api_key
        return params

    async def esearch(
        self,
        term: str,
        *,
        db: str = "pubmed",
        retmax: int = 200,
        retstart: int = 0,
        sort: str = ESEARCH_SORT,
        node: str = "search",
    ) -> ESearchResult:
        params = {
            **self._common(),
            "db": db,
            "term": term,
            "retmax": str(min(retmax, ESEARCH_MAX_RETMAX)),
            "retstart": str(retstart),
            "sort": sort,
            "retmode": "json",
        }
        raw = await self._http.get_text(f"{BASE}/esearch.fcgi", params=params, node=node)
        return _parse_esearch(term, raw)

    async def efetch(
        self,
        pmids: Sequence[str],
        *,
        node: str = "search",
    ) -> list[PubMedRecord]:
        """Fetch full PubMed records, chunked to NCBI's per-request id limit.

        Warns when PubMed returns fewer records than were asked for. A short response
        is not an error and PubMed gives no indication of one, so without this a
        parser gap or a withdrawn id just quietly shrinks the corpus.
        """
        requested = [str(p) for p in pmids]
        records: list[PubMedRecord] = []
        for chunk in _chunks(requested, EFETCH_BATCH):
            params = {
                **self._common(),
                "db": "pubmed",
                "id": ",".join(chunk),
                "retmode": "xml",
            }
            raw = await self._http.get_text(
                f"{BASE}/efetch.fcgi", params=params, node=node
            )
            records.extend(parse_pubmed_xml(raw))

        missing = [p for p in requested if p not in {r.pmid for r in records}]
        if missing and self._http.bus is not None:
            self._http.bus.emit(
                WarningEvent(
                    node=node,
                    message=(
                        f"efetch returned {len(records)} of {len(requested)} "
                        f"requested record(s); missing "
                        f"{', '.join(missing[:5])}"
                        f"{'...' if len(missing) > 5 else ''}"
                    ),
                )
            )
        return records


# --- parsing ---------------------------------------------------------------


def _parse_esearch(term: str, raw: str) -> ESearchResult:
    try:
        payload: dict[str, Any] = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"esearch returned non-JSON: {raw[:200]!r}") from exc

    result = payload.get("esearchresult", {})
    if "ERROR" in result:
        raise ValueError(f"esearch rejected the query: {result['ERROR']}")

    errors = result.get("errorlist", {}) or {}
    return ESearchResult(
        term=term,
        count=int(result.get("count", 0)),
        ids=tuple(result.get("idlist", [])),
        query_translation=str(result.get("querytranslation", "")),
        fields_not_found=tuple(errors.get("fieldsnotfound", []) or []),
        phrases_not_found=tuple(errors.get("phrasesnotfound", []) or []),
    )


def parse_pubmed_xml(raw: str) -> list[PubMedRecord]:
    """Parse an efetch `PubmedArticleSet` into records.

    Handles both element types the set can contain. `PubmedBookArticle` is a *sibling*
    of `PubmedArticle`, not a variant of it, so a parser that looks only for the latter
    drops book chapters silently — returning fewer records than ids requested, with no
    error anywhere.

    Public because the emitter tests and the fixture recorder both parse saved XML
    without going through a client.
    """
    root = ET.fromstring(raw)
    records: list[PubMedRecord] = []
    for child in root:  # iterated in document order, so input order is preserved
        if child.tag == "PubmedArticle":
            records.append(_journal_record(child))
        elif child.tag == "PubmedBookArticle":
            records.append(_book_record(child))
    return records


def _article_ids(container: ET.Element, path: str) -> dict[str, str]:
    """Identifiers belonging to *this* record.

    Scoped deliberately. `.//ArticleIdList/ArticleId` also matches every
    `Reference/ArticleIdList` in the cited-reference list — 19 matches instead of 4 on
    a real record (verified 2026-08-03) — so an unscoped lookup silently returns some
    cited paper's PMC id and DOI. That misroutes the full-text fetch and attributes
    another article's text to this one.
    """
    return {
        (el.get("IdType") or "").lower(): (el.text or "").strip()
        for el in container.findall(path)
    }


def _journal_record(article: ET.Element) -> PubMedRecord:
    citation = article.find("MedlineCitation")
    citation = citation if citation is not None else article
    art = citation.find("Article")
    art = art if art is not None else citation

    ids = _article_ids(article, "PubmedData/ArticleIdList/ArticleId")
    journal = art.find("Journal")
    year, pub_date_raw = _pub_date(journal)

    return PubMedRecord(
        pmid=_text(citation.find("PMID")) or ids.get("pubmed", ""),
        title=_text(art.find("ArticleTitle")),
        abstract=_abstract(art.find("Abstract")),
        journal=_text(journal.find("Title")) if journal is not None else "",
        journal_abbrev=(
            _text(journal.find("ISOAbbreviation")) if journal is not None else ""
        ),
        year=year,
        pub_date_raw=pub_date_raw,
        authors=_authors(art.find("AuthorList")),
        doi=ids.get("doi") or None,
        pmcid=ids.get("pmc") or None,
        publication_types=tuple(
            _text(el) for el in art.findall("PublicationTypeList/PublicationType")
        ),
        mesh_terms=_mesh(citation.find("MeshHeadingList")),
        keywords=tuple(
            t
            for t in (_text(el) for el in citation.findall("KeywordList/Keyword"))
            if t
        ),
        language=_text(art.find("Language")),
        source_type="journal",
        retraction_refs=tuple(
            _text(el.find("PMID"))
            for el in citation.findall("CommentsCorrectionsList/CommentsCorrections")
            if el.get("RefType") == "RetractionIn"
        ),
    )


def _book_record(article: ET.Element) -> PubMedRecord:
    """A `PubmedBookArticle` — GeneReviews, StatPearls and similar.

    The shape differs enough to warrant its own function: the container is
    `BookDocument`, bibliographic data hangs off `Book`, and `PublicationType` is a
    direct child rather than sitting in a `PublicationTypeList`.
    """
    doc = article.find("BookDocument")
    doc = doc if doc is not None else article
    book = doc.find("Book")

    ids = _article_ids(doc, "ArticleIdList/ArticleId")
    year_text = _text(book.find("PubDate/Year")) if book is not None else ""
    title = _text(doc.find("ArticleTitle")) or (
        _text(book.find("BookTitle")) if book is not None else ""
    )

    return PubMedRecord(
        pmid=_text(doc.find("PMID")) or ids.get("pubmed", ""),
        title=title,
        abstract=_abstract(doc.find("Abstract")),
        journal=_text(book.find("BookTitle")) if book is not None else "",
        journal_abbrev="",
        year=int(year_text) if year_text.isdigit() else None,
        pub_date_raw=year_text,
        authors=_authors(doc.find("AuthorList")),
        doi=ids.get("doi") or None,
        pmcid=ids.get("pmc") or None,
        publication_types=tuple(_text(el) for el in doc.findall("PublicationType")),
        mesh_terms=(),
        keywords=tuple(
            t for t in (_text(el) for el in doc.findall("KeywordList/Keyword")) if t
        ),
        language=_text(doc.find("Language")),
        source_type="book",
    )


def _text(element: ET.Element | None) -> str:
    """All descendant text, joined.

    Titles and abstracts carry inline markup — `<i>`, `<sup>`, `<sub>` — so `.text`
    alone silently truncates at the first tag.
    """
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _abstract(abstract: ET.Element | None) -> str:
    if abstract is None:
        return ""
    parts: list[str] = []
    for node in abstract.findall("AbstractText"):
        body = _text(node)
        if not body:
            continue
        # Structured abstracts label each section; keeping the labels preserves the
        # Methods/Results distinction that extraction depends on.
        label = node.get("Label")
        parts.append(f"{label.strip().title()}: {body}" if label else body)
    return "\n\n".join(parts)


def _authors(author_list: ET.Element | None) -> tuple[Author, ...]:
    if author_list is None:
        return ()
    authors: list[Author] = []
    for author in author_list.findall("Author"):
        # A study group ("The ARDS Network") occupies an author slot but has no
        # surname, so it is kept whole and flagged rather than dropped.
        collective = _text(author.find("CollectiveName"))
        if collective:
            authors.append(Author(surname=collective, collective=True))
            continue
        surname = _text(author.find("LastName"))
        if surname:
            authors.append(
                Author(surname=surname, initials=_text(author.find("Initials")))
            )
    return tuple(authors)


def _pub_date(journal: ET.Element | None) -> tuple[int | None, str]:
    """Publication year and the raw date string.

    PubDate is either Year/Month/Day or a free-text MedlineDate such as
    "2021 Jan-Feb" or "1998 Nov-Dec", so a plain `int(Year)` misses a real slice of
    older records.
    """
    if journal is None:
        return None, ""
    node = journal.find("JournalIssue/PubDate")
    if node is None:
        return None, ""

    year_text = _text(node.find("Year"))
    if year_text.isdigit():
        parts = [year_text, _text(node.find("Month")), _text(node.find("Day"))]
        return int(year_text), " ".join(p for p in parts if p)

    medline = _text(node.find("MedlineDate"))
    for token in medline.split():
        if len(token) == 4 and token.isdigit():
            return int(token), medline
    return None, medline


def _mesh(mesh_list: ET.Element | None) -> tuple[MeshTerm, ...]:
    if mesh_list is None:
        return ()
    terms: list[MeshTerm] = []
    for heading in mesh_list.findall("MeshHeading"):
        descriptor = heading.find("DescriptorName")
        if descriptor is None:
            continue
        qualifiers = heading.findall("QualifierName")
        terms.append(
            MeshTerm(
                descriptor=_text(descriptor),
                ui=descriptor.get("UI", ""),
                major=descriptor.get("MajorTopicYN") == "Y"
                or any(q.get("MajorTopicYN") == "Y" for q in qualifiers),
                qualifiers=tuple(_text(q) for q in qualifiers),
            )
        )
    return tuple(terms)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
