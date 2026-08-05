"""The four data-source clients, replayed against recorded fixtures.

Every assertion here is against a real recorded payload, so these tests fail if an API
changes shape rather than passing against a hand-written idea of one.

Re-record with `python scripts/record_fixtures.py --email you@example.org`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from okf_loremaster.clients import Clients, build_clients
from okf_loremaster.clients.bioc import NON_CONTENT_SECTIONS, is_unavailable, normalize_pmcid
from okf_loremaster.config import ConfigError
from okf_loremaster.events import EventBus

# Fixed ids, matching the recorder. See scripts/record_fixtures.py for why each is here.
RETRACTED = "9500320"
BOOK = "20301425"
NO_PMC = "33745404"
WITH_REFS = "30035690"
PMIDS = [RETRACTED, BOOK, NO_PMC]

TERM = "postoperative respiratory failure[tiab] AND risk factors[tiab]"
BAD_FIELD_TERM = "postoperative respiratory failure[nosuchfield]"


# --- esearch ---------------------------------------------------------------


async def test_esearch_returns_ids_and_count(replay_clients: Clients) -> None:
    result = await replay_clients.eutils.esearch(TERM, retmax=10)

    assert result.count > 0
    assert len(result.ids) == 10
    assert all(pmid.isdigit() for pmid in result.ids)
    assert result.truncated, "count exceeds retmax, so this page is not the whole set"


async def test_esearch_exposes_how_pubmed_read_the_query(replay_clients: Clients) -> None:
    result = await replay_clients.eutils.esearch(TERM, retmax=10)
    assert "Title/Abstract" in result.query_translation


async def test_an_unknown_field_tag_is_silently_rewritten(replay_clients: Clients) -> None:
    """The hazard the search node must defend against.

    PubMed does not reject `[nosuchfield]`. It rewrites the term to a free-text search,
    returns two orders of magnitude more hits than the real query, and reports an
    *empty* errorlist while doing it. Nothing but the translation reveals this, which
    is why `query_translation` is part of the result type rather than dropped.
    """
    good = await replay_clients.eutils.esearch(TERM, retmax=10)
    bad = await replay_clients.eutils.esearch(BAD_FIELD_TERM, retmax=5)

    assert bad.fields_not_found == (), "PubMed reports no error at all"
    assert "[All Fields]" in bad.query_translation
    assert bad.count > good.count * 10


# --- efetch ----------------------------------------------------------------


async def test_efetch_parses_a_journal_article(replay_clients: Clients) -> None:
    records = {r.pmid: r for r in await replay_clients.eutils.efetch(PMIDS)}
    record = records[NO_PMC]

    assert record.source_type == "journal"
    assert record.title.startswith("Effects of exercise modality")
    assert record.year == 2022
    assert record.journal
    assert record.has_abstract
    assert record.authors
    assert record.doi == "10.1080/09540121.2021.1902932"
    assert record.pmcid is None
    assert "Meta-Analysis" in record.publication_types
    assert record.mesh_terms
    # Compound surname: "Ferrari Silva", not "Ferrari".
    assert record.first_author_surname == "Ferrari Silva"
    assert record.authors[0].initials == "B"
    assert record.authors[0].display == "Ferrari Silva B"


async def test_efetch_returns_book_articles_too(replay_clients: Clients) -> None:
    """`PubmedBookArticle` is a sibling element, not a variant of `PubmedArticle`.

    A parser that looks only for the latter drops book chapters and returns fewer
    records than ids requested, with no error raised anywhere.
    """
    records = {r.pmid: r for r in await replay_clients.eutils.efetch(PMIDS)}

    assert len(records) == len(PMIDS)
    book = records[BOOK]
    assert book.source_type == "book"
    assert book.title
    assert book.journal, "the book title stands in for the journal"


async def test_retraction_is_detected(replay_clients: Clients) -> None:
    records = {r.pmid: r for r in await replay_clients.eutils.efetch(PMIDS)}

    assert records[RETRACTED].is_retracted
    assert not records[NO_PMC].is_retracted


async def test_ids_come_from_this_record_not_its_references(
    replay_clients: Clients,
) -> None:
    """The defect this test exists for.

    An unscoped `.//ArticleIdList/ArticleId` also matches every entry in the cited
    reference list — 19 matches instead of 4 on this record — and the last one wins.
    That silently yields some *cited* paper's PMC id, which would send the full-text
    fetch to the wrong article and file its text under this PMID.
    """
    records = {r.pmid: r for r in await replay_clients.eutils.efetch([WITH_REFS])}
    record = records[WITH_REFS]

    assert record.pmcid == "PMC6340782"
    assert record.doi == "10.1177/1750458918788978"


async def test_structured_abstract_keeps_its_labels(replay_clients: Clients) -> None:
    records = {r.pmid: r for r in await replay_clients.eutils.efetch([WITH_REFS])}
    abstract = records[WITH_REFS].abstract

    assert abstract
    # Labeled sections are what let extraction tell a method from a result.
    assert ":" in abstract


# --- BioC ------------------------------------------------------------------


async def test_bioc_returns_full_text_and_license(replay_clients: Clients) -> None:
    document = await replay_clients.bioc.fetch("PMC13424880")

    assert document is not None
    assert document.license == "CC BY", "license is recorded verbatim, never inferred"
    assert document.pmcid == "PMC13424880"
    assert len(document.sections) > 5
    assert document.word_count > 100


async def test_bioc_drops_non_evidence_sections(replay_clients: Clients) -> None:
    document = await replay_clients.bioc.fetch("PMC13424880")

    assert document is not None
    kept = {s.section_type for s in document.content_sections}
    assert kept, "something must survive"
    assert not (kept & NON_CONTENT_SECTIONS)
    assert "REF" in {s.section_type for s in document.sections}, (
        "the fixture must actually contain a reference section for this to prove anything"
    )


async def test_an_article_outside_the_oa_subset_returns_none(
    replay_clients: Clients,
) -> None:
    """BioC signals "not available" with HTTP 200 and an `[Error]` text body.

    `raise_for_status()` passes it straight through, and `json.loads` then fails a long
    way from the cause. Most of any corpus is not open access, so this is an ordinary
    outcome and must not raise.
    """
    assert await replay_clients.bioc.fetch("PMC99999999") is None


def test_unavailable_body_is_recognized() -> None:
    assert is_unavailable("[Error] : No result can be found.")
    assert is_unavailable("\n  [Error] : No result can be found.")
    assert not is_unavailable('[{"documents": []}]')


def test_pmcid_normalization() -> None:
    assert normalize_pmcid("PMC123") == "PMC123"
    assert normalize_pmcid("123") == "PMC123"
    assert normalize_pmcid("pmc123") == "PMC123"
    assert normalize_pmcid(" PMC123 ") == "PMC123"
    assert normalize_pmcid("not-an-id") is None


# --- PubTator --------------------------------------------------------------


async def test_pubtator_returns_annotations_keyed_by_pmid(
    replay_clients: Clients,
) -> None:
    """Keyed by the id in the payload: documents come back in arbitrary order."""
    annotated = await replay_clients.pubtator.annotate(PMIDS)

    assert set(annotated) <= set(PMIDS)
    document = annotated[BOOK]
    assert document.pmid == BOOK
    assert len(document.annotations) > 10
    assert "Disease" in document.concept_types
    assert all(a.text and a.concept_type for a in document.annotations)


async def test_pubtator_surface_forms_are_ranked_by_frequency(
    replay_clients: Clients,
) -> None:
    annotated = await replay_clients.pubtator.annotate(PMIDS)
    diseases = annotated[BOOK].by_type("Disease")

    assert diseases
    assert len(diseases) == len(set(diseases)), "distinct surface forms only"


# --- iCite -----------------------------------------------------------------


async def test_icite_returns_metrics(replay_clients: Clients) -> None:
    metrics = await replay_clients.icite.metrics([*PMIDS, WITH_REFS])

    record = metrics[NO_PMC]
    assert record.citation_count > 0
    assert record.relative_citation_ratio is not None
    assert record.year == 2022
    assert record.journal


async def test_rcr_defaults_to_field_average_when_unscored() -> None:
    """A paper too new to have an RCR must not rank below every scored paper."""
    from okf_loremaster.clients.icite import CitationMetrics

    unscored = CitationMetrics(
        pmid="1",
        citation_count=0,
        citations_per_year=0.0,
        relative_citation_ratio=None,
        nih_percentile=None,
        field_citation_rate=None,
        expected_citations_per_year=None,
        apt=None,
        is_clinical=False,
        is_research_article=True,
        cited_by_clinical=0,
        year=2026,
        journal="",
    )
    assert unscored.rcr_or_default == 1.0


async def test_a_highly_cited_paper_outranks_a_quiet_one(
    replay_clients: Clients,
) -> None:
    metrics = await replay_clients.icite.metrics([*PMIDS, WITH_REFS])
    assert metrics[RETRACTED].rcr_or_default > metrics[NO_PMC].rcr_or_default


# --- elink -------------------------------------------------------------------


def test_a_paper_with_no_citing_records_counts_zero_rather_than_going_missing() -> None:
    """Live elink omits `linksetdbs` entirely instead of sending an empty one.

    Every requested id comes back either way, so presence in the response carries no
    information and the count has to be read off the links themselves.
    """
    from okf_loremaster.clients.eutils import _parse_elink_counts

    body = json.dumps(
        {
            "linksets": [
                {
                    "dbfrom": "pubmed",
                    "ids": ["1"],
                    "linksetdbs": [
                        {"linkname": "pubmed_pubmed_citedin", "links": ["10", "11", "12"]}
                    ],
                },
                {"dbfrom": "pubmed", "ids": ["2"]},
            ]
        }
    )
    assert _parse_elink_counts(body) == {"1": 3, "2": 0}


def test_a_linkset_covering_several_papers_is_dropped_rather_than_divided() -> None:
    """The shape a comma-joined request comes back as, and it must never be believed.

    Its links are the union across every id in it, belonging to no single paper.
    Attributing them to one — or splitting them evenly — would invent a number, which
    is worse than losing one: nothing downstream could tell it from a measurement.
    """
    from okf_loremaster.clients.eutils import _parse_elink_counts

    body = json.dumps(
        {
            "linksets": [
                {
                    "ids": ["1", "2", "3"],
                    "linksetdbs": [
                        {"linkname": "pubmed_pubmed_citedin", "links": ["10", "11"]}
                    ],
                }
            ]
        }
    )
    assert _parse_elink_counts(body) == {}


async def test_one_oversized_chunk_costs_its_own_papers_and_no_others(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The failure a live run actually hit, and it took the whole corpus down with it.

    elink returns every citing PMID in full where only a count is wanted, so the body
    grows with how well-read the papers are — a large enough one is cut off mid-response
    and arrives as a transport error. One chunk of a corpus failing that way must not
    cost the counts of every chunk that succeeded.
    """
    from urllib.parse import parse_qs

    import httpx

    from okf_loremaster.clients import build_clients
    from okf_loremaster.clients.eutils import ELINK_BATCH

    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ids = [v for v in parse_qs(request.url.query.decode()).get("id", []) if v]
        seen.append(len(ids))
        if len(seen) == 1:  # the first chunk dies mid-body, as the live one did
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        return httpx.Response(
            200,
            json={
                "linksets": [
                    {"ids": [pmid], "linksetdbs": [
                        {"linkname": "pubmed_pubmed_citedin", "links": ["1", "2"]}
                    ]}
                    for pmid in ids
                ]
            },
        )

    settings = settings_factory(
        ncbi_email="test@example.org",
        http_cache_enabled=False,
        cache_dir=tmp_path / "cache",
        http_max_retries=1,
    )
    bus = EventBus()
    clients = build_clients(settings, bus=bus, transport=httpx.MockTransport(handler))
    try:
        pmids = [str(9_000_000 + i) for i in range(ELINK_BATCH * 2)]
        counts = await clients.eutils.cited_by_counts(pmids)
    finally:
        await clients.aclose()
        bus.close()

    assert len(seen) == 2, "chunked, and the second was still attempted"
    assert len(counts) == ELINK_BATCH, "the surviving chunk's papers are all measured"
    assert set(counts) == set(pmids[ELINK_BATCH:])
    assert all(value == 2 for value in counts.values())


def test_an_unparseable_elink_body_is_no_measurement_rather_than_a_corpus_of_zeroes() -> None:
    """Zero for every paper is the floor applied corpus-wide, which inverts the signal.

    The caller reads `{}` as "ask nothing of this" and leaves the citation component
    unmeasured, where a constant across the corpus changes no ordering at all.
    """
    from okf_loremaster.clients.eutils import _parse_elink_counts

    assert _parse_elink_counts("") == {}
    assert _parse_elink_counts("<html>an error page</html>") == {}
    assert _parse_elink_counts("[]") == {}


def test_a_batch_of_ids_never_builds_a_url_icite_refuses() -> None:
    """iCite answers an over-long URL with HTTP 413, and the old batch of 500 always
    built one: every run with a real pool ranked with no citation metrics, and said so
    in a single line that read like the service was down.

    Measured 2026-08-04: a 4,107-character URL answered, 4,217 did not.
    """
    from okf_loremaster.clients.icite import ID_BUDGET, batches

    chunks = list(batches([str(30000000 + i) for i in range(1310)]))

    assert all(len(",".join(chunk)) <= ID_BUDGET for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == 1310, "no id may be dropped"


def test_batching_is_by_length_because_pmids_are_not_all_one_length() -> None:
    """A count tuned for 8-digit ids overflows on 9-digit ones. The budget is on the
    string the ids actually build, so both pack to the same number of characters."""
    from okf_loremaster.clients.icite import batches

    short = list(batches([str(1000000 + i) for i in range(600)], budget=100))
    long = list(batches([str(100000000 + i) for i in range(600)], budget=100))

    assert all(len(",".join(c)) <= 100 for c in short + long)
    assert len(long) > len(short), "longer ids must pack fewer to a request"


def test_an_id_longer_than_the_whole_budget_is_still_sent() -> None:
    """Dropping it would lose a paper's metrics silently; looping forever would hang."""
    from okf_loremaster.clients.icite import batches

    assert list(batches(["1", "x" * 50, "2"], budget=10)) == [["1"], ["x" * 50], ["2"]]


# --- shared plumbing -------------------------------------------------------


async def test_ncbi_clients_share_one_rate_limiter(replay_clients: Clients) -> None:
    """The limit is enforced per IP across E-utilities, BioC and PubTator alike.

    A limiter per client would be three times the configured rate from NCBI's side.
    """
    assert replay_clients.eutils._http is replay_clients.bioc._http
    assert replay_clients.bioc._http is replay_clients.pubtator._http
    assert replay_clients.icite._http is not replay_clients.ncbi_http


async def test_stats_accumulate_across_hosts(replay_clients: Clients) -> None:
    await replay_clients.eutils.esearch(TERM, retmax=10)
    await replay_clients.icite.metrics([*PMIDS, WITH_REFS])

    assert replay_clients.stats.requests == 2
    assert replay_clients.ncbi_http.stats.requests == 1
    assert replay_clients.icite_http.stats.requests == 1


def test_a_ca_bundle_that_is_not_there_fails_before_the_first_request(
    settings_factory: Any, tmp_path: Path
) -> None:
    """Otherwise a typo in the path presents as every host being unreachable.

    Which is the failure this variable exists to explain, so getting it wrong must not
    produce the same symptom the right value cures.
    """
    settings = settings_factory(
        ncbi_email="test@example.org", ca_bundle=tmp_path / "not-here.pem"
    )
    with pytest.raises(ConfigError) as excinfo:
        build_clients(settings, bus=EventBus())
    assert "OKF_LOREMASTER_CA_BUNDLE" in str(excinfo.value)
