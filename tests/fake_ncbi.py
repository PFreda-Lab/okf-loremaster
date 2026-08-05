"""A synthetic PubMed, for testing the graph rather than the parsers.

The recorded cassette in `fixtures/ncbi.jsonl` pins real API shapes against a handful
of awkward records. It cannot drive the graph: `deterministic_plan` writes whatever
queries the charter implies, and a cassette only answers the exact URLs somebody once
recorded.

So this synthesizes a corpus with a known shape — four topics, each with a crowd of
near-identical titles, stacked by citation impact so that pure relevance rank leaves
the pool badly lopsided across them. That is what makes "MMR and the quota changed the
retained set" a measurable claim instead of an assertion that some numbers differ.

It still speaks real E-utilities JSON and real PubMed XML, so every client parser on
the path runs for real. Nothing here names a condition: the topics are `alpha`
through `delta`.

One more shape, for the conditional re-query edge: a slice of `delta` that no ordinary
query returns. Only a query carrying `UNLOCK_PHRASE` finds it, and that phrase can only
reach a query by way of a curator saying its topic is missing that topic. So "the second
round found papers the first could not" is a measurement here rather than an assumption
— and a re-query edge that silently re-ran the first round's searches would come back
with nothing new and fail the test.

It also serves BioC. Two fifths of the corpus is open access and the rest answers the
way BioC really answers for a paper outside the subset — HTTP 200 with a plain-text
`[Error]` body — so the trap `bioc.py` exists to contain is exercised on every run
rather than only in the one test that names it. Every open-access paper prints one
result sentence carrying real numbers, which is what makes an extraction's numbers
checkable and a fabricated one a contrast rather than the only case the suite sees.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlencode

import httpx

TOPICS = ("alpha", "beta", "gamma", "delta")
PER_TOPIC = 40

# The withheld slice. Numbered past `PER_TOPIC` so it is invisible to `all_pmids` and to
# every per-topic query — nothing that already passes has to change to accommodate it.
UNLOCK_PHRASE = "rescue cohort"
RESCUE_TOPIC = 3
RESCUE_COUNT = 12

# Records given a defect on purpose, so dedupe has something to count.
NO_ABSTRACT = {"10005", "10006"}
RETRACTED = {"10007"}
# 10008 is given 10009's title verbatim — a preprint and its journal version, which
# arrive under different PMIDs and identical titles.
DUPLICATE_TITLE = {"10008": "10009"}


def pmid_for(topic_index: int, n: int) -> str:
    return str(10000 + topic_index * 100 + n)


def topic_pmids(topic_index: int) -> list[str]:
    return [pmid_for(topic_index, n) for n in range(PER_TOPIC)]


def rescue_pmids() -> list[str]:
    """The withheld slice, in the order the unlocking query returns it."""
    return [pmid_for(RESCUE_TOPIC, n) for n in range(PER_TOPIC, PER_TOPIC + RESCUE_COUNT)]


def all_pmids() -> list[str]:
    """Every PMID, topic by topic.

    Deliberately *not* interleaved. This is the order the charter's base query returns,
    so a paper's rank in it is one more thing that varies by topic rather than at
    random — real corpora cluster, and a shuffled one would hide the clustering the
    quota exists to correct.
    """
    return [pmid for index in range(len(TOPICS)) for pmid in topic_pmids(index)]


def _owner(pmid: str) -> tuple[int, int]:
    number = int(pmid) - 10000
    return number // 100, number % 100


def title_for(pmid: str) -> str:
    topic_index, n = _owner(pmid)
    topic = TOPICS[topic_index]
    if pmid in DUPLICATE_TITLE:
        return title_for(DUPLICATE_TITLE[pmid])
    if n >= PER_TOPIC:
        # A withheld paper says in its title what the query that found it asked for,
        # which is how a screening policy tells one apart from the rest of its topic.
        return f"Rescue cohort {n}: {topic} exposure and measured outcome"
    # The first third of every topic shares a title skeleton, differing only in the
    # trailing number. Distinct enough that dedupe keeps them all — it matches on the
    # normalized title — and near-identical enough that MMR has something to thin out.
    if n < PER_TOPIC // 3:
        return f"Perioperative {topic} exposure and measured outcome in cohort {n}"
    return f"Study {n} of {topic} exposure, {topic} timing, and outcome variation"


def sample_size(pmid: str) -> int:
    """The analytic `n`, stated identically by the abstract and by the full text."""
    _, n = _owner(pmid)
    return 200 + n


def code_for(pmid: str) -> str:
    """The one diagnosis code this paper prints, in the section that would print one.

    Verification checks a code against the source the same way it checks a number, so the
    corpus needs papers that actually print one — a faithful extractor that could only
    ever report an empty `codes` list would exercise none of that path. Shaped like an
    ICD-10 code because that is what a methods section prints, and deterministic per pmid
    so the same paper always carries the same one.
    """
    topic_index, n = _owner(pmid)
    return f"{'EFGH'[topic_index % len(TOPICS)]}{10 + n % 90}.{n % 10}"


def snomed_for(pmid: str) -> str:
    """A second coding system, printed only in the full text.

    Two systems rather than one because nothing in the package filters hints by system,
    and a fixture printing a single system could not tell that apart from a fixture whose
    one system happened to be the only one allowed. Full text only, so an abstract-only
    paper carrying fewer codes than a full-text one is exercised as well.
    """
    _, n = _owner(pmid)
    return str(44000000 + n)


def abstract_for(pmid: str) -> str:
    topic_index, _ = _owner(pmid)
    if pmid in NO_ABSTRACT:
        return ""
    return (
        f"Background: we examined {TOPICS[topic_index]} exposure. "
        f"Methods: a cohort of {sample_size(pmid)} adults, identified by "
        f"ICD-10 {code_for(pmid)}. "
        f"Results: the association was estimated and reported."
    )


# --- full text --------------------------------------------------------------

# BioC's real answer for a paper outside the open-access subset: HTTP 200, and a body
# that is not JSON. `raise_for_status()` sails past it.
NOT_OPEN_ACCESS = "[Error] : No result can be found.\n"

# Which papers PMC serves in full. Deterministic rather than sampled, and two in five so
# that the fixture's reality and `llm.estimate.OPEN_ACCESS_RATE`'s assumption are the
# same claim rather than two guesses.
FULL_TEXT_IN = 2
FULL_TEXT_OF = 5

# Cycled so both answers `is_export_safe` can give actually occur, and so a run's
# exportable subset is a strict subset rather than everything or nothing.
LICENSES = ("CC BY", "CC BY-NC", "CC0", "NO-CC CODE")

# A number that appears only in the reference list — a section `content_sections` drops
# before the extractor ever sees it. An extraction claiming it is claiming something out
# of a paper nobody read, which is the difference between checking against the source
# and checking against whatever came back in the response body.
REFERENCE_ONLY = 8.77


def pmcid_for(pmid: str) -> str:
    return f"PMC{900000 + int(pmid)}"


def pmid_for_pmcid(pmcid: str) -> str | None:
    digits = pmcid.upper().removeprefix("PMC")
    if not digits.isdigit():
        return None
    pmid = int(digits) - 900000
    return str(pmid) if pmid >= 10000 else None


def has_full_text(pmid: str) -> bool:
    _, n = _owner(pmid)
    return n % FULL_TEXT_OF < FULL_TEXT_IN


def license_for(pmid: str) -> str:
    _, n = _owner(pmid)
    return LICENSES[n % len(LICENSES)]


def effect_for(pmid: str) -> float:
    topic_index, n = _owner(pmid)
    return round(1.1 + topic_index * 0.35 + (n % 9) * 0.05, 2)


def interval_for(pmid: str) -> tuple[float, float]:
    effect = effect_for(pmid)
    return round(effect - 0.28, 2), round(effect + 0.41, 2)


def methods_sentence(pmid: str) -> str:
    """The section carrying this paper's codes, on its own so a test can reuse it.

    A test that assembles a source by hand has to assemble the same one the graph would,
    or a code verification will fail against text nobody would have been shown.
    """
    topic_index, _ = _owner(pmid)
    return (
        f"We followed a cohort of {sample_size(pmid)} adults for one year, "
        f"measuring {TOPICS[topic_index]} exposure before the outcome window opened. "
        f"Cases were identified by ICD-10 {code_for(pmid)} "
        f"(SNOMED CT {snomed_for(pmid)})."
    )


def finding_sentence(pmid: str) -> str:
    """The one sentence carrying this paper's numbers, on one line and verbatim.

    Verification is only worth testing against a source that has something in it to
    find. This is that something: a real effect size with a real interval, written the
    way a results section writes one, so a quote of it can be matched and a number that
    is not in it can be caught.
    """
    low, high = interval_for(pmid)
    return (
        f"In adjusted models the association was {effect_for(pmid):.2f} "
        f"(95% CI {low:.2f}-{high:.2f}; p = 0.03)."
    )


def _passages(pmid: str) -> list[dict[str, Any]]:
    topic_index, _ = _owner(pmid)
    topic = TOPICS[topic_index]
    blocks = [
        ("TITLE", "front", title_for(pmid)),
        ("ABSTRACT", "abstract", abstract_for(pmid) or f"We examined {topic} exposure."),
        ("INTRO", "paragraph", f"Reports of {topic} exposure and this outcome disagree."),
        ("METHODS", "paragraph", methods_sentence(pmid)),
        ("RESULTS", "paragraph", finding_sentence(pmid)),
        ("TABLE", "table_caption", f"Table 1. {topic.title()} exposure by outcome status."),
        ("DISCUSS", "paragraph", f"The direction of the {topic} association was expected."),
        (
            "REF",
            "ref",
            f"1. Author00 A. An odds ratio of {REFERENCE_ONLY} in an earlier series. "
            f"J {topic.title()} Stud. 2019;8:1-9.",
        ),
    ]
    passages: list[dict[str, Any]] = []
    offset = 0
    for section_type, kind, text in blocks:
        passages.append(
            {
                "infons": {"section_type": section_type, "type": kind},
                "offset": offset,
                "text": text,
            }
        )
        offset += len(text) + 1
    return passages


def _bioc_body(pmid: str) -> str:
    return json.dumps(
        [
            {
                "source": "PMC",
                "documents": [
                    {
                        # The live service really does return "unknown" here, so the id
                        # the client asked for has to be the one it keeps.
                        "id": "unknown",
                        "infons": {"license": license_for(pmid)},
                        "passages": _passages(pmid),
                    }
                ],
            }
        ]
    )


# --- response builders ------------------------------------------------------


def _esearch_body(term: str, ids: list[str], retmax: int) -> str:
    returned = ids[:retmax]
    payload = {
        "header": {"type": "esearch", "version": "0.3"},
        "esearchresult": {
            "count": str(len(ids)),
            "retmax": str(len(returned)),
            "retstart": "0",
            "idlist": returned,
            "translationset": [],
            "querytranslation": term,
        },
    }
    return json.dumps(payload)


def _article_xml(pmid: str) -> str:
    topic_index, n = _owner(pmid)
    topic = TOPICS[topic_index]
    year = 2018 + (n % 7)
    retracted = (
        '<PublicationType UI="D016441">Retracted Publication</PublicationType>'
        if pmid in RETRACTED
        else ""
    )
    abstract = abstract_for(pmid)
    abstract_xml = (
        f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract>" if abstract else ""
    )
    return f"""<PubmedArticle>
  <MedlineCitation Status="MEDLINE">
    <PMID Version="1">{pmid}</PMID>
    <Article PubModel="Print">
      <Journal>
        <Title>Journal of {topic.title()} Studies</Title>
        <ISOAbbreviation>J {topic.title()} Stud</ISOAbbreviation>
        <JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue>
      </Journal>
      <ArticleTitle>{title_for(pmid)}</ArticleTitle>
      {abstract_xml}
      <AuthorList>
        <Author><LastName>Author{n:02d}</LastName><Initials>A</Initials></Author>
      </AuthorList>
      <Language>eng</Language>
      <PublicationTypeList>
        <PublicationType UI="D016428">Journal Article</PublicationType>
        {retracted}
      </PublicationTypeList>
    </Article>
    <MeshHeadingList>
      <MeshHeading>
        <DescriptorName UI="D000001" MajorTopicYN="Y">{topic.title()}</DescriptorName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName UI="D000002" MajorTopicYN="N">Cohort Studies</DescriptorName>
      </MeshHeading>
    </MeshHeadingList>
    <KeywordList Owner="NOTNLM">
      <Keyword MajorTopicYN="N">{topic}</Keyword>
    </KeywordList>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="pubmed">{pmid}</ArticleId>
      <ArticleId IdType="pmc">{pmcid_for(pmid)}</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>"""


def _efetch_body(pmids: list[str]) -> str:
    articles = "\n".join(_article_xml(pmid) for pmid in pmids)
    return f'<?xml version="1.0" ?><PubmedArticleSet>{articles}</PubmedArticleSet>'


def _rcr(topic_index: int, n: int) -> float:
    """Citation impact, stacked by topic.

    The one signal that is *not* uniform across topics, and the reason pure relevance
    rank buries three of the four: every other component — agreement, best rank, year,
    abstract, article type — is identical by construction, so this alone decides the
    ordering. `alpha` lands near the RCR ceiling, `delta` near the floor.
    """
    return round(2.9 - topic_index * 0.8 + (n % 5) * 0.02, 2)


def _icite_body(pmids: list[str]) -> str:
    rows = []
    for pmid in pmids:
        topic_index, n = _owner(pmid)
        rows.append(
            {
                "pmid": int(pmid),
                "year": 2018 + (n % 7),
                "journal": f"J {TOPICS[topic_index].title()} Stud",
                "citation_count": (n * 3) % 90,
                "citations_per_year": 2.0,
                "relative_citation_ratio": _rcr(topic_index, n),
                "nih_percentile": 50.0,
                "field_citation_rate": 4.0,
                "expected_citations_per_year": 2.0,
                "apt": 0.25,
                "is_clinical": False,
                "is_research_article": True,
                "cited_by_clin": [],
            }
        )
    return json.dumps({"data": rows})


def elink_count(pmid: str) -> int:
    """PubMed's cited-by count for a paper in the fake corpus.

    Deliberately unequal to `_icite_body`'s `citation_count` for the same paper, so a
    test can tell which service a ranking actually used rather than assuming.
    """
    _, n = _owner(pmid)
    return (n * 2) % 40


def _elink_body(pmids: list[str]) -> str:
    """One linkset per id, `linksetdbs` omitted where the count is zero.

    The omission is what the live service does, and it is the case worth reproducing:
    an uncited paper is not absent from the response, it is present with nothing on it.
    """
    linksets = []
    for pmid in pmids:
        linkset: dict[str, Any] = {"dbfrom": "pubmed", "ids": [pmid]}
        count = elink_count(pmid)
        if count:
            linkset["linksetdbs"] = [
                {
                    "dbto": "pubmed",
                    "linkname": "pubmed_pubmed_citedin",
                    "links": [str(900000 + i) for i in range(count)],
                }
            ]
        linksets.append(linkset)
    return json.dumps({"linksets": linksets})


# --- the transport ----------------------------------------------------------


BIOC_PATH = "/BioC_json/"


class FakeNCBI:
    """An `httpx` transport standing in for E-utilities, BioC and iCite.

    Counts what it was asked for, so a test can assert on request shape — that one
    `efetch` covered the whole plan, for instance, rather than one per query.
    """

    def __init__(self, *, finds_nothing: bool = False, icite_fails: bool = False) -> None:
        self.esearch_terms: list[str] = []
        self.efetch_batches: list[list[str]] = []
        self.icite_batches: list[list[str]] = []
        self.elink_batches: list[list[str]] = []
        self.bioc_requests: list[str] = []
        # PubMed answers a query that matches nothing with a perfectly successful
        # search of zero results, which is how a whole run once reached the end with
        # an empty pool and reported a valid bundle.
        self.finds_nothing = finds_nothing
        # iCite is a separate host and fails independently — on some networks it fails
        # every time while E-utilities answers normally throughout.
        self.icite_fails = icite_fails

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.url.query.decode())
        params = {key: values[-1] for key, values in query.items()}
        path = request.url.path

        if path.endswith("elink.fcgi"):
            # Every `id`, not the last one. elink is asked with the key repeated, and
            # collapsing that here would test a request the client never sends.
            pmids = [p for p in query.get("id", []) if p]
            self.elink_batches.append(pmids)
            return httpx.Response(200, text=_elink_body(pmids))

        if path.endswith("esearch.fcgi"):
            term = params.get("term", "")
            self.esearch_terms.append(term)
            ids = self._ids_for(term)
            retmax = int(params.get("retmax", "200"))
            return httpx.Response(200, text=_esearch_body(term, ids, retmax))

        if path.endswith("efetch.fcgi"):
            pmids = [p for p in params.get("id", "").split(",") if p]
            self.efetch_batches.append(pmids)
            return httpx.Response(200, text=_efetch_body(pmids))

        if BIOC_PATH in path:
            pmcid = path.split(BIOC_PATH)[-1].split("/")[0]
            self.bioc_requests.append(pmcid)
            pmid = pmid_for_pmcid(pmcid)
            if pmid is None or not has_full_text(pmid):
                return httpx.Response(200, text=NOT_OPEN_ACCESS)
            return httpx.Response(200, text=_bioc_body(pmid))

        if "icite" in (request.url.host or ""):
            pmids = [p for p in params.get("pmids", "").split(",") if p]
            self.icite_batches.append(pmids)
            if self.icite_fails:
                raise httpx.ConnectError("iCite is unreachable", request=request)
            return httpx.Response(200, text=_icite_body(pmids))

        return httpx.Response(404, text=f"unexpected request: {request.url}")

    def _ids_for(self, term: str) -> list[str]:
        """Which corpus slice a query matches.

        A query naming exactly one topic gets that topic; anything else — the charter's
        outcome-and-population base query — gets everything, in topic order. The
        withheld slice answers to `UNLOCK_PHRASE` and to nothing else, so a round that
        asks the same questions again retrieves the same corpus again.
        """
        if self.finds_nothing:
            return []
        if UNLOCK_PHRASE in term.lower():
            return rescue_pmids()
        named = [index for index, topic in enumerate(TOPICS) if f'"{topic}"[tiab]' in term]
        if len(named) == 1:
            return topic_pmids(named[0])
        return all_pmids()


def url_with(base: str, **params: str) -> str:
    """Small helper for tests that need to build a request URL by hand."""
    return f"{base}?{urlencode(params)}"


_IDENTITY = re.compile(r"we examined (\w+) exposure.*?cohort of (\d+) adults", re.DOTALL)


def identify(text: str) -> tuple[str, int]:
    """`(topic, n)` for a paper, read back out of the text a screener would see.

    The screener is shown a title and an abstract and nothing else — no PMID — so a
    scripted screening policy has to recognize a paper the same way a real one would.
    Reading it back out of the abstract keeps that honest; being handed the PMID would
    let a policy key on something the model never sees.
    """
    match = _IDENTITY.search(text)
    if match is None:
        raise AssertionError(f"not a paper from this corpus: {text[:80]!r}")
    return match.group(1), int(match.group(2)) - 200


def is_rescue(n: int) -> bool:
    return n >= PER_TOPIC
