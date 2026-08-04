"""iCite citation metrics.

Feeds the deterministic ranker. The metric that matters is the Relative Citation Ratio:
citations normalized against the field's own citation rate and benchmarked to NIH-funded
work, so a well-cited paper in a small field is not buried under a mediocre one in a
large field. A raw citation count would systematically favor whichever specialty
happens to publish most.

`apt` (Approximate Potential to Translate) and `is_clinical` are carried through because
they discriminate translational relevance, which raw counts do not.

Different host from NCBI, therefore a different rate limiter — iCite traffic must not
consume the E-utilities budget.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from okf_loremaster.clients._http import HttpClient

BASE = "https://icite.od.nih.gov/api/pubs"

# iCite accepts large id lists; 500 keeps URLs comfortably short.
BATCH = 500


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    pmid: str
    citation_count: int
    citations_per_year: float
    relative_citation_ratio: float | None
    nih_percentile: float | None
    field_citation_rate: float | None
    expected_citations_per_year: float | None
    apt: float | None
    is_clinical: bool
    is_research_article: bool
    cited_by_clinical: int
    year: int | None
    journal: str

    @property
    def rcr_or_default(self) -> float:
        """RCR with a neutral stand-in.

        iCite leaves RCR null for papers too recent to have one — roughly the last two
        years, which is exactly the slice a literature review most wants. Treating that
        as zero would rank new work below everything; 1.0 means "field average", which
        is the honest prior for a paper with no evidence either way.
        """
        return self.relative_citation_ratio if self.relative_citation_ratio is not None else 1.0


class ICiteClient:
    def __init__(self, http: HttpClient) -> None:
        self._http = http

    async def metrics(
        self, pmids: Sequence[str], *, node: str = "rank"
    ) -> dict[str, CitationMetrics]:
        out: dict[str, CitationMetrics] = {}
        ids = [str(p) for p in pmids]
        for start in range(0, len(ids), BATCH):
            chunk = ids[start : start + BATCH]
            raw = await self._http.get_text(
                BASE, params={"pmids": ",".join(chunk)}, node=node
            )
            out.update(parse_icite(raw))
        return out


def parse_icite(raw: str) -> dict[str, CitationMetrics]:
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}

    out: dict[str, CitationMetrics] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pmid = str(row.get("pmid") or row.get("_id") or "").strip()
        if not pmid:
            continue
        out[pmid] = CitationMetrics(
            pmid=pmid,
            citation_count=_int(row.get("citation_count")),
            citations_per_year=_float(row.get("citations_per_year")) or 0.0,
            relative_citation_ratio=_float(row.get("relative_citation_ratio")),
            nih_percentile=_float(row.get("nih_percentile")),
            field_citation_rate=_float(row.get("field_citation_rate")),
            expected_citations_per_year=_float(row.get("expected_citations_per_year")),
            apt=_float(row.get("apt")),
            is_clinical=bool(row.get("is_clinical")),
            is_research_article=bool(row.get("is_research_article")),
            cited_by_clinical=len(row.get("cited_by_clin") or []),
            year=_int(row.get("year")) or None,
            journal=str(row.get("journal") or "").strip(),
        )
    return out


def _int(value: object) -> int:
    parsed = _float(value)
    return int(parsed) if parsed is not None else 0


def _float(value: object) -> float | None:
    """Coerce a JSON scalar to a float, or `None` when it is not one.

    iCite returns `null` for metrics it has not computed and occasionally a numeric
    string, so neither a bare `float()` nor a type check alone covers the input.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
