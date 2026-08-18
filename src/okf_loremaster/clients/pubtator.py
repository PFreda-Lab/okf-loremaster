"""PubTator3 concept annotations.

Supplies normalized biomedical entities — genes, diseases, chemicals, species — with
stable identifiers (MeSH, NCBI Gene). Used as a deterministic vocabulary signal so the
extraction agent is not the only source of controlled terms.

Note this is a *hint* source. What ends up in `vocabulary_hints` is what the extractor
read in the paper, not whatever PubTator happened to tag.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from okf_loremaster.clients._http import HttpClient

BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"

# The service accepts up to 100 ids per request.
BATCH = 100


@dataclass(frozen=True, slots=True)
class Annotation:
    concept_type: str
    text: str
    identifier: str


@dataclass(frozen=True, slots=True)
class AnnotatedDocument:
    pmid: str
    pmcid: str | None
    annotations: tuple[Annotation, ...]

    def by_type(self, concept_type: str) -> tuple[str, ...]:
        """Distinct surface forms of one concept type, most frequent first."""
        counts = Counter(a.text for a in self.annotations if a.concept_type == concept_type)
        return tuple(text for text, _ in counts.most_common())

    @property
    def concept_types(self) -> tuple[str, ...]:
        return tuple(sorted({a.concept_type for a in self.annotations}))


class PubTatorClient:
    def __init__(self, http: HttpClient) -> None:
        self._http = http

    async def annotate(
        self, pmids: Sequence[str], *, node: str = "extract"
    ) -> dict[str, AnnotatedDocument]:
        """Annotations keyed by PMID. Missing ids are simply absent from the result."""
        out: dict[str, AnnotatedDocument] = {}
        ids = [str(p) for p in pmids]
        for start in range(0, len(ids), BATCH):
            chunk = ids[start : start + BATCH]
            raw = await self._http.get_text(BASE, params={"pmids": ",".join(chunk)}, node=node)
            out.update(parse_pubtator(raw))
        return out


def parse_pubtator(raw: str) -> dict[str, AnnotatedDocument]:
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}

    # The response is wrapped in a `PubTator3` envelope, and documents come back in
    # arbitrary order relative to the requested ids (verified 2026-08-03) — so results
    # are keyed by the id in the payload, never zipped against the request.
    documents = payload.get("PubTator3") if isinstance(payload, dict) else payload
    if not isinstance(documents, list):
        return {}

    out: dict[str, AnnotatedDocument] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        pmid = str(document.get("pmid") or document.get("id") or "").strip()
        if not pmid:
            continue

        annotations: list[Annotation] = []
        pmcid: str | None = _clean(document.get("pmcid"))
        for passage in document.get("passages", []):
            infons = passage.get("infons") or {}
            pmcid = pmcid or _clean(infons.get("article-id_pmc"))
            for annotation in passage.get("annotations", []):
                ann_infons = annotation.get("infons") or {}
                concept_type = str(ann_infons.get("type", "")).strip()
                text = str(annotation.get("text", "")).strip()
                if concept_type and text:
                    annotations.append(
                        Annotation(
                            concept_type=concept_type,
                            text=text,
                            identifier=str(ann_infons.get("identifier") or "").strip(),
                        )
                    )
        out[pmid] = AnnotatedDocument(pmid=pmid, pmcid=pmcid, annotations=tuple(annotations))
    return out


def _clean(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"none", "null"} else None
