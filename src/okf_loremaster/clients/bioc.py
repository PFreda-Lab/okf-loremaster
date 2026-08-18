"""PMC full text via the BioC REST service.

Why BioC and not `oa.fcgi`: BioC returns full-text JSON per article with the license in
`infons.license`, which is exactly what we need and what we must record. `oa.fcgi` only
ever returned package locations for FTP or cloud download — it still answers HTTP 200
today (verified 2026-08-03), so choosing BioC is a design decision, not a workaround.
Do not "fix" this back. Web pages are never scraped.

The trap this module exists to contain: **BioC signals "not available" with HTTP 200
and a plain-text `[Error] : No result can be found.` body** (verified 2026-08-03).
`raise_for_status()` sails straight past it and `json.loads` then fails with a decode
error a long way from the cause. Availability is therefore checked on the body, and a
missing article returns `None` rather than raising — most of any corpus is not in the
open-access subset, so that is an ordinary outcome, not a failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from okf_loremaster.clients._http import HttpClient

BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"

# Sections that are not evidence: references, funding statements, author
# contributions, competing interests, supplementary pointers. Including them inflates
# the extraction prompt and invites the model to cite a reference list entry as a
# finding.
NON_CONTENT_SECTIONS = frozenset(
    {
        "REF",
        "ACK_FUND",
        "AUTH_CONT",
        "COMP_INT",
        "SUPPL",
        "APPENDIX",
        "ABBR",
        "REVIEW_INFO",
    }
)

_PMCID = re.compile(r"^(?:PMC)?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BioCSection:
    section_type: str
    passage_type: str
    text: str
    offset: int


@dataclass(frozen=True, slots=True)
class BioCDocument:
    pmcid: str
    license: str
    sections: tuple[BioCSection, ...]

    @property
    def content_sections(self) -> tuple[BioCSection, ...]:
        return tuple(s for s in self.sections if s.section_type not in NON_CONTENT_SECTIONS)

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.content_sections)

    def body_text(self, *, include: frozenset[str] | None = None) -> str:
        """Content sections joined with their headings, for an extraction prompt."""
        chosen = [s for s in self.content_sections if include is None or s.section_type in include]
        return "\n\n".join(f"## {s.section_type}\n{s.text}" for s in chosen if s.text)


class BioCClient:
    def __init__(self, http: HttpClient) -> None:
        self._http = http

    async def fetch(self, pmcid: str, *, node: str = "fulltext") -> BioCDocument | None:
        """Full text for a PMC id, or `None` when it is not in the open-access subset."""
        normalized = normalize_pmcid(pmcid)
        if normalized is None:
            return None
        raw = await self._http.get_text(f"{BASE}/{normalized}/unicode", node=node)
        return parse_bioc(raw, normalized)


def normalize_pmcid(pmcid: str) -> str | None:
    match = _PMCID.match(pmcid.strip())
    return f"PMC{match.group(1)}" if match else None


def is_unavailable(raw: str) -> bool:
    """Whether a 200 response body is actually BioC's not-found message."""
    return raw.lstrip().startswith("[Error]")


def parse_bioc(raw: str, pmcid: str) -> BioCDocument | None:
    if is_unavailable(raw):
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        # An unrecognized body shape means no full text, not a crashed run.
        return None

    # The service returns a list holding one collection; tolerate either form.
    collection = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(collection, dict):
        return None
    documents = collection.get("documents") or []
    if not documents:
        return None
    document = documents[0]

    sections: list[BioCSection] = []
    for passage in document.get("passages", []):
        text = (passage.get("text") or "").strip()
        if not text:
            continue
        infons = passage.get("infons") or {}
        sections.append(
            BioCSection(
                section_type=str(infons.get("section_type", "")).upper(),
                passage_type=str(infons.get("type", "")),
                text=text,
                offset=int(passage.get("offset", 0) or 0),
            )
        )

    return BioCDocument(
        # The document's own `id` is frequently the literal string "unknown"
        # (verified 2026-08-03), so the requested id is authoritative.
        pmcid=pmcid,
        license=str((document.get("infons") or {}).get("license", "")).strip(),
        sections=tuple(sections),
    )
