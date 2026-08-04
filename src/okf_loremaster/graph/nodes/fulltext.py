"""The fulltext node: assemble exactly what each extraction will read. No model call.

Every retained paper leaves this node with a `PaperText`, and every `PaperText` holds
the finished prompt block rather than raw sections. Two consequences follow, and both
are the point:

- **The length budget is applied here, not in `extract`.** `verification` checks the
  model's numbers against `PaperText.text`, so that string has to be the one the model
  saw. Truncating downstream of storage, or storing raw sections and rebuilding the
  prompt per call, would let the two drift and correct extractions would start reporting
  as fabricated.
- **The checkpoint stays a sane size.** LangGraph re-serializes state after every
  super-step, and a shelf of 200 unbudgeted full texts is tens of megabytes rewritten
  repeatedly.

Most of any corpus is not open access, so an abstract-only source is an ordinary
outcome, not a failure. It is recorded as `text_basis` so a downstream agent can see
which claims rest on a full reading, and `license` stays empty for those papers because
none was ever served to us.
"""

from __future__ import annotations

import asyncio
from typing import Any

from okf_loremaster.clients.bioc import BioCDocument, BioCSection
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.schemas import MAX_SOURCE_CHARS, Candidate, PaperText, TextBasis

__all__ = ["fulltext_node"]

NODE = "fulltext"

# Which sections survive the budget, best first. The order is about where the evidence
# is: the abstract states the findings, results carries the numbers those findings rest
# on, and table and figure captions carry the numbers that never reached the prose.
# Discussion and introduction go last because they mostly restate other papers, which is
# also the fastest way to get a number from a different study attributed to this one.
SECTION_PRIORITY = (
    "TITLE",
    "ABSTRACT",
    "RESULTS",
    "TABLE",
    "FIG",
    "CONCL",
    "METHODS",
    "DISCUSS",
    "INTRO",
)


async def fulltext_node(state: RunState, deps: Deps) -> dict[str, Any]:
    shelves = state.get("shelves") or {}
    by_pmid = {candidate.pmid: candidate for candidate in state.get("unique") or []}
    warnings = list(state.get("warnings") or [])
    # A resumed or re-queried run keeps what it already fetched. Retrieval is the slow
    # part of this node and the answer does not change between rounds.
    texts = dict(state.get("texts") or {})

    with span(deps, NODE) as report:
        todo = [
            by_pmid[pmid]
            for pmids in shelves.values()
            for pmid in pmids
            if pmid not in texts and pmid in by_pmid
        ]
        failures: list[str] = []
        fetched = await _fetch_all(deps, todo, failures)
        for source in fetched:
            texts[source.pmid] = source

        if failures:
            note = (
                f"{len(failures)} full-text request(s) failed; those papers fall back to "
                f"their abstract — first: {failures[0]}"
            )
            warnings.append(note)
            deps.warn(NODE, note)

        full = sum(1 for source in texts.values() if source.is_full_text)
        truncated = sum(1 for source in texts.values() if source.truncated)
        report["summary"] = (
            f"{len(texts)} paper(s): {full} full text, {len(texts) - full} abstract only"
            + (f", {truncated} truncated to the prompt budget" if truncated else "")
        )

    return {"texts": texts, "warnings": warnings}


async def _fetch_all(
    deps: Deps, candidates: list[Candidate], failures: list[str]
) -> list[PaperText]:
    """Every paper at once. The shared NCBI rate limiter decides how many are in flight."""
    if not candidates:
        return []
    total = len(candidates)
    done = 0

    async def one(candidate: Candidate) -> PaperText:
        nonlocal done
        source = await _source_for(deps, candidate, failures)
        done += 1
        deps.progress(NODE, f"retrieved {done} of {total}", current=done, total=total)
        return source

    return list(await asyncio.gather(*(one(candidate) for candidate in candidates)))


async def _source_for(deps: Deps, candidate: Candidate, failures: list[str]) -> PaperText:
    document: BioCDocument | None = None
    if candidate.pmcid:
        try:
            document = await deps.clients.bioc.fetch(candidate.pmcid, node=NODE)
        except Exception as exc:
            # An abstract is a worse source than a full text and a better one than
            # nothing, so a failed request costs this paper depth, not its place.
            failures.append(f"{candidate.pmid}: {type(exc).__name__}: {exc}")

    if document is not None:
        source = _from_document(candidate, document)
        if source is not None:
            return source
    return _from_abstract(candidate)


def _from_document(candidate: Candidate, document: BioCDocument) -> PaperText | None:
    """The budgeted prompt block for a paper we hold full text for.

    `None` when nothing fit — a pathological document whose every section is larger than
    the whole budget — which sends the paper down the abstract path instead of into an
    extraction call with an empty source.
    """
    kept, truncated = _select(document.content_sections, MAX_SOURCE_CHARS)
    if not kept:
        return None
    body = "\n\n".join(f"## {section.section_type or 'TEXT'}\n{section.text}" for section in kept)
    return PaperText(
        pmid=candidate.pmid,
        basis=TextBasis.FULL_TEXT,
        license=document.license,
        pmcid=document.pmcid,
        text=_block(candidate, "full text (PMC)", document.license, body),
        sections=[section.section_type or "TEXT" for section in kept],
        source_chars=sum(len(section.text) for section in document.content_sections),
        truncated=truncated,
    )


def _from_abstract(candidate: Candidate) -> PaperText:
    body = candidate.abstract.strip() or "(no abstract available)"
    return PaperText(
        pmid=candidate.pmid,
        basis=TextBasis.ABSTRACT,
        # Deliberately empty. PubMed serves no license with an abstract, and an inferred
        # one is how a bundle becomes undistributable without anyone noticing.
        license="",
        pmcid=candidate.pmcid or "",
        text=_block(candidate, "abstract only", "", f"## ABSTRACT\n{body}"),
        source_chars=len(body),
    )


def _block(candidate: Candidate, basis: str, license_: str, body: str) -> str:
    """The exact text the extractor is shown, header and all.

    The header travels inside the stored text rather than being added by the extract
    node, so that what verification checks against and what the model read are one
    string with no assembly step between them.
    """
    where = candidate.journal_abbrev or candidate.journal
    citation = " ".join(part for part in (where, str(candidate.year or "")) if part)
    lines = [f"Title: {candidate.title or '(untitled)'}"]
    if citation:
        lines.append(f"Published in: {citation}")
    lines.append(f"Read from: {basis}" + (f", license {license_}" if license_ else ""))
    return "\n".join(lines) + "\n\n" + body


def _select(
    sections: tuple[BioCSection, ...], budget: int
) -> tuple[list[BioCSection], bool]:
    """Fit sections into the budget by priority, then restore reading order.

    A section that does not fit is skipped rather than ending the loop, so one enormous
    methods section cannot cost a paper its conclusions. Reading order is restored
    afterward because a prompt whose sections arrive shuffled by priority reads as a
    different paper than the one that was published.
    """
    ranked = sorted(enumerate(sections), key=lambda pair: (_priority(pair[1]), pair[0]))
    kept: list[tuple[int, BioCSection]] = []
    spent = 0
    truncated = False
    for index, section in ranked:
        if not section.text:
            continue
        cost = len(section.text) + len(section.section_type) + 6
        if spent + cost > budget:
            truncated = True
            continue
        kept.append((index, section))
        spent += cost
    return [section for _, section in sorted(kept, key=lambda pair: pair[0])], truncated


def _priority(section: BioCSection) -> int:
    kind = section.section_type.upper()
    if kind in SECTION_PRIORITY:
        return SECTION_PRIORITY.index(kind)
    # A caption's `section_type` is the section it sits in, so the passage type is what
    # identifies it. Ranked with its own kind rather than as an unknown.
    for index, name in enumerate(SECTION_PRIORITY):
        if name.lower() in section.passage_type.lower():
            return index
    return len(SECTION_PRIORITY)
