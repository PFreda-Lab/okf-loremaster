"""The reconcile node: turn extractions into records the emitter can write. No model call.

Four deterministic steps per paper, in this order:

1. **Length budgets** (`Extraction.enforce_budgets`) — before verification rather than
   after, so nothing is checked that a budget was about to drop, and so the verification
   counts describe what is actually in the bundle rather than what a model returned.
2. **Numeric verification** (`verification.verify_extraction`) — every number looked for
   in the text the extractor read. One that is not there is removed, its row's confidence
   is lowered, and the run continues. This is the whole reason the node exists.
3. **Vocabulary partition** (`partition_vocabulary`) — hints split against the charter's
   keys. What does not match goes to `unmapped_vocab`, which never reaches frontmatter
   and is what `validate` later aggregates into a concrete rerun command.
4. **Assembly** — bibliographic fields read from the PubMed record, license and text
   basis from the retrieval, provenance stamped here.

A paper with no extraction is dropped from its topic here rather than left as a topic
entry with no file behind it. That can take a topic under the floor curation worked to
meet, which is worth saying out loud: the shortfall is ours, not the literature's, and
the warning says so.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from okf_loremaster.config import ConfigError, Role
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.schemas import (
    Candidate,
    Charter,
    ConceptRecord,
    Extraction,
    PaperText,
    TextBasis,
    VerificationSummary,
    partition_vocabulary,
)
from okf_loremaster.verification import verify_extraction

__all__ = ["reconcile_node"]

NODE = "reconcile"

# How many stripped rows to name in the summary. Enough to see the shape of the problem,
# few enough that a run where everything failed says so in one line instead of two
# hundred. The full list is not lost: every record carries what survived.
MAX_EXAMPLES = 5


async def reconcile_node(state: RunState, deps: Deps) -> dict[str, Any]:
    charter = state.get("charter")
    if charter is None:
        raise RuntimeError("reconcile reached without a charter — the graph is wired wrong")

    topics = {slug: list(pmids) for slug, pmids in (state.get("topics") or {}).items()}
    texts = state.get("texts") or {}
    extractions = state.get("extractions") or {}
    by_pmid = {candidate.pmid: candidate for candidate in state.get("unique") or []}
    warnings = list(state.get("warnings") or [])

    with span(deps, NODE) as report:
        stamp = _provenance(deps)
        records: list[ConceptRecord] = []
        summary = VerificationSummary()
        dropped: list[str] = []
        trimmed_papers = 0
        unmapped_keys: set[str] = set()

        for slug, pmids in topics.items():
            kept: list[str] = []
            for pmid in pmids:
                extraction = extractions.get(pmid)
                candidate = by_pmid.get(pmid)
                if extraction is None or candidate is None:
                    dropped.append(pmid)
                    continue
                record, was_trimmed, unmapped = _reconcile_one(
                    charter,
                    slug,
                    candidate,
                    extraction,
                    texts.get(pmid),
                    summary,
                    stamp,
                )
                trimmed_papers += int(was_trimmed)
                unmapped_keys.update(unmapped)
                records.append(record)
                kept.append(pmid)
            topics[slug] = kept

        _report(
            deps,
            warnings,
            summary,
            dropped=dropped,
            trimmed=trimmed_papers,
            unmapped=sorted(unmapped_keys),
        )
        report["summary"] = f"{len(records)} record(s); {summary.line()}"

    return {
        "records": records,
        "topics": topics,
        "verification": summary,
        "warnings": warnings,
    }


def _reconcile_one(
    charter: Charter,
    slug: str,
    candidate: Candidate,
    extraction: Extraction,
    source: PaperText | None,
    summary: VerificationSummary,
    stamp: str,
) -> tuple[ConceptRecord, bool, list[str]]:
    trimmed, budget_notes = extraction.enforce_budgets()

    # No stored source means the paper was never retrieved, which the graph makes
    # impossible — but an unchecked extraction must not pass as a checked one, so an
    # empty source is used and every number in it fails. Loud, not silent.
    check = verify_extraction(trimmed, source.text if source is not None else "")

    summary.papers += 1
    summary.rows += len(check.rows)
    summary.effects_dropped += check.effects_dropped
    summary.intervals_dropped += check.intervals_dropped
    summary.quotes_dropped += check.quotes_dropped
    summary.sample_sizes_dropped += int(check.sample_size_missing)
    for note in check.notes():
        if len(summary.examples) < MAX_EXAMPLES:
            summary.examples.append(f"{candidate.pmid} {note}")

    recognized, unmapped = partition_vocabulary(
        check.extraction.vocabulary_hints, list(charter.vocabularies)
    )
    verified = check.extraction.model_copy(update={"vocabulary_hints": recognized})

    record = ConceptRecord(
        pmid=candidate.pmid,
        title=candidate.title,
        journal=candidate.journal,
        journal_abbrev=candidate.journal_abbrev,
        year=candidate.year,
        authors=list(candidate.authors),
        doi=candidate.doi,
        pmcid=(source.pmcid if source is not None else None) or candidate.pmcid,
        domain=slug,
        license=source.license if source is not None else "",
        text_basis=source.basis if source is not None else TextBasis.ABSTRACT,
        extraction=verified,
        unmapped_vocab=unmapped,
        generated_by=stamp,
        generated_at=datetime.now(UTC),
    )
    # Set after construction because the entries are derived from the identifiers the
    # record was just given.
    record.sources = record.default_sources()
    return record, bool(budget_notes), list(unmapped)


def _provenance(deps: Deps) -> str:
    """`generated.by`, naming the model that actually did the reading."""
    try:
        model = deps.settings.model_for(Role.REASONING) if deps.router is not None else ""
    except ConfigError:
        model = ""
    return f"okf-loremaster/extract/{model}" if model else "okf-loremaster/extract"


def _report(
    deps: Deps,
    warnings: list[str],
    summary: VerificationSummary,
    *,
    dropped: list[str],
    trimmed: int,
    unmapped: list[str],
) -> None:
    """One warning per category, never one per row."""
    if summary.effects_dropped:
        note = (
            f"numeric verification removed {summary.effects_dropped} effect size(s) not "
            f"found in the source text; those rows keep their claim at a lower confidence"
        )
        if summary.examples:
            note += " — " + "; ".join(summary.examples)
        warnings.append(note)
        deps.warn(NODE, note)

    if summary.intervals_dropped:
        note = (
            f"numeric verification removed {summary.intervals_dropped} confidence "
            f"interval(s) not found in the source text; the point estimates were kept"
        )
        warnings.append(note)
        deps.warn(NODE, note)

    if summary.quotes_dropped:
        note = (
            f"{summary.quotes_dropped} quoted sentence(s) were not in the source text "
            f"and were dropped"
        )
        warnings.append(note)
        deps.warn(NODE, note)

    if summary.sample_sizes_dropped:
        note = (
            f"{summary.sample_sizes_dropped} sample size(s) were not in the source text "
            f"and were dropped"
        )
        warnings.append(note)
        deps.warn(NODE, note)

    if trimmed:
        note = f"{trimmed} extraction(s) ran over a length budget and were trimmed"
        warnings.append(note)
        deps.warn(NODE, note)

    if unmapped:
        note = (
            f"extractions used {len(unmapped)} vocabulary key(s) the charter did not "
            f"list: {', '.join(unmapped)} — recorded as unmapped, not in frontmatter"
        )
        warnings.append(note)
        deps.warn(NODE, note)

    if dropped:
        note = (
            f"{len(dropped)} paper(s) had no usable extraction and were dropped from "
            f"their topic, so topic counts are below what curation kept: "
            f"{', '.join(dropped[:MAX_EXAMPLES])}"
        )
        warnings.append(note)
        deps.warn(NODE, note)
