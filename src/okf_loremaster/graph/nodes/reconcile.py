"""The reconcile node: turn extractions into records the emitter can write. No model call.

Five deterministic steps per paper, in this order:

1. **Length budgets** (`Extraction.enforce_budgets`) — before verification rather than
   after, so nothing is checked that a budget was about to drop, and so the verification
   counts describe what is actually in the bundle rather than what a model returned.
2. **Verification** (`verification.verify_extraction`) — every number, quoted sentence
   and vocabulary code looked for in the text the extractor read. What is not there is
   removed, the affected row's confidence is lowered, and the run continues. This is the
   whole reason the node exists.
3. **Mirroring** (`interactions.mirror_interactions`) — an interaction stated on one row
   given its other half on the row it names. After verification, never before: a mirror
   copies a coefficient, so mirroring first would duplicate every fabricated number and
   then check both halves against the same absent source.
4. **Evidence strength** (`strength.score_extraction`) — design, sample size, adjustment
   and reading depth, weighted into a score per paper and per row. After verification and
   not before: a row whose interval was just deleted for not appearing in the source text
   has lost the precision its score would otherwise have read off it.
5. **Assembly** — bibliographic fields read from the PubMed record, license and text
   basis from the retrieval, the abstract copied verbatim off the candidate, provenance
   stamped here.

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
from okf_loremaster.interactions import mirror_interactions
from okf_loremaster.schemas import (
    CODE,
    EFFECT,
    INTERACTION,
    INTERVAL,
    MAX_BODY_WORDS,
    QUOTE,
    Candidate,
    Charter,
    ConceptRecord,
    Extraction,
    PaperText,
    TextBasis,
    TextBasisPolicy,
    VerificationExample,
    VerificationSummary,
)
from okf_loremaster.strength import score_extraction
from okf_loremaster.verification import verify_extraction

__all__ = ["reconcile_node"]

NODE = "reconcile"

# How many stripped rows to name in the summary. Enough to see the shape of the problem,
# few enough that a run where everything failed says so in one line instead of two
# hundred. The full list is not lost: every record carries what survived.
MAX_EXAMPLES = 5


async def reconcile_node(state: RunState, deps: Deps) -> dict[str, Any]:

    topics = {slug: list(pmids) for slug, pmids in (state.get("topics") or {}).items()}
    texts = state.get("texts") or {}
    extractions = state.get("extractions") or {}
    by_pmid = {candidate.pmid: candidate for candidate in state.get("unique") or []}
    charter = state.get("charter")
    warnings = list(state.get("warnings") or [])

    with span(deps, NODE) as report:
        stamp = _provenance(deps)
        records: list[ConceptRecord] = []
        summary = VerificationSummary()
        dropped: list[str] = []
        # What was cut, named, and what merely ran long. Two lists because they are two
        # different facts about a paper and were one warning for as long as they shared
        # a list.
        cuts: list[str] = []
        overruns: list[int] = []

        for slug, pmids in topics.items():
            kept: list[str] = []
            for pmid in pmids:
                extraction = extractions.get(pmid)
                candidate = by_pmid.get(pmid)
                if extraction is None or candidate is None:
                    dropped.append(pmid)
                    continue
                record, notes, overrun = _reconcile_one(
                    slug,
                    candidate,
                    extraction,
                    texts.get(pmid),
                    summary,
                    stamp,
                    charter,
                )
                cuts += [f"{pmid} {note}" for note in notes]
                if overrun is not None:
                    overruns.append(overrun)
                records.append(record)
                kept.append(pmid)
            topics[slug] = kept

        _report(
            deps,
            warnings,
            summary,
            dropped=dropped,
            cuts=cuts,
            overruns=overruns,
        )
        report["summary"] = f"{len(records)} record(s); {summary.line()}"

    return {
        "records": records,
        "topics": topics,
        "verification": summary,
        "warnings": warnings,
    }


def _reconcile_one(
    slug: str,
    candidate: Candidate,
    extraction: Extraction,
    source: PaperText | None,
    summary: VerificationSummary,
    stamp: str,
    charter: Charter | None,
) -> tuple[ConceptRecord, list[str], int | None]:
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
    summary.codes_dropped += check.codes_dropped
    summary.sample_sizes_dropped += int(check.sample_size_missing)
    summary.interactions_dropped += check.interactions_dropped
    for kind, note in check.notes():
        # Per kind rather than overall. A global cap spent by whichever check failed
        # first leaves every warning after it counting casualties it cannot name.
        if len(summary.examples_for(kind)) < MAX_EXAMPLES:
            summary.examples.append(VerificationExample(kind=kind, note=f"{candidate.pmid} {note}"))

    basis = source.basis if source is not None else TextBasis.ABSTRACT
    reconciled = mirror_interactions(check.extraction)
    record = ConceptRecord(
        pmid=candidate.pmid,
        title=candidate.title,
        journal=candidate.journal,
        journal_abbrev=candidate.journal_abbrev,
        year=candidate.year,
        authors=list(candidate.authors),
        doi=candidate.doi,
        pmcid=(source.pmcid if source is not None else None) or candidate.pmcid,
        # Straight off the PubMed record, never off `source.text` — which for a full-text
        # paper is the whole article and for an abstract-only one is the same abstract
        # wrapped in the prompt's own header. This is the one field in the bundle that is
        # the publisher's words rather than ours, and it is copied rather than checked
        # because there is nothing to check it against.
        abstract=candidate.abstract,
        domain=slug,
        license=source.license if source is not None else "",
        text_basis=basis,
        strength=score_extraction(reconciled, charter=charter, basis=basis),
        extraction=reconciled,
        generated_by=stamp,
        generated_at=datetime.now(UTC),
    )
    # Set after construction because the entries are derived from the identifiers the
    # record was just given.
    record.sources = record.default_sources()
    # The notes themselves, not a boolean. `enforce_budgets` names the predictor rows and
    # null findings it drops, and collapsing that to "something was trimmed" threw away
    # the only record of what had left the bundle.
    # `reconciled`, not `check.extraction`: mirroring adds words to the body, and an
    # overrun measured before it would understate what actually gets written.
    return record, budget_notes, reconciled.body_overrun()


def _provenance(deps: Deps) -> str:
    """`generated.by`, naming the model that actually did the reading.

    The balanced tier, because that is the one `extract` calls. This said reasoning for
    as long as extraction ran there, and went on saying it after the node moved — so
    every file in a bundle credited its contents to a model that never saw the paper.
    Provenance nobody can check is provenance nobody should trust, so it is bound to
    the same `Role` the extract node passes to the router.
    """
    try:
        model = deps.settings.model_for(Role.BALANCED) if deps.router is not None else ""
    except ConfigError:
        model = ""
    return f"okf-loremaster/extract/{model}" if model else "okf-loremaster/extract"


def _report(
    deps: Deps,
    warnings: list[str],
    summary: VerificationSummary,
    *,
    dropped: list[str],
    cuts: list[str],
    overruns: list[int],
) -> None:
    """One warning per category, never one per row."""

    def emit(note: str, kind: str = "") -> None:
        """Append a warning, with the examples belonging to `kind` and no others.

        The examples used to be one untagged list attached to whichever warning came
        first, so the effect warning said it had removed one effect size and then named
        five dropped intervals — a count and a list that were about different checks.
        """
        examples = summary.examples_for(kind) if kind else []
        if examples:
            note += " — " + "; ".join(examples)
        warnings.append(note)
        deps.warn(NODE, note)

    if summary.effects_dropped:
        emit(
            f"numeric verification removed {summary.effects_dropped} effect size(s) not "
            f"found in the source text; those rows keep their claim at a lower confidence",
            EFFECT,
        )

    if summary.intervals_dropped:
        emit(
            f"numeric verification removed {summary.intervals_dropped} confidence "
            f"interval(s) not found in the source text; the point estimates were kept",
            INTERVAL,
        )

    if summary.quotes_dropped:
        emit(
            f"{summary.quotes_dropped} quoted sentence(s) were not in the source text "
            f"and were dropped",
            QUOTE,
        )

    if summary.interactions_dropped:
        emit(
            f"numeric verification removed {summary.interactions_dropped} interaction "
            f"value(s) not found in the source text; those interactions were kept and "
            f"now read as stated rather than measured",
            INTERACTION,
        )

    if summary.codes_dropped:
        emit(
            f"{summary.codes_dropped} vocabulary code(s) were not printed in the source "
            f"text and were dropped; the concepts they were attached to were kept",
            CODE,
        )

    if summary.sample_sizes_dropped:
        emit(
            f"{summary.sample_sizes_dropped} sample size(s) were not in the source text "
            f"and were dropped"
        )

    if cuts:
        note = f"{len(cuts)} length budget(s) cut something: " + "; ".join(cuts[:MAX_EXAMPLES])
        if len(cuts) > MAX_EXAMPLES:
            note += f"; and {len(cuts) - MAX_EXAMPLES} more"
        emit(note)

    if overruns:
        emit(
            f"{len(overruns)} extraction(s) are over the ~{MAX_BODY_WORDS} word body "
            f"guideline, the largest at ~{max(overruns)} words; nothing was cut, since "
            f"every field is already inside its own budget"
        )

    if dropped:
        emit(
            f"{len(dropped)} paper(s) had no usable extraction and were dropped from "
            f"their topic, so topic counts are below what curation kept: "
            f"{', '.join(dropped[:MAX_EXAMPLES])}"
        )

    if deps.basis is not TextBasisPolicy.ANY:
        # `strength` reads what a paper was read from as one of its axes, on the reasoning
        # that a finding taken from a full text is better established than the same
        # finding taken from an abstract. That is still true here — and under a uniform
        # basis it is the same term added to every paper in the bundle, so it separates
        # none of them from each other. The arithmetic is deliberately not adjusted: a
        # bundle's grades should mean the same thing as another bundle's, and rescaling
        # this axis to restore spread within the run is exactly how they would stop.
        emit(
            f"--basis {deps.basis.value} gave every paper the same text basis, so that "
            f"axis of `strength` contributes equally to all of them and does not separate "
            f"papers within this bundle; the grades remain comparable across bundles"
        )
