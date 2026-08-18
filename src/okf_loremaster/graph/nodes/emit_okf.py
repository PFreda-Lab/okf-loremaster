"""The emit node: assemble the run manifest and write the bundle. No model call.

Two things happen here and they are separable on purpose. `_manifest` turns the run
state into the one object that answers "what is this bundle and can I still trust it" —
which prompt, which charter, which models, how many papers survived each stage, what it
cost, when it goes stale. `write_bundle` then renders it, and knows nothing about
`RunState`. That is what lets the emitter be tested against a handful of records instead
of a whole run, and what will let build step 8 walk a finished bundle without ever
touching the state that produced it.

The bundle directory is decided before the graph starts and carried on `Deps`. Deciding
it here would mean a resumed run could write somewhere else than the run it resumed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from okf_loremaster import __version__
from okf_loremaster.clients.eutils import ESEARCH_SORT
from okf_loremaster.config import ConfigError, Role
from okf_loremaster.emitters.okf import log_markdown, write_bundle
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.schemas import (
    BundleCounts,
    Charter,
    ConceptRecord,
    CostSummary,
    RunManifest,
    TextBasis,
    TopicSummary,
)

__all__ = ["emit_okf_node", "manifest_for"]

NODE = "emit_okf"


async def emit_okf_node(state: RunState, deps: Deps) -> dict[str, Any]:
    charter = state.get("charter")
    if charter is None:
        raise RuntimeError("emit_okf reached without a charter — the graph is wired wrong")

    records: list[ConceptRecord] = list(state.get("records") or [])
    warnings = list(state.get("warnings") or [])
    directory = _directory(state, deps)

    with span(deps, NODE) as report:
        manifest = manifest_for(state, deps, charter=charter, records=records)
        verification = state.get("verification")
        result = write_bundle(
            directory,
            records=records,
            charter=charter,
            manifest=manifest,
            log=log_markdown(
                charter,
                manifest,
                verification=verification.line() if verification is not None else "",
            ),
        )
        for warning in result.warnings:
            warnings.append(warning)
            deps.warn(NODE, warning)
        report["summary"] = (
            f"{result.documents} document(s) across {result.topics} topic/topics "
            f"-> {directory}"
        )

    return {
        "bundle": str(directory),
        "manifest": manifest,
        "warnings": warnings,
    }


def _directory(state: RunState, deps: Deps) -> Path:
    if deps.bundle_dir is not None:
        return deps.bundle_dir
    return deps.settings.output_dir / str(state.get("run_id") or "bundle")


def manifest_for(
    state: RunState,
    deps: Deps,
    *,
    charter: Charter,
    records: list[ConceptRecord],
) -> RunManifest:
    """Everything the bundle has to be able to say about itself later."""
    finished = datetime.now(UTC)
    started = state.get("started_at")
    manifest = RunManifest(
        run_id=str(state.get("run_id") or ""),
        prompt=str(state.get("prompt") or charter.prompt),
        charter_digest=charter.digest(),
        tool_version=f"okf-loremaster/{__version__}",
        started_at=started,
        finished_at=finished,
        models=_models(deps),
        counts=_counts(state, records),
        topics=_topics(charter, records),
        queries=list(state.get("executed") or []),
        retmax=deps.per_query_retmax,
        sort=ESEARCH_SORT,
        # From state, not from `deps`. The two agree on a fresh run and can disagree on a
        # resumed one, and state is the one that was in force when the corpus was chosen.
        text_basis_policy=str(state.get("basis") or ""),
        cost=_cost(deps),
        stale_after=None,
        warnings=list(state.get("warnings") or []),
        verified_by=str(state.get("verified_by") or ""),
    )
    return manifest.with_staleness(built_on=(started or finished).date())


def _models(deps: Deps) -> dict[str, str]:
    """The models that actually ran, not whatever config says when someone reads this."""
    if deps.router is None:
        return {}
    models: dict[str, str] = {}
    for role in Role:
        try:
            models[role.value] = deps.settings.model_for(role)
        except ConfigError:
            continue
    return models


def _counts(state: RunState, records: list[ConceptRecord]) -> BundleCounts:
    executed = list(state.get("executed") or [])
    verdicts = list(state.get("verdicts") or [])
    curation = state.get("curation")
    texts = dict(state.get("texts") or {})
    return BundleCounts(
        found=sum(query.retrieved for query in executed),
        unique=len(state.get("unique") or []),
        screened=len(verdicts),
        included=sum(1 for verdict in verdicts if verdict.include),
        curated=len(curation.kept) if curation is not None else 0,
        full_text_fetched=sum(1 for text in texts.values() if text.is_full_text),
        extracted=len(state.get("extractions") or {}),
        emitted=len(records),
    )


def _topics(charter: Charter, records: list[ConceptRecord]) -> list[TopicSummary]:
    grouped: dict[str, list[ConceptRecord]] = {topic.slug: [] for topic in charter.topic_taxonomy}
    for record in records:
        grouped.setdefault(record.domain, []).append(record)
    summaries: list[TopicSummary] = []
    for slug, items in grouped.items():
        topic = charter.topic(slug)
        full = sum(1 for record in items if record.text_basis is TextBasis.FULL_TEXT)
        summaries.append(
            TopicSummary(
                slug=slug,
                title=topic.title if topic is not None else "",
                papers=len(items),
                full_text=full,
                abstract_only=len(items) - full,
            )
        )
    return summaries


def _cost(deps: Deps) -> CostSummary:
    """Rendered by the ledger, never re-formatted here.

    `format_usd` is the one function allowed to turn these numbers into a string, so a
    manifest cannot report `$0.00` for calls that were merely unpriced.
    """
    if deps.router is None:
        return CostSummary()
    ledger = deps.router.ledger
    return CostSummary(
        calls=ledger.calls,
        prompt_tokens=ledger.prompt_tokens,
        completion_tokens=ledger.completion_tokens,
        usd=ledger.usd,
        unpriced_calls=ledger.unpriced_calls,
        display=ledger.format_usd(),
    )
