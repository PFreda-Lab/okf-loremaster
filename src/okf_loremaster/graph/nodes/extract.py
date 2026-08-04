"""The extract node: one reasoning-tier reading per paper.

One call per paper, for the same reason as `screen`: a batched call returns rows whose
alignment to the input is the model's to get right, and here a misaligned row would
attach one paper's effect size to another paper's PMID — a provenance error that reads
as a perfectly ordinary finding forever after.

Unlike screening, a failed parse is retried once. A screening verdict is five short
fields and a lost one costs a paper that was probably being excluded anyway; an
extraction is the entire content of a bundle file, the call that produced it is the most
expensive in the pipeline, and `SchemaError.hint` names the field that was wrong — which
a model will usually supply when told. Once, not twice: a second failure is a model that
cannot satisfy the schema, and paying a third reasoning-tier call to confirm it is waste.

The source text is not built here. It arrives from `fulltext` already budgeted and
already the exact string `verification` will check against, and is passed through
unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

from okf_loremaster.config import Role
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.prompts import EXTRACT_SYSTEM, extract_context, extract_user
from okf_loremaster.schemas import Charter, Extraction, PaperText
from okf_loremaster.schemas.parse import SchemaError, parse_model, response_format_for

__all__ = ["extract_node"]

NODE = "extract"

# Enough for a full concept: twelve predictor rows with quotes, null findings, and the
# prose fields, each already under its own length budget.
MAX_EXTRACTION_TOKENS = 3072

# How much of a failed reply to show the model when asking it to try again. The reply
# is echoed back so the repair is an edit rather than a second attempt from nothing;
# beyond a few thousand characters it is a model that ignored the schema, and re-sending
# the whole thing pays twice for the same mistake.
MAX_ECHOED_CHARS = 3000


async def extract_node(state: RunState, deps: Deps) -> dict[str, Any]:
    charter = state.get("charter")
    if charter is None:
        raise RuntimeError("extract reached without a charter — the graph is wired wrong")

    topics = state.get("topics") or {}
    texts = state.get("texts") or {}
    warnings = list(state.get("warnings") or [])
    # Keyed by PMID so a resumed run pays only for what it has not extracted.
    extractions = dict(state.get("extractions") or {})

    with span(deps, NODE) as report:
        todo = [
            (slug, texts[pmid])
            for slug, pmids in topics.items()
            for pmid in pmids
            if pmid not in extractions and pmid in texts
        ]
        for pmid, extraction in await _extract_all(deps, charter, todo, warnings):
            extractions[pmid] = extraction

        wanted = sum(len(pmids) for pmids in topics.values())
        report["summary"] = f"{len(extractions)} of {wanted} paper(s) extracted"

    return {"extractions": extractions, "warnings": warnings}


async def _extract_all(
    deps: Deps,
    charter: Charter,
    todo: list[tuple[str, PaperText]],
    warnings: list[str],
) -> list[tuple[str, Extraction]]:
    """Every paper at once, bounded by the router's reasoning-tier semaphore."""
    if not todo:
        return []
    if deps.router is None:
        note = "no model is available, so nothing was extracted"
        warnings.append(note)
        deps.warn(NODE, note)
        return []

    # One context per topic, built once. Papers on a topic are extracted back to back,
    # so a byte-identical prefix across them is what a provider's prompt cache can
    # charge for once instead of on every call in the most expensive node here.
    contexts = {
        slug: extract_context(
            task=charter.task or charter.prompt,
            outcome=charter.outcome,
            topic=slug,
            scope=_scope(charter, slug),
        )
        for slug in {slug for slug, _ in todo}
    }
    schema = response_format_for(Extraction, name="extraction")
    total = len(todo)
    failures: list[str] = []
    retried = 0
    done = 0

    async def read(slug: str, source: PaperText) -> tuple[str, Extraction] | None:
        nonlocal done, retried
        outcome, used_retry = await _extract_one(deps, contexts[slug], schema, source)
        if used_retry:
            retried += 1
        if isinstance(outcome, str):
            failures.append(f"{source.pmid}: {outcome}")
            result = None
        else:
            result = (source.pmid, outcome)
        done += 1
        deps.progress(NODE, f"extracted {done} of {total}", current=done, total=total)
        return result

    gathered = await asyncio.gather(*(read(slug, source) for slug, source in todo))
    _report(deps, warnings, failures, retried=retried, total=total)
    return [item for item in gathered if item is not None]


async def _extract_one(
    deps: Deps, context: str, schema: dict[str, Any], source: PaperText
) -> tuple[Extraction | str, bool]:
    """One paper, with a single schema repair. Returns the extraction or why not."""
    assert deps.router is not None  # guarded by the caller
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": extract_user(context=context, paper=source.text)},
    ]
    used_retry = False

    for attempt in (1, 2):
        try:
            result = await deps.router.complete(
                Role.REASONING,
                messages,
                node=NODE,
                max_tokens=MAX_EXTRACTION_TOKENS,
                response_format=schema,
            )
        except Exception as exc:
            return f"extraction call failed ({type(exc).__name__}: {exc})", used_retry

        try:
            return parse_model(result.text, Extraction), used_retry
        except SchemaError as exc:
            if attempt == 2 or not exc.hint:
                return f"extraction reply did not parse ({exc})", used_retry
            used_retry = True
            messages = [
                *messages,
                {"role": "assistant", "content": result.text[:MAX_ECHOED_CHARS]},
                {"role": "user", "content": exc.hint},
            ]

    # Unreachable: both branches of the loop return. Present so the type is honest.
    return "extraction did not complete", used_retry


def _scope(charter: Charter, slug: str) -> str:
    topic = charter.topic(slug)
    if topic is None:
        return ""
    return topic.scope or topic.title


def _report(
    deps: Deps, warnings: list[str], failures: list[str], *, retried: int, total: int
) -> None:
    """Say what went wrong, without repeating it once per paper."""
    if retried:
        deps.progress(NODE, f"{retried} of {total} extraction(s) needed a schema repair")
    if failures:
        note = (
            f"{len(failures)} of {total} extraction(s) failed and those papers were "
            f"dropped — first: {failures[0]}"
        )
        warnings.append(note)
        deps.warn(NODE, note)
    if failures and len(failures) * 2 > total:
        note = (
            "more than half the extractions failed: the bundle below is not a reading of "
            "the literature, it is whatever survived the model being unavailable"
        )
        warnings.append(note)
        deps.warn(NODE, note)
