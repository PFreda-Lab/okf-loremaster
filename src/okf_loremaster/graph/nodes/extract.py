"""The extract node: one balanced-tier reading per paper.

One call per paper, for the same reason as `screen`: a batched call returns rows whose
alignment to the input is the model's to get right, and here a misaligned row would
attach one paper's effect size to another paper's PMID — a provenance error that reads
as a perfectly ordinary finding forever after.

**Balanced tier, not reasoning.** This node makes one call per kept paper — two hundred
of them against every other node's handful — so whatever it is bound to sets the price of
a run, and the reasoning tier made extraction most of the bill on its own. The work does
not need that tier: the fields are a transcription of what one paper printed, and the
only judgment in them is which relationships answer the review's question and how sure to
be about each row. What guards the rest is code, not model quality — every number is
checked against the source afterward, every quote is sliced out of it, and a fabricated
code is deleted. That check does not get better on a more expensive model, because a
number that is not in the paper fails it either way.

Unlike screening, a failed parse is retried once. A screening verdict is five short
fields and a lost one costs a paper that was probably being excluded anyway; an
extraction is the entire content of a bundle file, the call that produced it is the most
expensive in the pipeline, and `SchemaError.hint` names the field that was wrong — which
a model will usually supply when told. Once, not twice: a second failure is a model that
cannot satisfy the schema, and paying a third call to confirm that is waste.

The source text is not built here. It arrives from `fulltext` already budgeted and
already the exact string `verification` will check against, and is passed through
unchanged.

**A paper is never bought twice.** Two mechanisms, because they cover different
accidents. `extractions` is checkpointed and keyed by PMID, so a run resumed after this
node finished skips the whole thing. `extraction_cache` is written per paper as each one
comes back, so a run interrupted *inside* this node — the likely place, since it is the
long one — resumes having kept everything it already paid for.
"""

from __future__ import annotations

import asyncio
from typing import Any

from okf_loremaster.config import Role
from okf_loremaster.extraction_cache import fingerprint
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.prompts import EXTRACT_SYSTEM, extract_context, extract_user
from okf_loremaster.schemas import Charter, Extraction, PaperText
from okf_loremaster.schemas.parse import SchemaError, parse_model, response_format_for

__all__ = ["extract_node"]

NODE = "extract"

# Enough for a full concept: `MAX_PREDICTOR_ROWS` rows with quote locators, null
# findings, and the prose fields, each already under its own length budget.
#
# Twice measured wrong, at 3072 and at 6144, both times by capping the budget instead of
# the reply. What a truncated extraction actually costs is the whole paper: the JSON
# never closes, so it never parses, so the repair retry has nothing to repair and the
# paper is dropped — after being billed for in full, twice. A ceiling is not a purchase
# either, since a reply that finishes in 2,000 tokens is billed for 2,000 whatever the
# ceiling says. So the ceiling was never the lever; the size of the reply was. The prompt
# now caps the two lists that had no limit at all, asks for compact JSON, and asks for
# quote locators rather than copied sentences — together about a third of what replies
# used to run to. This is generous room for one that respects all three.
MAX_EXTRACTION_TOKENS = 8192

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
    # Keyed by PMID so a run resumed after this node finished skips it entirely. The
    # per-paper cache below is what covers a run interrupted part way through it.
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
    """Every paper at once, bounded by the router's balanced-tier semaphore."""
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
    cache = deps.extraction_cache
    total = len(todo)
    failures: list[str] = []
    retried = 0
    reused = 0
    done = 0

    async def read(slug: str, source: PaperText) -> tuple[str, Extraction] | None:
        nonlocal done, retried, reused
        # Rendered here rather than inside the call, because the cache key has to be the
        # request itself. Fingerprinting the pieces instead would keep answering from
        # cache after an edit to the prompt template that changed every one of them.
        user = extract_user(context=contexts[slug], paper=source.text)
        key = fingerprint(EXTRACT_SYSTEM, user)
        hit = cache.get(source.pmid, key) if cache is not None else None
        # A blank entry is not a saving, it is a poisoned one, and it was written by a
        # version that could not tell the difference. Treated as a miss it is re-read
        # once and overwritten, so a cache full of them heals as it is used rather than
        # having to be found and deleted by hand.
        if hit is not None and _is_blank(hit):
            hit = None
        if hit is not None:
            reused += 1
            outcome: Extraction | str = hit
            used_retry = False
        else:
            outcome, used_retry = await _extract_one(deps, user, schema)
        if used_retry:
            retried += 1
        if isinstance(outcome, str):
            failures.append(f"{source.pmid}: {outcome}")
            result = None
        else:
            result = (source.pmid, outcome)
            if cache is not None and hit is None:
                cache.put(source.pmid, key, outcome)
        done += 1
        deps.progress(NODE, f"extracted {done} of {total}", current=done, total=total)
        return result

    gathered = await asyncio.gather(*(read(slug, source) for slug, source in todo))
    _report(deps, warnings, failures, retried=retried, reused=reused, total=total)
    return [item for item in gathered if item is not None]


async def _extract_one(
    deps: Deps, user: str, schema: dict[str, Any]
) -> tuple[Extraction | str, bool]:
    """One paper, with a single schema repair. Returns the extraction or why not."""
    assert deps.router is not None  # guarded by the caller
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user},
    ]
    used_retry = False

    for attempt in (1, 2):
        try:
            result = await deps.router.complete(
                Role.BALANCED,
                messages,
                node=NODE,
                max_tokens=MAX_EXTRACTION_TOKENS,
                response_format=schema,
            )
        except Exception as exc:
            return f"extraction call failed ({type(exc).__name__}: {exc})", used_retry

        try:
            parsed = parse_model(result.text, Extraction)
            if _is_blank(parsed):
                raise SchemaError(
                    "reply validated but carried no extracted content",
                    hint=(
                        "Reply with a single JSON object whose top-level keys are the "
                        "schema's own fields — description, bottom_line, study_design, "
                        "n, population, outcome_definition, predictors, null_findings, "
                        "vocabulary_hints, caveats, tags. Do not nest them under a "
                        "wrapper key."
                    ),
                    raw=result.text,
                )
            return parsed, used_retry
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


def _is_blank(extraction: Extraction) -> bool:
    """Whether an extraction says nothing whatsoever about the paper.

    Every field on `Extraction` is optional, which is deliberate — a paper that reports
    no null finding should not fail a schema over it. The cost of that is a reply the
    model never really answered validating cleanly into defaults, so the emptiness has
    to be caught here instead. A paper can honestly have no predictor rows; none can
    have no description, no bottom line and no predictors at once, because the first two
    describe the paper rather than its findings.
    """
    return not (extraction.description or extraction.bottom_line or extraction.predictors)


def _scope(charter: Charter, slug: str) -> str:
    topic = charter.topic(slug)
    if topic is None:
        return ""
    return topic.scope or topic.title


def _report(
    deps: Deps, warnings: list[str], failures: list[str], *, retried: int, reused: int, total: int
) -> None:
    """Say what went wrong, without repeating it once per paper."""
    if reused:
        # Said out loud because it is the difference between a resumed run costing
        # nothing and costing what the first attempt cost, and a saving nobody can see
        # is one nobody trusts.
        deps.progress(NODE, f"{reused} of {total} paper(s) were already read, and cost nothing")
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
