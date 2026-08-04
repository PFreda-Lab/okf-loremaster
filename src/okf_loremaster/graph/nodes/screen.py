"""The screen node: one FAST judgment per paper in the pool.

One call per paper, deliberately. Batching forty abstracts into one call is cheaper per
token and is the obvious optimization, but it returns forty verdicts whose alignment to
the input is the model's to get right, and a single dropped row shifts every verdict
after it onto the wrong paper — silently, and in the one node whose output nothing
downstream can sanity-check. One paper per call cannot misalign, fails in isolation, and
is the shape `llm.estimate` projects, so the call count printed at the retrieve pause is
the call count the run then makes.

Concurrency belongs to the router's FAST semaphore. Every paper is submitted at once and
the semaphore decides how many are in flight, which keeps the rate limit in the one place
that already knows about it.

A re-query round screens only what it has not already screened, and the budget is global
across rounds rather than per round. The retrieve pause is where a person approved a
number of papers to spend on; a second round that quietly doubled it would make that
approval mean nothing.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from okf_loremaster.config import Role
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.prompts import SCREEN_SYSTEM, screen_context, screen_user
from okf_loremaster.schemas import Candidate, Charter, Confidence, ScoredCandidate, ScreenVerdict
from okf_loremaster.schemas.parse import SchemaError, parse_model_with, response_format_for

__all__ = ["screen_node"]

NODE = "screen"

# A verdict is five short fields. Anything longer is a model narrating, and cutting it
# off costs nothing that would have parsed.
MAX_VERDICT_TOKENS = 256


async def screen_node(state: RunState, deps: Deps) -> dict[str, Any]:
    charter = state.get("charter")
    if charter is None:
        raise RuntimeError("screen reached without a charter — the graph is wired wrong")

    pool: list[ScoredCandidate] = list(state.get("pool") or [])
    warnings = list(state.get("warnings") or [])
    verdicts = {verdict.pmid: verdict for verdict in state.get("verdicts") or []}

    with span(deps, NODE) as report:
        pending = [item.candidate for item in pool if item.pmid not in verdicts]
        todo = pending[: max(0, deps.screen_budget - len(verdicts))]
        if len(pending) > len(todo):
            note = (
                f"screen budget of {deps.screen_budget} reached: "
                f"{len(pending) - len(todo)} pooled paper(s) were not screened"
            )
            warnings.append(note)
            deps.warn(NODE, note)

        for verdict in await _screen_all(deps, charter, todo, warnings):
            verdicts[verdict.pmid] = verdict

        included = sum(1 for verdict in verdicts.values() if verdict.include)
        report["summary"] = (
            f"{len(todo)} screened this round, {included} included of {len(verdicts)}"
        )

    return {"verdicts": list(verdicts.values()), "warnings": warnings}


async def _screen_all(
    deps: Deps, charter: Charter, candidates: list[Candidate], warnings: list[str]
) -> list[ScreenVerdict]:
    """Every paper at once, bounded by the router's FAST semaphore."""
    if not candidates:
        return []
    if deps.router is None:
        note = "no model is available, so nothing was screened"
        warnings.append(note)
        deps.warn(NODE, note)
        return []

    context = screen_context(
        task=charter.task or charter.prompt,
        population=charter.population,
        outcome=charter.outcome,
        inclusion=list(charter.inclusion),
        exclusion=list(charter.exclusion),
        topics=[(topic.slug, topic.scope or topic.title) for topic in charter.topic_taxonomy],
    )
    schema = response_format_for(ScreenVerdict, name="screen_verdict")
    known = set(charter.slugs)
    folded = {_fold(slug): slug for slug in charter.slugs}
    total = len(candidates)
    failures: list[str] = []
    unknown: set[str] = set()
    done = 0

    async def judge(candidate: Candidate) -> ScreenVerdict:
        nonlocal done
        try:
            verdict = await _screen_one(deps, context, schema, candidate)
        except SchemaError as exc:
            failures.append(f"{candidate.pmid}: {exc}")
            verdict = _unreadable(candidate.pmid, "screening reply did not parse")
        except Exception as exc:  # a call that never returned is not a judgment
            failures.append(f"{candidate.pmid}: {type(exc).__name__}: {exc}")
            verdict = _unreadable(candidate.pmid, f"screening call failed ({type(exc).__name__})")

        if verdict.topic and verdict.topic not in known:
            recovered = _resolve(verdict.topic, folded)
            if recovered is None:
                # `repr`, not the bare string: what makes one of these unusable is
                # almost always invisible in a log line. Reading these back is what
                # showed the screener listing four topics in a one-topic field.
                unknown.add(repr(verdict.topic))
            verdict = verdict.model_copy(update={"topic": recovered or ""})

        done += 1
        deps.progress(NODE, f"screened {done} of {total}", current=done, total=total)
        return verdict

    verdicts = list(await asyncio.gather(*(judge(candidate) for candidate in candidates)))
    _report(deps, warnings, failures, unknown, total=total)
    return verdicts


async def _screen_one(
    deps: Deps, context: str, schema: dict[str, Any], candidate: Candidate
) -> ScreenVerdict:
    assert deps.router is not None  # guarded by the caller
    result = await deps.router.complete(
        Role.FAST,
        [
            {"role": "system", "content": SCREEN_SYSTEM},
            {
                "role": "user",
                "content": screen_user(context=context, paper=candidate.screening_text),
            },
        ],
        node=NODE,
        max_tokens=MAX_VERDICT_TOKENS,
        response_format=schema,
    )
    return parse_model_with(result.text, ScreenVerdict, pmid=candidate.pmid)


def _fold(slug: str) -> str:
    """A topic name reduced to what a model cannot get cosmetically wrong.

    Only what the charter itself would treat as the same directory: letters and digits,
    lowercased. Anything narrower would have to guess at which punctuation is meaningful.
    """
    return "".join(char for char in slug.lower() if char.isalnum())


def _resolve(named: str, folded: Mapping[str, str]) -> str | None:
    """The charter topic the screener meant, or None if it named none of them.

    Two things go wrong with this field, and both are the model being helpful rather
    than wrong. It returns a slug that differs only in case or dash style — the
    charter plainly has that topic, and dropping the hint over typography wastes a
    judgment already paid for. Or, for a paper that genuinely spans several topics, it
    lists them all in a field that holds one: `"a; b; c"`. Taking the first named
    topic the charter has is the honest reading of that — the field is advisory, the
    curator sees every paper on a topic at once and decides, and a first choice is
    better information than a blank.
    """
    whole = folded.get(_fold(named))
    if whole is not None:
        return whole
    for part in re.split(r"[,;/|\n]", named):
        match = folded.get(_fold(part))
        if match is not None:
            return match
    return None


def _unreadable(pmid: str, reason: str) -> ScreenVerdict:
    """The verdict for a paper the screener could not answer for.

    Excluded at relevance 0, which also keeps it out of the floor backfill: `borderline`
    means a paper we read and nearly kept, and a call that never returned is not evidence
    of anything at all about the paper.
    """
    return ScreenVerdict(
        pmid=pmid,
        include=False,
        relevance=0,
        reason=reason[:240],
        confidence=Confidence.LOW,
    )


def _report(
    deps: Deps, warnings: list[str], failures: list[str], unknown: set[str], *, total: int
) -> None:
    """Say what went wrong, without repeating it once per paper."""
    if failures:
        note = (
            f"{len(failures)} of {total} screening call(s) failed and were excluded — "
            f"first: {failures[0]}"
        )
        warnings.append(note)
        deps.warn(NODE, note)
    if len(failures) * 2 > total:
        note = (
            "more than half the screening calls failed: what follows is not a judgment "
            "about the literature, it is whatever survived the model being unavailable"
        )
        warnings.append(note)
        deps.warn(NODE, note)
    if unknown:
        note = (
            "screener named topic(s) the charter does not have, so those papers carry no "
            "topic hint: " + ", ".join(sorted(unknown))
        )
        warnings.append(note)
        deps.warn(NODE, note)
