"""The charter node: turn a prompt into the run's terms of reference.

One reasoning-tier call, and the most consequential one in the graph — everything downstream is
scoped by what it decides. Three paths reach the same place:

- a caller-supplied charter short-circuits it, and no model is called at all.
- A dry run without a charter builds a skeleton from the prompt alone, so that
  `--dry-run` can still plan real queries against real hit counts without spending
  anything. The skeleton has no taxonomy, and says so.
- Otherwise the reasoning-tier model drafts one.

The prompt is never read back from the model's reply. A charter records what the user
asked for, and a paraphrase of the request is not the request.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from okf_loremaster.config import Role
from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.prompts import CHARTER_SYSTEM, charter_user
from okf_loremaster.schemas import Charter
from okf_loremaster.schemas.parse import (
    SchemaError,
    parse_model_with,
    response_format_for,
)

__all__ = ["charter_node"]

NODE = "charter"


async def charter_node(state: RunState, deps: Deps) -> dict[str, Any]:
    prompt = state.get("prompt", "")
    warnings = list(state.get("warnings") or [])

    with span(deps, NODE) as report:
        supplied = state.get("charter")
        if supplied is not None:
            charter = supplied.model_copy(deep=True)
            source = "supplied"
        elif deps.router is None:
            charter = _skeleton(prompt)
            source = "skeleton"
            note = (
                "no charter and no model calls allowed, so this run has no topic taxonomy; "
                "run without --dry-run to plan the real queries"
            )
            warnings.append(note)
            deps.warn(NODE, note)
        else:
            charter = await _draft(deps, prompt)
            source = "drafted"

        charter = _apply_overrides(charter, prompt=prompt, deps=deps)

        for problem in charter.problems():
            warnings.append(problem)
            deps.warn(NODE, problem)

        report["summary"] = f"{source}: {len(charter.topic_taxonomy)} topics"

    return {"charter": charter, "warnings": warnings}


# --- the three paths --------------------------------------------------------


def _skeleton(prompt: str) -> Charter:
    """A charter with nothing decided but the prompt.

    Enough for `queries.deterministic_plan` to build real, valid PubMed queries from the
    prompt's own phrases. Not enough to build a bundle, which is why it only ever
    appears on a dry run.
    """
    return Charter(prompt=prompt, task=prompt)


async def _draft(deps: Deps, prompt: str) -> Charter:
    """One reasoning-tier call, plus at most one repair attempt.

    The repair exists because a charter is expensive to lose: the reply is long, and a
    single missing field would otherwise cost the whole call. Past one attempt the reply
    is wrong in kind rather than in detail, and asking a third time is just spending.
    """
    assert deps.router is not None  # guarded by the caller; a dry run never gets here
    messages = [
        {"role": "system", "content": CHARTER_SYSTEM},
        {"role": "user", "content": charter_user(prompt)},
    ]
    result = await deps.router.complete(
        Role.REASONING,
        messages,
        node=NODE,
        max_tokens=4096,
        response_format=response_format_for(Charter, name="charter"),
    )
    try:
        return _parse(result.text, prompt=prompt)
    except SchemaError as exc:
        deps.warn(NODE, f"charter reply needed repair: {exc}")
        retry = await deps.router.complete(
            Role.REASONING,
            [
                *messages,
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": exc.hint or "Reply with a single JSON object."},
            ],
            node=NODE,
            max_tokens=4096,
            response_format=response_format_for(Charter, name="charter"),
        )
        return _parse(retry.text, prompt=prompt)


def _parse(text: str, *, prompt: str) -> Charter:
    """Validate a charter reply, with the user's own prompt substituted in.

    Not `parse_model`: `prompt` is required by the schema and is ours, not the model's.
    Asking for it back costs output tokens to receive a paraphrase we would discard.
    """
    return parse_model_with(text, Charter, prompt=prompt)


# --- overrides that apply on every path -------------------------------------


def _apply_overrides(charter: Charter, *, prompt: str, deps: Deps) -> Charter:
    """Settle the fields config, not the model, has the last word on."""
    updated = charter.model_copy(deep=True)
    updated.prompt = prompt or updated.prompt
    updated.target_papers = deps.target_papers
    updated.topic_min = deps.topic_min
    updated.topic_max = deps.topic_max
    if not updated.generated_by:
        updated.generated_by = _generated_by(deps)
    if updated.generated_at is None:
        updated.generated_at = datetime.now(UTC)
    # Revalidate: the assignments above bypass field validators.
    return Charter.model_validate(updated.model_dump(mode="json"))


def _generated_by(deps: Deps) -> str:
    if deps.router is None:
        return "okf-loremaster/charter/none"
    try:
        model = deps.settings.model_for(Role.REASONING)
    except Exception:
        model = "unknown"
    return f"okf-loremaster/charter/{model}"
