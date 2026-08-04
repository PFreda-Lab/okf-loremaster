"""Turning a model's reply into a validated schema object.

Every judgment node in the graph ends here, so this is where the ordinary ways a
structured reply arrives malformed get absorbed: wrapped in a fenced code block,
preceded by a sentence of preamble, or carrying a trailing comma. None of those are
worth a retry — a retry costs a whole call to fix punctuation.

What is *not* absorbed is a reply whose fields are wrong. That raises `SchemaError`
carrying a short repair hint the node can put in a follow-up message, because a model
that omitted a required field will usually supply it when told which one.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import ValidationError

from okf_loremaster.schemas.common import Model

__all__ = [
    "SchemaError",
    "extract_json",
    "parse_model",
    "parse_model_with",
    "repair_hint",
    "response_format_for",
    "strip_fences",
]

M = TypeVar("M", bound=Model)

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
# A comma directly before a closing brace or bracket. Legal in JavaScript, common in
# generated JSON, and rejected by json.loads.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

_OPENERS = {"{": "}", "[": "]"}


class SchemaError(ValueError):
    """A reply that could not be validated, with a hint for asking again."""

    def __init__(self, message: str, *, hint: str = "", raw: str = "") -> None:
        super().__init__(message)
        self.hint = hint
        # Truncated: the raw reply goes into an exception message that may be logged,
        # and a full extraction reply is thousands of tokens.
        self.raw = raw[:2000]


def strip_fences(text: str) -> str:
    """Unwrap a fenced code block, if the whole reply is one."""
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


def extract_json(text: str) -> str:
    """The first complete JSON value in the text.

    Scans for a balanced object or array rather than slicing between the first `{` and
    the last `}`, because a reply that ends with a sentence after the JSON would
    otherwise swallow the prose. Strings are tracked so a brace inside a quoted value
    does not shift the count — quoted braces show up in verbatim source quotes often
    enough to matter.
    """
    body = strip_fences(text)
    start = next((i for i, ch in enumerate(body) if ch in _OPENERS), -1)
    if start < 0:
        raise SchemaError("reply contains no JSON object or array", raw=text)

    closer = _OPENERS[body[start]]
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            depth += 1
        elif char in ("}", "]"):
            depth -= 1
            if depth == 0:
                if char != closer:
                    raise SchemaError("mismatched JSON brackets in reply", raw=text)
                return body[start : index + 1]
    raise SchemaError("reply ends inside an unterminated JSON value", raw=text)


def parse_model(text: str, model_cls: type[M]) -> M:
    """Validate a model reply into `model_cls`, or raise `SchemaError`."""
    return _validate(_payload(text), model_cls, text)


def parse_model_with(text: str, model_cls: type[M], **known: Any) -> M:
    """`parse_model`, with fields we already hold substituted into the reply.

    A PMID, or the user's own prompt, is ours and not the model's. Asking for it back
    costs output tokens to receive something we would discard, and risks receiving it
    wrong — a screener that transposes two digits of a PMID files its verdict against a
    different paper, and nothing downstream can tell.

    `known` wins over whatever the reply said, so a model that volunteered the field
    anyway cannot override us.
    """
    payload = _payload(text)
    if not isinstance(payload, dict):
        raise SchemaError(
            "reply is not a JSON object",
            hint="Reply with a single JSON object and no other text.",
            raw=text,
        )
    return _validate({**payload, **known}, model_cls, text)


def _payload(text: str) -> Any:
    fragment = extract_json(text)
    try:
        return json.loads(fragment)
    except ValueError:
        # One repair attempt, for the single most common generation artifact.
        try:
            return json.loads(_TRAILING_COMMA.sub(r"\1", fragment))
        except ValueError as exc:
            raise SchemaError(
                f"reply is not valid JSON: {exc}",
                hint="Reply with a single JSON object and no other text.",
                raw=text,
            ) from exc


def _validate(payload: Any, model_cls: type[M], text: str) -> M:
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise SchemaError(
            f"reply did not match {model_cls.__name__}: {exc.error_count()} problem(s)",
            hint=repair_hint(exc),
            raw=text,
        ) from exc


def repair_hint(exc: ValidationError) -> str:
    """A short, field-level instruction for a follow-up message.

    Capped at three problems: past that the reply is wrong in kind rather than in
    detail, and a long list of field paths is a worse prompt than a short one.
    """
    lines: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"{location}: {error['msg']}")
    return "Fix these fields and reply with JSON only — " + "; ".join(lines)


def response_format_for(model_cls: type[Model], *, name: str = "") -> dict[str, Any]:
    """A `response_format` payload constraining a reply to this schema.

    Best effort, never a guarantee — which is why `parse_model` still handles fences and
    preamble rather than trusting the constraint, and why every prompt asks for a single
    JSON object in words as well.

    `drop_params` covers a provider that does not support this at all. It does not cover
    a provider that supports it for accounts other than yours: LiteLLM's capability map
    answers for the model, the gate is on the workspace, and the parameter sails through
    to a 400. The router recognizes that refusal and stops sending the constraint.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or model_cls.__name__,
            "schema": model_cls.model_json_schema(),
        },
    }
