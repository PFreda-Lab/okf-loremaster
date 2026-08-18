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
from collections.abc import Iterator
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
    return _validate(_unwrap(_payload(text), model_cls), model_cls, text)


def parse_model_with(text: str, model_cls: type[M], **known: Any) -> M:
    """`parse_model`, with fields we already hold substituted into the reply.

    A PMID, or the user's own prompt, is ours and not the model's. Asking for it back
    costs output tokens to receive something we would discard, and risks receiving it
    wrong — a screener that transposes two digits of a PMID files its verdict against a
    different paper, and nothing downstream can tell.

    `known` wins over whatever the reply said, so a model that volunteered the field
    anyway cannot override us.
    """
    payload = _unwrap(_payload(text), model_cls)
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


def _unwrap(payload: Any, model_cls: type[Model]) -> Any:
    """Peel one envelope key off a reply that nested the object we asked for.

    Providers implement schema-constrained output as a forced tool call: the schema
    becomes a tool's input schema, and the model fills that tool. Models regularly fill
    it with an envelope — `{"parameters": {...}}`, or the tool's own name as the key —
    rather than the schema's top-level fields. Both were observed on consecutive calls
    to the same model, so the key cannot be matched by name.

    This is the most expensive failure this module can have, because nothing about it
    looks like one. The reply is valid JSON and validates cleanly, so no error is
    raised, no repair is retried, and on a schema whose fields are all optional every
    field lands on its default. That is how one run emitted 184 blank papers, at full
    price, and passed every check downstream.

    Peeled only when the outer object cannot be the schema itself: exactly one key, not
    a field the schema declares, wrapping an object. Anything else is returned as it
    came and fails validation in the ordinary way.
    """
    if not isinstance(payload, dict) or len(payload) != 1:
        return payload
    ((key, inner),) = payload.items()
    if not isinstance(inner, dict) or key in _field_names(model_cls):
        return payload
    return inner


def _field_names(model_cls: type[Model]) -> set[str]:
    """Every name the schema answers to, field names and aliases alike."""
    names: set[str] = set()
    for name, field in model_cls.model_fields.items():
        names.add(name)
        if field.alias:
            names.add(field.alias)
    return names


def _decoded(value: Any) -> Any:
    """`value` parsed, when it is a string carrying JSON and nothing else."""
    if not isinstance(value, str) or value.strip()[:1] not in ("[", "{"):
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def _restrung(payload: Any, model_cls: type[Model]) -> Iterator[Any]:
    """Candidates for a reply whose structure came back as text instead of as JSON.

    The same forced tool call that nests an envelope sometimes serializes the object
    it was asked for and hands back the *string*. Reproduced on the balanced model
    2026-08-04, twice minutes apart on the same charter:

        {"queries": "{\\"queries\\": [{\\"topic\\": \\"\\", \\"term\\": \\"...\\"}]}"}

    The whole plan, correct, inside a string, under its own field name. It cost a
    run: planning failed, the deterministic fallback took over, the fallback's anchor
    matched nothing, and an empty bundle came out calling itself valid.

    Only reached once honest validation has already failed, so a reply that was right
    the first time never comes near this.
    """
    if not isinstance(payload, dict):
        return
    if len(payload) == 1:
        ((_, only),) = payload.items()
        inner = _decoded(only)
        if inner is not None:
            yield _unwrap(inner, model_cls)
    swapped = {
        key: value if (parsed := _decoded(value)) is None else parsed
        for key, value in payload.items()
    }
    if swapped != payload:
        yield swapped


def _validate(payload: Any, model_cls: type[M], text: str) -> M:
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        for candidate in _restrung(payload, model_cls):
            try:
                return model_cls.model_validate(candidate)
            except ValidationError:
                continue
        raise SchemaError(
            f"reply did not match {model_cls.__name__}: "
            f"{exc.error_count()} problem(s) — {_problems(exc)}",
            hint=repair_hint(exc),
            raw=text,
        ) from exc


def _problems(exc: ValidationError) -> str:
    """The first few failures, as `field: what was wrong`.

    Capped at three problems: past that the reply is wrong in kind rather than in
    detail, and a long list of field paths is a worse prompt than a short one.
    """
    lines: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


def repair_hint(exc: ValidationError) -> str:
    """A short, field-level instruction for a follow-up message.

    The same problems the message carries. A count on its own — "1 problem(s)" — is
    what a warning says when it has the answer in hand and declines to pass it on:
    two topics failed curation for a whole run and the log could not say which field
    was wrong, when the exception knew all along.
    """
    return "Fix these fields and reply with JSON only — " + _problems(exc)


def _require_every_property(node: Any) -> Any:
    """Every property of every object listed in `required`, in place of pydantic's.

    A field with a default is a field pydantic leaves out of `required`, and a provider
    counts each one as an optional parameter it has to branch on while compiling the
    schema into a decoding grammar. There is a ceiling on that count, it is not
    documented, and it is reached silently: the schema is rejected, the router reads the
    rejection as a refusal, and every call for the rest of the run goes out unconstrained.

    Measured on an Azure AI Foundry Anthropic deployment (2026-08-18), which answered
    `Schemas contains too many optional parameters (38)`. Extraction had 31 and was fine;
    `interacts_with` and the five fields of `Interaction` took it to 38 and it stopped
    compiling — so the cap sits somewhere in between, and the next field added would have
    found it anyway.

    Requiring everything takes the count to zero and keeps the meaning. A field that was
    already `X | None` still accepts null, because `anyOf` is untouched; one that was a
    string or a list must now be written out as `""` or `[]`, which is the default it
    would have taken. Nothing here changes the models themselves, so a reply that omits a
    field still validates on the fallback path — this constrains what a provider generates,
    not what we accept.
    """
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = sorted(properties)
        for value in node.values():
            _require_every_property(value)
    elif isinstance(node, list):
        for value in node:
            _require_every_property(value)
    return node


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
            "schema": _require_every_property(model_cls.model_json_schema()),
        },
    }
