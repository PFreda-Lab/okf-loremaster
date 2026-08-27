"""LiteLLM-backed router: three role-bound models, retries, and honest cost accounting.

The cost accounting is the delicate part, and it is wrong in two directions rather than
one. `litellm.completion_cost()` returns 0.0 for a model it does not recognize rather
than raising, and a model reached through a gateway or under a custom deployment name is
routinely absent from its price map — so a run that reports $0.00 is indistinguishable
from a run that was genuinely free, which is the worst possible failure: it looks like
good news. The other direction is quieter. That price map is a static JSON file shipped
inside the installed wheel, dated the day the version was cut; nothing refreshes it, no
provider is consulted, and a published price that moves afterward leaves it confidently
quoting history.

So pricing goes through three stages and the third is explicit ignorance:
  1. `OKF_LOREMASTER_PRICE_<ROLE>_IN` / `_OUT`, in USD per 1M tokens.
  2. LiteLLM's own price map.
  3. `usd = None` — the call is counted in tokens and reported as unpriced.

Configured first, because a number somebody set deliberately beats one that shipped with
a dependency, and because reading the map first made stage 1 unreachable for every model
the map happens to name.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from okf_loremaster.config import Effort, Role, Settings
from okf_loremaster.events import EventBus, LLMCall, Progress, WarningEvent

# Provider errors that are worth another attempt. Anything else - a bad key, a
# malformed request, a model that does not exist - fails immediately, because
# retrying it just burns the rate limit and delays the real error.
_TRANSIENT_NAMES = (
    "RateLimitError",
    "APIConnectionError",
    "Timeout",
    "ServiceUnavailableError",
    "InternalServerError",
    "APIResponseValidationError",
)
_PERMANENT_NAMES = (
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "NotFoundError",
    "UnprocessableEntityError",
    "ContextWindowExceededError",
)

# Schema-constrained output is gated on the account, not just the model, and LiteLLM's
# capability map only answers for the model. `drop_params` therefore sees a model that
# supports `response_format`, passes it through, and a workspace without access gets a
# 400 instead of a dropped parameter — one node deep, mid-run. Matched narrowly on the
# provider's own wording: a wider match would silently discard the schema on unrelated
# bad requests, which is the failure this whole module is written to avoid.
#
# "Grammar compilation timed out" is the same event said differently, and it is worth
# naming because it does not read like a refusal: a provider compiles the schema into a
# decoding grammar before the model sees anything, and a schema it cannot compile in
# time is a schema it will not honor. Observed on an Azure AI Foundry Anthropic
# deployment (2026-08-17), where it killed the charter node on three consecutive runs
# across two models — deterministic for one schema, so retrying the constraint would
# only fail again. The word "timed out" is what makes it look transient and is exactly
# why the transient path is the wrong one: nothing about waiting makes a schema smaller.
_SCHEMA_REFUSALS = ("structured_outputs", "structured outputs", "grammar compilation")


def _refuses_schema(exc: BaseException) -> bool:
    """Whether this failure is the provider declining schema-constrained output."""
    return any(marker in str(exc).lower() for marker in _SCHEMA_REFUSALS)


# Providers state the wait in the body of a 429 rather than only in a header, and the
# number is the one fact that makes the difference between a retry that can succeed and
# one that cannot: a token-per-minute limit clears on the provider's clock, not ours.
_WAIT_HINT = re.compile(r"wait\s+(\d+(?:\.\d+)?)\s*second", re.IGNORECASE)
_RETRY_AFTER = re.compile(r"retry[-_ ]?after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.IGNORECASE)

# A ceiling below the provider's window makes every retry futile. Rate limits are
# quoted per 60s, so a backoff that tops out under that can only ever fail.
_MAX_BACKOFF = 75.0

# One doubling of the token budget when a reply is cut off. Two would quadruple the
# budget of the most expensive nodes on a model that simply will not stop talking,
# and by then the budget is not the problem.
_TRUNCATION_ATTEMPTS = 2


def _requested_wait(exc: BaseException) -> float | None:
    """The pause the provider asked for, if it named one.

    Read from the exception text because that is where it reliably is: LiteLLM wraps
    the provider's JSON body into the message, and the `retry-after` header is not
    exposed uniformly across providers.
    """
    text = str(exc)
    for pattern in (_WAIT_HINT, _RETRY_AFTER):
        found = pattern.search(text)
        if found:
            # Trust it only within reason: a provider is free to name a wait longer
            # than any run should sit idle for.
            return min(_MAX_BACKOFF, float(found.group(1)))
    return None


@lru_cache(maxsize=1)
def _exception_types() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    """(transient, permanent) exception classes, resolved lazily from litellm."""
    import litellm

    def resolve(names: tuple[str, ...]) -> tuple[type[BaseException], ...]:
        found: list[type[BaseException]] = []
        for name in names:
            candidate = getattr(litellm, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                found.append(candidate)
        return tuple(found)

    return resolve(_TRANSIENT_NAMES), resolve(_PERMANENT_NAMES)


class StreamStalled(TimeoutError):
    """A streamed reply stopped arriving. Carries whatever got through before it did.

    A `TimeoutError` subclass on purpose: `_is_transient` already answers yes for that,
    so a stall reaches the ordinary retry path without any error taxonomy learning a new
    word. It arrives sooner than `request_timeout` would have, and nothing else changes.

    `partial` is the reassembled prefix, or None if nothing arrived at all. Those tokens
    were generated and will be billed whether or not we waited for the rest, so they are
    ledgered before the retry — the same rule that forbids reporting an unpriced call as
    free forbids reporting a spent one as never made.
    """

    def __init__(self, message: str, partial: Any = None) -> None:
        super().__init__(message)
        self.partial = partial


# `StopAsyncIteration` does not survive being awaited inside a task, so exhaustion is
# reported as a value instead of an exception.
_END = object()


async def _next_chunk(iterator: Any) -> Any:
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return _END


def _assemble(chunks: list[Any], messages: list[dict[str, str]]) -> Any:
    """Rebuild one completion response out of the chunks that arrived.

    LiteLLM's own reassembler, rather than ours, because everything downstream reads a
    completion's shape and a hand-rolled approximation would be a second shape to keep
    true. Verified against the live gateway 2026-08-27: it preserves the provider's real
    usage when the final chunk carries one, preserves `finish_reason='length'` so the
    truncation retry still fires, and puts a schema-constrained reply in
    `message.content` exactly as the unstreamed path does rather than leaving it in
    `tool_calls`. That last one is why this was measured before it was written: every
    judgment node sends a `response_format`, so content landing anywhere else would have
    emptied all of them at once.

    Returns None for no chunks at all, which the caller treats as a failed call.
    """
    if not chunks:
        return None
    import litellm

    return litellm.stream_chunk_builder(chunks, messages=messages)


async def _aclose(stream: Any) -> None:
    """Best-effort release of an abandoned stream's connection.

    Abandoning is the point, but the socket underneath is still open, and a run that
    stalls on several extractions would leak one per attempt.
    """
    for name in ("aclose", "close"):
        closer = getattr(stream, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass
        return


def _is_transient(exc: BaseException) -> bool:
    transient, permanent = _exception_types()
    if permanent and isinstance(exc, permanent):
        return False
    if transient and isinstance(exc, transient):
        return True
    # Connection-level failures from the underlying HTTP stack.
    return isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError))


class CompletionFn(Protocol):
    """The async completion callable. Swapped out wholesale in tests."""

    async def __call__(self, **kwargs: Any) -> Any: ...


def format_cost(usd: float, *, calls: int, unpriced: int) -> str:
    """Render a cost without ever implying that an unpriced call was free.

    The single source of truth for this, shared by the ledger and every renderer, so
    that no display path can drift into printing a bare $0.00.
    """
    if calls == 0:
        return "$0.00"
    if unpriced == 0:
        return f"${usd:,.4f}"
    if unpriced == calls:
        return "cost unavailable"
    return f"${usd:,.4f} + {unpriced} unpriced"


@dataclass
class RoleUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    unpriced_calls: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CostLedger:
    """Running token and USD totals, per role and overall."""

    by_role: dict[Role, RoleUsage] = field(
        default_factory=lambda: {role: RoleUsage() for role in Role}
    )

    def record(
        self,
        role: Role,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        usd: float | None,
    ) -> None:
        usage = self.by_role[role]
        usage.calls += 1
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        if usd is None:
            usage.unpriced_calls += 1
        else:
            usage.usd += usd

    @property
    def calls(self) -> int:
        return sum(u.calls for u in self.by_role.values())

    @property
    def prompt_tokens(self) -> int:
        return sum(u.prompt_tokens for u in self.by_role.values())

    @property
    def completion_tokens(self) -> int:
        return sum(u.completion_tokens for u in self.by_role.values())

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def usd(self) -> float:
        """USD for the calls that could be priced. Never the whole story on its own."""
        return sum(u.usd for u in self.by_role.values())

    @property
    def unpriced_calls(self) -> int:
        return sum(u.unpriced_calls for u in self.by_role.values())

    @property
    def fully_priced(self) -> bool:
        return self.unpriced_calls == 0

    def format_usd(self) -> str:
        return format_cost(self.usd, calls=self.calls, unpriced=self.unpriced_calls)

    def format_tokens(self) -> str:
        return f"{self.tokens:,} tok ({self.prompt_tokens:,} in / {self.completion_tokens:,} out)"


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    role: Role
    model: str
    prompt_tokens: int
    completion_tokens: int
    usd: float | None
    seconds: float
    raw: Any = None


class Router:
    """Routes role-tagged completion requests, meters them, and reports as it goes."""

    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        *,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._completion_fn = completion_fn
        self._semaphores = {
            role: asyncio.Semaphore(settings.concurrency_for(role)) for role in Role
        }
        self.ledger = CostLedger()
        self._budget_warned = False
        # Models this deployment refuses schema-constrained output for, learned as they
        # are called. Every later call to one of them omits `response_format` rather
        # than spending another round trip discovering the same thing.
        #
        # Per model, not per router. Access is granted model by model — measured on one
        # workspace where `claude-opus-5` was refused while `claude-sonnet-5` and
        # `claude-haiku-4-5` were allowed. A single flag meant the charter call, which
        # runs on the reasoning tier, spoke for the other two: it tripped on the first
        # node of every run and quietly dropped the schema from screening and
        # extraction, which had never been refused anything.
        self._schema_refused: set[str] = set()

    async def complete(
        self,
        role: Role,
        messages: list[dict[str, str]],
        *,
        node: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        model = self._settings.model_for(role)
        # Thinking is spent from the same ceiling as the reply, and every node's ceiling
        # here was measured against replies with no thinking in them. Added rather than
        # shared, so turning effort on cannot quietly truncate the answer it was turned on
        # to improve — and so screening's 256 tokens, which is under every thinking budget
        # there is, does not become a request the provider refuses outright.
        budget = max_tokens + self._settings.thinking_tokens_for(role)
        async with self._semaphores[role]:
            started = time.monotonic()
            for growth in range(_TRUNCATION_ATTEMPTS):
                response = await self._call_with_retries(
                    role=role,
                    node=node,
                    model=model,
                    messages=messages,
                    max_tokens=budget,
                    temperature=temperature,
                    response_format=response_format,
                )
                if not _hit_token_ceiling(response, budget):
                    break
                if growth + 1 == _TRUNCATION_ATTEMPTS:
                    self._bus.emit(
                        WarningEvent(
                            node=node,
                            message=(
                                f"{role.value} reply was still cut off at {budget} tokens "
                                f"({_shape(response)}); it will be parsed or repaired as-is"
                            ),
                        )
                    )
                    break
                # A cut-off reply is unparseable JSON however good the model was: the
                # bracket scanner needs the closing brace and there isn't one. Nothing
                # about re-asking identically would help, so the budget grows. This is
                # what took out all six curation calls in a run whose replies were fine.
                self._record(role, node, model, response, time.monotonic() - started)
                budget *= 2
                self._bus.emit(
                    WarningEvent(
                        node=node,
                        message=(
                            f"{role.value} reply was cut off by the token budget; "
                            f"retrying with {budget} ({_shape(response)})"
                        ),
                    )
                )
            seconds = time.monotonic() - started

        text, prompt_tokens, completion_tokens = _extract(response)
        usd = self._record(role, node, model, response, seconds)
        self._check_budget(node)
        return LLMResult(
            text=text,
            role=role,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd=usd,
            seconds=seconds,
            raw=response,
        )

    # --- internals ---------------------------------------------------------

    def _record(
        self, role: Role, node: str, model: str, response: Any, seconds: float
    ) -> float | None:
        """Ledger one completed call and announce it. Returns its price, if known.

        Every response reaches this, including one discarded for being cut off: the
        tokens were spent whether or not the reply was usable, and a cost report that
        omits them is the quiet kind of wrong this module exists to prevent.
        """
        _, prompt_tokens, completion_tokens = _extract(response)
        usd = self._price(role, response, prompt_tokens, completion_tokens)
        self.ledger.record(
            role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd=usd,
        )
        self._bus.emit(
            LLMCall(
                node=node,
                role=role.value,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usd=usd,
                seconds=seconds,
                total_prompt_tokens=self.ledger.prompt_tokens,
                total_completion_tokens=self.ledger.completion_tokens,
                total_usd=self.ledger.usd,
                total_calls=self.ledger.calls,
                unpriced_calls=self.ledger.unpriced_calls,
            )
        )
        return usd

    async def _call_with_retries(
        self,
        *,
        role: Role,
        node: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None,
    ) -> Any:
        attempts = max(1, self._settings.max_retries)
        last: BaseException | None = None
        attempt = 0
        # Per call, not per router. Every call in flight when the workspace first
        # refuses a schema gets the same refusal, and each one has to be allowed to
        # drop the schema and retry itself. Keyed on the router instead, only the
        # first call to handle the error would recover: the rest would find the flag
        # already flipped and re-raise a 400 that is now fixable. That is a race the
        # size of the role's concurrency, and it killed runs at the first parallel node.
        schema_retried = False
        while attempt < attempts:
            attempt += 1
            attempt_started = time.monotonic()
            try:
                return await self._invoke(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    effort=self._settings.effort_for(role),
                    timeout=self._settings.timeout_for(role),
                )
            except BaseException as exc:
                if isinstance(exc, StreamStalled) and exc.partial is not None:
                    # Abandoning a stream does not un-generate what already streamed, and
                    # the bill will say so. Ledgered here rather than in `_drain` because
                    # this is where the role and node are known, and counted in addition
                    # to the retry that follows, which is what the provider charges for.
                    self._record(role, node, model, exc.partial, time.monotonic() - attempt_started)
                if response_format is not None and not schema_retried and _refuses_schema(exc):
                    # The request was rejected before the model saw it, so this costs
                    # nothing and is not the caller's retry to spend. `schema_retried`
                    # can only flip once per call, so this cannot loop.
                    attempt -= 1
                    schema_retried = True
                    if model not in self._schema_refused:
                        # Said once per model, however many calls discover it, and said
                        # quietly. Nothing is wrong and nothing is lost: every prompt
                        # asks for JSON in words too, and `parse_model` absorbs fences,
                        # preamble and envelopes either way. A yellow banner on the
                        # first node of every single run trains people to ignore the
                        # banner, which is the opposite of what warnings are for.
                        self._schema_refused.add(model)
                        self._bus.emit(
                            Progress(
                                node=node,
                                message=(
                                    f"{model} does not accept schema-constrained output "
                                    f"here; asking for JSON in the prompt instead"
                                ),
                            )
                        )
                    continue
                if not _is_transient(exc) or attempt == attempts:
                    raise
                last = exc
                delay = self._backoff(attempt, exc)
                self._bus.emit(
                    WarningEvent(
                        node=node,
                        message=(
                            f"{role.value} call failed ({type(exc).__name__}), "
                            f"retry {attempt}/{attempts - 1} in {delay:.1f}s"
                        ),
                    )
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable") from last

    def _backoff(self, attempt: int, exc: BaseException) -> float:
        """How long to wait before retrying, in seconds.

        Two changes from textbook exponential backoff, both learned from a run that
        lost 56 of 252 screening calls to a token-per-minute limit:

        The provider's own number wins when it gives one. A limit of N tokens per 60s
        clears when the window rolls, and guessing shorter than that guarantees the
        retry fails too — the run had a ceiling of 8s against a server asking for 19.

        Jitter is partial, not full. `cap * random()` spreads a burst evenly across
        [0, cap), which means a good share of a burst retries almost immediately and
        re-triggers the same limit. Half the wait is fixed and half is jittered, so
        every retry actually waits while the burst still fans out.
        """
        cap = min(_MAX_BACKOFF, 2.0**attempt)
        floor = _requested_wait(exc)
        if floor is not None:
            # Past the named wait, not up to it: the window has to have rolled.
            return floor + min(5.0, cap) * random.random()
        return cap / 2.0 + (cap / 2.0) * random.random()

    async def _invoke(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None,
        effort: Effort | None = None,
        timeout: float,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        # Effort and temperature are mutually exclusive, and the two branches below are the
        # only requests the providers accept.
        #
        # `reasoning_effort` is LiteLLM's own vocabulary, translated per provider — a
        # thinking budget for Anthropic, the native parameter for OpenAI — so one setting
        # means the same thing across deployments. Omitted entirely when unset, which is
        # not the same as sending "none": a model that reasons by default keeps doing so.
        #
        # Sending it means sending no temperature at all. Anthropic refuses any value but 1
        # once thinking is on ("`temperature` may only be set to 1 when thinking is
        # enabled") and OpenAI's reasoning models refuse the parameter outright, so the
        # parameter goes rather than the value: 1.0 would satisfy the first and fail the
        # second. `drop_params` cannot help, because it drops what a model does not support
        # and Anthropic supports temperature perfectly well — it is the *combination* that
        # is rejected, which LiteLLM has no way to see.
        #
        # It cost a real run to find. Screening asks at temperature 0, so effort on the FAST
        # tier turned that node into a wall of 400s — 30 calls, 30 failures — and the run
        # went on to emit a bundle that validated cleanly on the papers nothing had screened.
        #
        # The trade is real and belongs to whoever sets the variable: a tier with effort on
        # no longer samples deterministically. That is the provider's rule, and no
        # configuration of ours buys back both.
        if effort is None:
            kwargs["temperature"] = temperature
        else:
            kwargs["reasoning_effort"] = effort.value
        if self._settings.api_key:
            kwargs["api_key"] = self._settings.api_key
        if self._settings.api_base:
            kwargs["api_base"] = self._settings.api_base
        if response_format is not None and model not in self._schema_refused:
            kwargs["response_format"] = response_format

        stall = self._settings.stream_stall_seconds
        if stall > 0:
            kwargs["stream"] = True
            # Without this the final chunk carries no usage and every streamed call would
            # be counted by estimate instead of by the provider's own numbers. Confirmed
            # honored on all three tiers through an Azure gateway, 2026-08-27.
            kwargs["stream_options"] = {"include_usage": True}

        if self._completion_fn is not None:
            response = await self._completion_fn(**kwargs)
        else:
            # Imported here, not at module scope: litellm costs seconds to import and
            # `okf-loremaster --help` should not pay for it.
            import litellm

            litellm.suppress_debug_info = True
            litellm.drop_params = True
            response = await litellm.acompletion(**kwargs)

        # Asked of the response rather than of `stall`, so a fake that answers with a
        # finished completion keeps taking the unstreamed path unchanged.
        if hasattr(response, "__aiter__"):
            return await self._drain(response, messages=messages, first=timeout, stall=stall)
        return response

    async def _drain(
        self, stream: Any, *, messages: list[dict[str, str]], first: float, stall: float
    ) -> Any:
        """Consume a streamed reply, abandoning it if it goes quiet mid-flight.

        Two deadlines, because the two silences mean different things. Nothing has
        arrived yet: the model may simply be thinking, which does not stream, so that
        window stays as wide as `request_timeout` ever was. Output started and then
        stopped: a healthy gap is under a second, so ten is not a slow call, it is a
        dead one, and waiting out the rest of the timeout only makes it expensive.
        """
        chunks: list[Any] = []
        iterator = stream.__aiter__()
        deadline = first
        while True:
            try:
                chunk = await asyncio.wait_for(_next_chunk(iterator), timeout=deadline)
            except TimeoutError as exc:
                await _aclose(stream)
                raise StreamStalled(
                    f"no output for {deadline:.0f}s after {len(chunks)} chunks",
                    _assemble(chunks, messages),
                ) from exc
            if chunk is _END:
                break
            chunks.append(chunk)
            deadline = stall

        assembled = _assemble(chunks, messages)
        if assembled is None:
            # A stream that closes having said nothing is a failed call, not an empty
            # reply: handing an empty completion onward would look to every parser like
            # a model that answered badly, and be retried by nobody.
            raise StreamStalled("the stream closed without producing anything")
        return assembled

    def _price(
        self, role: Role, response: Any, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        # A configured price wins, and this order is the whole point of the setting.
        #
        # `litellm.completion_cost()` reads a static JSON file shipped inside the
        # installed wheel — 1.6 MB of it, dated the day the version was cut. Nothing
        # about it is live: it does not ask the provider, and the provider does not
        # answer in dollars anyway, only in token counts. So for every model that file
        # names, it keeps returning whatever was true at release, indefinitely — and a
        # figure that is merely out of date looks exactly like one that is right.
        #
        # Consulting that file first made `OKF_LOREMASTER_PRICE_*` dead code for exactly
        # the models it matters most for: it was written for gateway deployment names
        # litellm cannot recognize, and a public model whose price has moved is the same
        # problem wearing a different hat. Understating a bill is worse than declining to
        # state one, which is already the rule the `$0.00` guard sets; this applies that
        # rule to a number that is wrong rather than missing.
        price_in, price_out = self._settings.price_for(role)
        if price_in is not None and price_out is not None:
            return (prompt_tokens / 1_000_000) * price_in + (
                completion_tokens / 1_000_000
            ) * price_out
        return self._price_from_litellm(response)

    def _price_from_litellm(self, response: Any) -> float | None:
        # Injected fakes are not real ModelResponse objects; do not ask litellm to
        # price them.
        if (
            self._completion_fn is not None
            and response is not None
            and not type(response).__module__.startswith("litellm")
        ):
            return None
        try:
            import litellm

            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:
            # Any failure here means "cannot price", which the caller handles.
            return None
        # 0.0 is litellm's answer for an unknown model, not an assertion that the
        # call was free. Treat it as no answer.
        return cost if cost > 0.0 else None

    def _check_budget(self, node: str) -> None:
        limit = self._settings.max_usd
        if limit is None or self._budget_warned:
            return
        if self.ledger.usd >= limit:
            self._budget_warned = True
            self._bus.emit(
                WarningEvent(
                    node=node,
                    message=(
                        f"soft budget reached: {self.ledger.format_usd()} "
                        f"of ${limit:,.2f} (OKF_LOREMASTER_MAX_USD)"
                    ),
                )
            )

    @property
    def over_budget(self) -> bool:
        limit = self._settings.max_usd
        return limit is not None and self.ledger.usd >= limit


def _extract(response: Any) -> tuple[str, int, int]:
    """Pull text and token counts out of a completion response, defensively.

    Providers and gateways vary in what they populate. Missing usage counts as zero
    rather than raising: losing a token count must not lose the response itself.
    """
    text = ""
    try:
        content = response.choices[0].message.content
        text = "" if content is None else str(content)
    except (AttributeError, IndexError, KeyError, TypeError):
        text = ""

    prompt_tokens = _int_attr(response, "usage", "prompt_tokens")
    completion_tokens = _int_attr(response, "usage", "completion_tokens")
    return text, prompt_tokens, completion_tokens


def _shape(response: Any) -> str:
    """How big a cut-off reply actually got, in the units the bill is in.

    "Cut off at 6144" says the budget was reached and nothing about what reached it,
    which is one guess per run at a fix. Characters are the honest measure here.
    Converting them to an estimated token count is what made this misleading the first
    time: the estimate ran about 40% under what the provider billed, the gap looked like
    tokens spent somewhere the reply does not show, and a reasoning trace was the obvious
    story to hang on it. There is no trace. Two probes at different reply lengths fitted
    the gap to `50 + 0.40 x estimate`, which is an approximation being wrong by a
    constant factor, not a model thinking. So report what was counted, not what it was
    guessed to be worth, and read a large character count as what it is: a reply that
    needs to be asked for more briefly.
    """
    text, _, completion_tokens = _extract(response)
    return f"{completion_tokens} tokens billed, {len(text):,} chars written"


def _hit_token_ceiling(response: Any, budget: int = 0) -> bool:
    """Whether the model was cut off mid-reply by `max_tokens`.

    Worth asking separately from "did it parse", because the two failures want
    opposite responses. A reply the model finished and got wrong is repaired by
    showing it the error; a reply it never finished is repaired only by room to
    finish, and re-asking with the same budget just truncates in the same place.

    Asked two ways, because the obvious one goes blind on the schema-constrained
    path. Measured 2026-08-04 on the balanced model, same prompt, three budgets:

        with a response_format     finish_reason='stop'    64/64, 256/256, 1024/1024
        without one                finish_reason='length'  256/256

    A schema is delivered as a forced tool call, and a tool call truncated mid
    arguments is reported as a clean stop with the whole budget spent and the
    partial JSON thrown away — `content` comes back as `{}` or empty. So the
    retry that exists precisely to rescue cut-off curation calls stopped firing
    the moment schemas were enabled, and the node just failed instead.

    Spending the entire budget is the signal that survives that. It can in
    principle be a reply that ended exactly on the boundary; the cost of being
    wrong is one retry with more room, against a topic curated on nothing.
    """
    try:
        reason = response.choices[0].finish_reason
    except (AttributeError, IndexError, KeyError, TypeError):
        try:
            reason = response["choices"][0]["finish_reason"]
        except (IndexError, KeyError, TypeError):
            reason = ""
    if str(reason) in ("length", "max_tokens"):
        return True
    return budget > 0 and _int_attr(response, "usage", "completion_tokens") >= budget


def _int_attr(response: Any, container: str, name: str) -> int:
    try:
        holder = getattr(response, container, None)
        if holder is None and isinstance(response, dict):
            holder = response.get(container)
        if holder is None:
            return 0
        value = getattr(holder, name, None)
        if value is None and isinstance(holder, dict):
            value = holder.get(name)
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
