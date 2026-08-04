"""LiteLLM-backed router: three role-bound models, retries, and honest cost accounting.

The cost accounting is the delicate part. `litellm.completion_cost()` returns 0.0 for a
model it does not recognize rather than raising, and a model reached through a gateway
or under a custom deployment name is routinely absent from its price map. A run that
reports $0.00 is therefore indistinguishable from a run that was genuinely free, which
is the worst possible failure: it looks like good news.

So pricing goes through three stages and the third is explicit ignorance:
  1. LiteLLM's own price map.
  2. `OKF_LOREMASTER_PRICE_<ROLE>_IN` / `_OUT`, in USD per 1M tokens.
  3. `usd = None` — the call is counted in tokens and reported as unpriced.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from okf_loremaster.config import Role, Settings
from okf_loremaster.events import EventBus, LLMCall, WarningEvent

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
        async with self._semaphores[role]:
            started = time.monotonic()
            response = await self._call_with_retries(
                role=role,
                node=node,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            seconds = time.monotonic() - started

        text, prompt_tokens, completion_tokens = _extract(response)
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
        for attempt in range(1, attempts + 1):
            try:
                return await self._invoke(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            except BaseException as exc:
                if not _is_transient(exc) or attempt == attempts:
                    raise
                last = exc
                # Exponential backoff with full jitter, so a burst of parallel
                # calls hitting the same rate limit does not retry in lockstep.
                delay = min(30.0, 2.0**attempt) * random.random()
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

    async def _invoke(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._settings.request_timeout,
        }
        if self._settings.api_key:
            kwargs["api_key"] = self._settings.api_key
        if self._settings.api_base:
            kwargs["api_base"] = self._settings.api_base
        if response_format is not None:
            kwargs["response_format"] = response_format

        if self._completion_fn is not None:
            return await self._completion_fn(**kwargs)

        # Imported here, not at module scope: litellm costs seconds to import and
        # `okf-loremaster --help` should not pay for it.
        import litellm

        litellm.suppress_debug_info = True
        litellm.drop_params = True
        return await litellm.acompletion(**kwargs)

    def _price(
        self, role: Role, response: Any, prompt_tokens: int, completion_tokens: int
    ) -> float | None:
        from_litellm = self._price_from_litellm(response)
        if from_litellm is not None:
            return from_litellm
        price_in, price_out = self._settings.price_for(role)
        if price_in is None or price_out is None:
            return None
        return (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out

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
