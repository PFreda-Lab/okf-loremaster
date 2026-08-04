"""Router behavior, with cost accounting as the main event.

The invariant under test throughout: a call that cannot be priced is reported as
unpriced, never as free. LiteLLM answers 0.0 for models it does not recognize, so
"$0.00" and "we have no idea" are the same value at the source and must not be the
same value by the time a human sees it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from okf_loremaster.config import Role
from okf_loremaster.events import EventBus, LLMCall, WarningEvent
from okf_loremaster.llm.fake import FakeChoice, FakeCompletion, FakeMessage, FakeResponse, FakeUsage
from okf_loremaster.llm.router import Router, format_cost

MESSAGES = [{"role": "user", "content": "a question worth about twenty tokens or so"}]


def _router(
    settings_factory: Any, *, completion: Any = None, **overrides: Any
) -> tuple[Router, EventBus]:
    settings = settings_factory(
        model_fast="gateway/deployment",
        model_mid="gateway/deployment",
        model_deep="gateway/deployment",
        api_key="k",
        **overrides,
    )
    bus = EventBus()
    return Router(settings, bus, completion_fn=completion or FakeCompletion()), bus


def _events(bus: EventBus, queue: asyncio.Queue[Any]) -> list[Any]:
    bus.close()
    drained: list[Any] = []
    while True:
        item = queue.get_nowait()
        if item is None:
            return drained
        drained.append(item)


# --- pricing ---------------------------------------------------------------


async def test_unknown_model_is_unpriced_not_free(settings_factory: Any) -> None:
    router, bus = _router(settings_factory)
    queue = bus.subscribe()

    result = await router.complete(Role.DEEP, MESSAGES, node="extract")

    assert result.usd is None, "an unpriceable call must not report a number"
    assert router.ledger.unpriced_calls == 1
    assert router.ledger.usd == 0.0
    assert router.ledger.format_usd() == "cost unavailable"

    call = next(e for e in _events(bus, queue) if isinstance(e, LLMCall))
    assert call.usd is None
    assert call.prompt_tokens > 0
    assert call.completion_tokens > 0


async def test_price_override_produces_a_real_figure(settings_factory: Any) -> None:
    router, _ = _router(settings_factory, price_deep_in=15.0, price_deep_out=75.0)

    result = await router.complete(Role.DEEP, MESSAGES, node="extract")

    assert result.usd is not None
    assert result.usd > 0.0
    expected = (result.prompt_tokens / 1e6) * 15.0 + (result.completion_tokens / 1e6) * 75.0
    assert result.usd == pytest.approx(expected)
    assert router.ledger.format_usd().startswith("$")
    assert router.ledger.fully_priced


async def test_half_priced_run_is_reported_as_such(settings_factory: Any) -> None:
    """FAST priced, DEEP not: the total must admit that part of it is missing."""
    router, _ = _router(settings_factory, price_fast_in=0.8, price_fast_out=4.0)

    await router.complete(Role.FAST, MESSAGES, node="screen")
    await router.complete(Role.DEEP, MESSAGES, node="extract")

    rendered = router.ledger.format_usd()
    assert rendered.startswith("$")
    assert "1 unpriced" in rendered
    assert not router.ledger.fully_priced


@pytest.mark.parametrize(
    ("usd", "calls", "unpriced", "expected"),
    [
        (0.0, 0, 0, "$0.00"),
        (0.0, 3, 3, "cost unavailable"),
        (1.5, 3, 0, "$1.5000"),
        (1.5, 3, 1, "$1.5000 + 1 unpriced"),
    ],
)
def test_format_cost(usd: float, calls: int, unpriced: int, expected: str) -> None:
    assert format_cost(usd, calls=calls, unpriced=unpriced) == expected


def test_zero_dollars_only_ever_means_zero_calls() -> None:
    """The one string that must never appear for work that actually happened."""
    for calls in range(1, 5):
        for unpriced in range(1, calls + 1):
            assert format_cost(0.0, calls=calls, unpriced=unpriced) != "$0.00"


# --- retries ---------------------------------------------------------------


async def test_transient_failures_are_retried_and_surfaced(settings_factory: Any) -> None:
    completion = FakeCompletion(replies=("recovered",), fail_times=2)
    router, bus = _router(settings_factory, completion=completion, max_retries=4)
    queue = bus.subscribe()

    result = await router.complete(Role.FAST, MESSAGES, node="screen")

    assert result.text == "recovered"
    assert completion.call_count == 3
    warnings = [e for e in _events(bus, queue) if isinstance(e, WarningEvent)]
    assert len(warnings) == 2, "each retry must be visible, not silent"
    assert "retry" in warnings[0].message


async def test_permanent_failure_is_not_retried(settings_factory: Any) -> None:
    """Retrying a bad request just burns rate limit and delays the real error."""
    completion = FakeCompletion(fail_times=99, failure=ValueError)
    router, _ = _router(settings_factory, completion=completion, max_retries=4)

    with pytest.raises(ValueError):
        await router.complete(Role.FAST, MESSAGES, node="screen")
    assert completion.call_count == 1


async def test_retries_are_exhausted_then_raised(settings_factory: Any) -> None:
    completion = FakeCompletion(fail_times=99, failure=ConnectionError)
    router, _ = _router(settings_factory, completion=completion, max_retries=3)

    with pytest.raises(ConnectionError):
        await router.complete(Role.FAST, MESSAGES, node="screen")
    assert completion.call_count == 3


# --- concurrency and budget ------------------------------------------------


async def test_per_role_concurrency_is_capped(settings_factory: Any) -> None:
    peak = 0
    current = 0

    async def probe(**kwargs: Any) -> FakeResponse:
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content="ok"))],
            usage=FakeUsage(prompt_tokens=10, completion_tokens=5),
        )

    router, _ = _router(settings_factory, completion=probe, concurrency_fast=3)
    await asyncio.gather(
        *(router.complete(Role.FAST, MESSAGES, node="screen") for _ in range(12))
    )

    assert peak <= 3
    assert router.ledger.calls == 12


async def test_soft_budget_warns_once(settings_factory: Any) -> None:
    router, bus = _router(
        settings_factory,
        price_fast_in=1000.0,
        price_fast_out=1000.0,
        max_usd=0.000001,
    )
    queue = bus.subscribe()

    for _ in range(3):
        await router.complete(Role.FAST, MESSAGES, node="screen")

    warnings = [e for e in _events(bus, queue) if isinstance(e, WarningEvent)]
    assert len(warnings) == 1, "the budget warning must not repeat on every subsequent call"
    assert "budget" in warnings[0].message
    assert router.over_budget


async def test_ledger_totals_split_by_role(settings_factory: Any) -> None:
    router, _ = _router(settings_factory)

    await router.complete(Role.FAST, MESSAGES, node="screen")
    await router.complete(Role.FAST, MESSAGES, node="screen")
    await router.complete(Role.DEEP, MESSAGES, node="extract")

    assert router.ledger.by_role[Role.FAST].calls == 2
    assert router.ledger.by_role[Role.DEEP].calls == 1
    assert router.ledger.by_role[Role.MID].calls == 0
    assert router.ledger.calls == 3
    assert router.ledger.tokens == router.ledger.prompt_tokens + router.ledger.completion_tokens
