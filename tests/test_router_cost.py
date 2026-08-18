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
from okf_loremaster.events import EventBus, LLMCall, Progress, WarningEvent
from okf_loremaster.llm.fake import FakeChoice, FakeCompletion, FakeMessage, FakeResponse, FakeUsage
from okf_loremaster.llm.router import Router, format_cost

MESSAGES = [{"role": "user", "content": "a question worth about twenty tokens or so"}]


def _router(
    settings_factory: Any, *, completion: Any = None, **overrides: Any
) -> tuple[Router, EventBus]:
    settings = settings_factory(
        **{
            "model_fast": "gateway/deployment",
            "model_balanced": "gateway/deployment",
            "model_reasoning": "gateway/deployment",
            "api_key": "k",
            **overrides,
        }
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

    result = await router.complete(Role.REASONING, MESSAGES, node="extract")

    assert result.usd is None, "an unpriceable call must not report a number"
    assert router.ledger.unpriced_calls == 1
    assert router.ledger.usd == 0.0
    assert router.ledger.format_usd() == "cost unavailable"

    call = next(e for e in _events(bus, queue) if isinstance(e, LLMCall))
    assert call.usd is None
    assert call.prompt_tokens > 0
    assert call.completion_tokens > 0


async def test_price_override_produces_a_real_figure(settings_factory: Any) -> None:
    router, _ = _router(settings_factory, price_reasoning_in=15.0, price_reasoning_out=75.0)

    result = await router.complete(Role.REASONING, MESSAGES, node="extract")

    assert result.usd is not None
    assert result.usd > 0.0
    expected = (result.prompt_tokens / 1e6) * 15.0 + (result.completion_tokens / 1e6) * 75.0
    assert result.usd == pytest.approx(expected)
    assert router.ledger.format_usd().startswith("$")
    assert router.ledger.fully_priced


async def test_half_priced_run_is_reported_as_such(settings_factory: Any) -> None:
    """FAST priced, REASONING not: the total must admit that part of it is missing."""
    router, _ = _router(settings_factory, price_fast_in=0.8, price_fast_out=4.0)

    await router.complete(Role.FAST, MESSAGES, node="screen")
    await router.complete(Role.REASONING, MESSAGES, node="extract")

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


# --- schema-constrained output ---------------------------------------------

SCHEMA = {"type": "json_schema", "json_schema": {"name": "t", "schema": {"type": "object"}}}


def _refusing_completion(
    replies: tuple[str, ...] = ("{}",), *, only: str | None = None
) -> FakeCompletion:
    """A provider that rejects `response_format` the way a gated workspace does.

    The wording is the provider's own, because that string is what the router matches
    on. A test that invented a friendlier message would pass while the real one failed.

    `only` refuses a single model and allows the rest, which is what access granted
    model by model looks like — and what a router-wide flag gets wrong.
    """

    def reply(kwargs: dict[str, Any]) -> str:
        refused = only is None or kwargs.get("model") == only
        if "response_format" in kwargs and refused:
            raise ValueError(
                'AnthropicException - {"type":"error","error":{"type":'
                '"invalid_request_error","message":"structured_outputs not supported '
                'in your workspace."}}'
            )
        return replies[0]

    return FakeCompletion(replies=reply)


async def test_a_workspace_without_schema_support_falls_back_rather_than_failing(
    settings_factory: Any,
) -> None:
    """The failure this replaces killed a run one node deep, after paying for the first.

    `drop_params` cannot catch it: LiteLLM's capability map answers for the model, and
    the gate is on the account. Every prompt already says "Return a single JSON object
    and nothing else", and `parse_model` never trusted the constraint, so the reply is
    still usable without it.
    """
    completion = _refusing_completion()
    router, bus = _router(settings_factory, completion=completion)
    queue = bus.subscribe()

    result = await router.complete(Role.FAST, MESSAGES, node="search", response_format=SCHEMA)

    assert result.text == "{}"
    assert completion.call_count == 2, "one rejected call, then one without the schema"
    assert "response_format" not in completion.calls[1]

    events = _events(bus, queue)
    # Not a warning. Nothing is wrong, nothing is lost, and it recurs on the first node
    # of every run against a deployment that gates this — a yellow banner every time
    # teaches people to stop reading yellow banners.
    assert not [e for e in events if isinstance(e, WarningEvent)]
    notes = [e for e in events if isinstance(e, Progress) and "schema-constrained" in e.message]
    assert len(notes) == 1, "a silent downgrade is still the thing to avoid"
    assert "gateway/deployment" in notes[0].message, "the note must name the model refused"


async def test_a_grammar_that_will_not_compile_is_a_refusal_in_other_words(
    settings_factory: Any,
) -> None:
    """The same event, worded so that it reads like a timeout rather than a decline.

    Verbatim from an Azure AI Foundry Anthropic deployment (2026-08-17), where it ended
    three consecutive runs at the charter node — the first one — on two different
    models. A provider compiles the schema into a decoding grammar before the model sees
    anything; one it cannot compile in time is one it will not honor, and the schema
    does not get smaller by waiting. Matching only "structured outputs" left the run
    dead on the first node with a bundle nobody could build.
    """

    def reply(kwargs: dict[str, Any]) -> str:
        if "response_format" in kwargs:
            raise ValueError(
                'AnthropicException - {"type":"error","error":{"type":'
                '"invalid_request_error","message":"Grammar compilation timed out."}}'
            )
        return "{}"

    completion = FakeCompletion(replies=reply)
    router, _ = _router(settings_factory, completion=completion)

    result = await router.complete(Role.REASONING, MESSAGES, node="charter", response_format=SCHEMA)

    assert result.text == "{}"
    assert completion.call_count == 2, "one rejected call, then one without the schema"
    assert "response_format" not in completion.calls[1]


async def test_the_refusal_is_learned_once_not_rediscovered_per_call(
    settings_factory: Any,
) -> None:
    """Otherwise every one of a few hundred extractions pays for a rejected round trip."""
    completion = _refusing_completion()
    router, _ = _router(settings_factory, completion=completion)

    for _ in range(3):
        await router.complete(Role.FAST, MESSAGES, node="extract", response_format=SCHEMA)

    assert completion.call_count == 4, "one rejection, then three clean calls"
    assert all("response_format" not in call for call in completion.calls[1:])


async def test_one_model_being_refused_does_not_speak_for_the_others(
    settings_factory: Any,
) -> None:
    """Access is granted model by model, so the memory has to be too.

    Measured on a live workspace: `claude-opus-5` was refused while `claude-sonnet-5`
    and `claude-haiku-4-5` were allowed. With one flag for the router, the charter call
    — which runs on the reasoning tier, first node of the run — tripped it and every
    later call dropped a schema it would have been given. Screening and extraction, the
    two nodes that make hundreds of calls, spent whole runs unconstrained because of a
    model neither of them uses.
    """
    completion = _refusing_completion(only="gateway/deep")
    router, _ = _router(
        settings_factory,
        completion=completion,
        model_reasoning="gateway/deep",
        model_balanced="gateway/mid",
        model_fast="gateway/quick",
    )

    await router.complete(Role.REASONING, MESSAGES, node="charter", response_format=SCHEMA)
    await router.complete(Role.BALANCED, MESSAGES, node="extract", response_format=SCHEMA)
    await router.complete(Role.FAST, MESSAGES, node="screen", response_format=SCHEMA)

    refused, balanced, fast = completion.calls[1], completion.calls[2], completion.calls[3]
    assert "response_format" not in refused, "the refused model kept being asked"
    assert "response_format" in balanced, "balanced lost a schema it was never refused"
    assert "response_format" in fast, "fast lost a schema it was never refused"


async def test_the_fallback_does_not_spend_a_retry(settings_factory: Any) -> None:
    """The request never reached the model, so it is not one of the caller's attempts.

    With `max_retries=1` a fallback that consumed an attempt would raise instead of
    recovering — which is the whole failure, moved one line later.
    """
    completion = _refusing_completion()
    router, _ = _router(settings_factory, completion=completion, max_retries=1)

    result = await router.complete(Role.FAST, MESSAGES, node="charter", response_format=SCHEMA)

    assert result.text == "{}"


async def test_an_unrelated_bad_request_still_fails_immediately(settings_factory: Any) -> None:
    """The match is narrow on purpose. A wider one would drop the schema on any 400."""
    completion = FakeCompletion(fail_times=99, failure=ValueError)
    router, _ = _router(settings_factory, completion=completion, max_retries=4)

    with pytest.raises(ValueError):
        await router.complete(Role.FAST, MESSAGES, node="screen", response_format=SCHEMA)
    assert completion.call_count == 1


async def test_every_call_in_flight_survives_the_first_refusal(settings_factory: Any) -> None:
    """The refusal arrives at all of them at once, so all of them must recover.

    This is the bug that killed a live run: the "have we learned it yet" flag lived on
    the router, and the first call to handle the rejection flipped it. Every sibling
    then found the flag already set, skipped the fallback, and re-raised a
    `BadRequestError` that had just become fixable. Learning it once is about not
    paying for the discovery twice — it was never a reason to let the other calls die.
    """
    completion = _refusing_completion()
    router, bus = _router(settings_factory, completion=completion, concurrency_fast=8)
    queue = bus.subscribe()

    results = await asyncio.gather(
        *(
            router.complete(Role.FAST, MESSAGES, node="screen", response_format=SCHEMA)
            for _ in range(8)
        )
    )

    assert [r.text for r in results] == ["{}"] * 8, "not one of them may be lost"
    notes = [
        e
        for e in _events(bus, queue)
        if isinstance(e, Progress) and "schema-constrained" in e.message
    ]
    assert len(notes) == 1, "announced once per model, however many discover it"


# --- truncation ------------------------------------------------------------


async def test_a_reply_cut_off_by_the_budget_is_retried_with_room(
    settings_factory: Any,
) -> None:
    """Truncated JSON never parses, so re-asking identically cannot help.

    A run lost all six of its curation calls to this: the replies were fine, the
    budget was a third of what they needed, and every one came back unterminated.
    """
    long_reply = "x" * 400  # ~100 tokens by the fake's estimate
    completion = FakeCompletion(replies=(long_reply,))
    router, bus = _router(settings_factory, completion=completion)
    queue = bus.subscribe()

    await router.complete(Role.BALANCED, MESSAGES, node="curate", max_tokens=64)

    assert [call["max_tokens"] for call in completion.calls] == [64, 128]
    messages = [e.message for e in _events(bus, queue) if isinstance(e, WarningEvent)]
    assert any("cut off" in m for m in messages)


async def test_the_tokens_of_a_discarded_truncated_reply_are_still_billed(
    settings_factory: Any,
) -> None:
    """It was paid for whether or not it was usable, and this module is about honesty."""
    completion = FakeCompletion(replies=("x" * 400,))
    router, _ = _router(settings_factory, completion=completion)

    await router.complete(Role.BALANCED, MESSAGES, node="curate", max_tokens=64)

    assert router.ledger.calls == 2, "the abandoned attempt is a call that happened"


async def test_a_schema_constrained_reply_cut_off_is_also_retried_with_room(
    settings_factory: Any,
) -> None:
    """The same rescue, on the path where the obvious signal lies.

    A schema is sent as a forced tool call, and a tool call truncated mid-arguments is
    reported as `finish_reason='stop'` with `content='{}'` and the budget fully spent —
    so the retry that exists for exactly this stopped firing the moment schemas were
    enabled. Two topics in one run were curated from the screener's fallback because of
    it, logged as a schema mismatch, which is the wrong problem with the wrong fix.

    Spending the whole budget is the signal that survives a provider lying about why
    it stopped.
    """
    completion = FakeCompletion(replies=("x" * 400,))
    router, bus = _router(settings_factory, completion=completion)
    queue = bus.subscribe()

    await router.complete(
        Role.BALANCED,
        MESSAGES,
        node="curate",
        max_tokens=64,
        response_format={"type": "json_schema", "json_schema": {"name": "t"}},
    )

    assert [call["max_tokens"] for call in completion.calls] == [64, 128]
    messages = [e.message for e in _events(bus, queue) if isinstance(e, WarningEvent)]
    assert any("cut off" in m for m in messages)


async def test_a_reply_that_fits_is_not_retried(settings_factory: Any) -> None:
    """The common case must not pay for the rare one."""
    completion = FakeCompletion(replies=("{}",))
    router, _ = _router(settings_factory, completion=completion)

    await router.complete(Role.BALANCED, MESSAGES, node="curate", max_tokens=64)

    assert completion.call_count == 1


# --- backoff ---------------------------------------------------------------


def test_a_stated_wait_is_honored_rather_than_guessed_under() -> None:
    """A token-per-minute limit clears on the provider's clock, not on ours.

    The run this comes from backed off a maximum of 8s against a provider asking for
    19, so all four attempts fell inside one window and 56 of 252 papers were lost.
    """
    from okf_loremaster.llm.router import _requested_wait

    exc = ValueError(
        'AnthropicException - {"error":{"code":"RateLimitReached","message":"Rate limit '
        'of 250000 per 60s exceeded for UserByModelByMinuteUncachedInputTokens. Please '
        'wait 19 seconds before retrying."}}'
    )

    assert _requested_wait(exc) == 19.0


def test_a_failure_that_names_no_wait_falls_back_to_exponential() -> None:
    from okf_loremaster.llm.router import _requested_wait

    assert _requested_wait(ConnectionError("connection reset by peer")) is None


def test_backoff_always_waits(settings_factory: Any) -> None:
    """Full jitter over [0, cap) retries a good share of a burst almost immediately.

    Against a rate limit that is the one thing that cannot work, because the burst is
    what tripped it. Half the delay is fixed so that every retry actually waits.
    """
    router, _ = _router(settings_factory)
    exc = ConnectionError("no hint here")

    delays = [router._backoff(attempt, exc) for attempt in range(1, 5) for _ in range(50)]

    assert min(delays) > 0.0


def test_a_stated_wait_is_cleared_before_retrying(settings_factory: Any) -> None:
    """Past the named wait, not up to it: the window has to have actually rolled."""
    router, _ = _router(settings_factory)
    exc = ValueError("Please wait 19 seconds before retrying.")

    assert all(router._backoff(1, exc) >= 19.0 for _ in range(50))


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
    await router.complete(Role.REASONING, MESSAGES, node="extract")

    assert router.ledger.by_role[Role.FAST].calls == 2
    assert router.ledger.by_role[Role.REASONING].calls == 1
    assert router.ledger.by_role[Role.BALANCED].calls == 0
    assert router.ledger.calls == 3
    assert router.ledger.tokens == router.ledger.prompt_tokens + router.ledger.completion_tokens
