"""A scripted run against a fake model, exercising the plumbing without a network.

This is the visible proof for the parts of the system that are easy to get quietly
wrong: that nodes emit rather than print, that the live meter tracks a run, that
retries surface as warnings, and above all that an unpriced model reports
"cost unavailable" rather than $0.00.

It deliberately runs twice — once with no price overrides configured and once with
them — because the difference between those two outputs is the whole point.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rich.console import Console

from okf_loremaster.config import Role, Settings
from okf_loremaster.events import (
    EventBus,
    NodeFinished,
    NodeStarted,
    Progress,
    RunFinished,
    RunStarted,
)
from okf_loremaster.llm.fake import FakeCompletion
from okf_loremaster.llm.router import Router
from okf_loremaster.ui.plain import PlainRenderer

# A name no price map will recognize — the situation behind any gateway or custom
# deployment, and the one that produces a misleading $0.00 if unhandled.
FAKE_MODEL = "gateway/custom-deployment-name"

# Illustrative USD per 1M tokens, used only to show the override path working.
_DEMO_PRICES = {
    "price_fast_in": 0.80,
    "price_fast_out": 4.00,
    "price_balanced_in": 3.00,
    "price_balanced_out": 15.00,
    "price_reasoning_in": 15.00,
    "price_reasoning_out": 75.00,
}


def _settings(*, priced: bool) -> Settings:
    """Settings for a pass. Init arguments outrank the environment in pydantic-settings,
    so this is unaffected by whatever the user has in their own .env."""
    overrides: dict[str, Any] = {
        "model_fast": FAKE_MODEL,
        "model_balanced": FAKE_MODEL,
        "model_reasoning": FAKE_MODEL,
        "api_key": "selftest",
        "api_base": None,
        "max_usd": None,
        "concurrency_fast": 4,
    }
    overrides |= _DEMO_PRICES if priced else dict.fromkeys(_DEMO_PRICES, None)
    return Settings(**overrides)


async def _node(
    router: Router,
    bus: EventBus,
    *,
    name: str,
    role: Role,
    calls: int,
    summary: str,
) -> None:
    bus.emit(NodeStarted(node=name))
    started = time.monotonic()

    async def one(index: int) -> None:
        await router.complete(
            role,
            [{"role": "user", "content": f"{name} item {index}: " + "context " * 40}],
            node=name,
        )
        bus.emit(Progress(node=name, message=f"item {index + 1}", current=index + 1, total=calls))

    await asyncio.gather(*(one(i) for i in range(calls)))
    bus.emit(NodeFinished(node=name, summary=summary, seconds=time.monotonic() - started))


async def _one_pass(*, priced: bool, live: bool | None, verbose: int, console: Console) -> str:
    settings = _settings(priced=priced)
    bus = EventBus()
    renderer = PlainRenderer(bus, console=console, live=live, verbose=verbose)
    task = asyncio.create_task(renderer.run())

    # The second node injects two transient failures, so the retry path and its
    # warning events are exercised rather than assumed.
    router = Router(settings, bus, completion_fn=FakeCompletion(replies=("scripted reply",)))
    flaky = Router(
        settings,
        bus,
        completion_fn=FakeCompletion(replies=("scripted reply",), fail_times=2),
    )
    flaky.ledger = router.ledger  # one ledger for the whole pass

    label = "with price overrides" if priced else "no price overrides"
    bus.emit(RunStarted(run_id=f"selftest-{'priced' if priced else 'unpriced'}", prompt=label))

    await _node(router, bus, name="charter", role=Role.REASONING, calls=1, summary="3 topics")
    await _node(flaky, bus, name="screen", role=Role.FAST, calls=6, summary="6 screened")
    await _node(router, bus, name="extract", role=Role.REASONING, calls=3, summary="3 concepts")

    rendered = router.ledger.format_usd()
    bus.emit(
        RunFinished(
            run_id=f"selftest-{'priced' if priced else 'unpriced'}",
            ok=True,
            summary=label,
        )
    )
    bus.close()
    await task
    return rendered


async def run_selftest(
    *, live: bool | None = None, verbose: int = 0, console: Console | None = None
) -> int:
    """Run both passes and check the cost-reporting invariant. Returns an exit code."""
    out = console if console is not None else Console(stderr=True)

    unpriced = await _one_pass(priced=False, live=live, verbose=verbose, console=out)
    out.print()
    priced = await _one_pass(priced=True, live=live, verbose=verbose, console=out)
    out.print()

    problems: list[str] = []
    if unpriced != "cost unavailable":
        problems.append(
            f"unpriced run reported {unpriced!r}; expected 'cost unavailable'. "
            "A model with no known price must never render as a dollar figure."
        )
    if not priced.startswith("$") or priced == "$0.00":
        problems.append(f"priced run reported {priced!r}; expected a non-zero USD figure.")

    if problems:
        for problem in problems:
            out.print(f"[red]FAIL[/red] {problem}")
        return 1

    out.print(
        f"[green]PASS[/green] unpriced -> [bold]{unpriced}[/bold]   priced -> [bold]{priced}[/bold]"
    )
    return 0
