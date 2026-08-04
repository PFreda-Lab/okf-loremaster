"""The renderer: what a human actually sees, and the fallback when nobody is watching."""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from rich.console import Console

from okf_loremaster.events import (
    EventBus,
    LLMCall,
    NodeFinished,
    NodeStarted,
    RunFinished,
    RunStarted,
    WarningEvent,
)
from okf_loremaster.ui.plain import PlainRenderer, rich_enabled


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    # no_color and a fixed width keep assertions about text stable.
    return Console(file=buffer, width=120, no_color=True, highlight=False), buffer


async def _render(events: list[Any]) -> str:
    console, buffer = _console()
    bus = EventBus()
    renderer = PlainRenderer(bus, console=console, live=False)
    task = asyncio.create_task(renderer.run())
    for event in events:
        bus.emit(event)
    bus.close()
    await task
    return buffer.getvalue()


async def test_unpriced_run_never_renders_as_free() -> None:
    output = await _render(
        [
            RunStarted(run_id="r1", prompt="a task"),
            NodeStarted(node="extract"),
            LLMCall(
                node="extract",
                role="reasoning",
                model="gateway/deployment",
                prompt_tokens=100,
                completion_tokens=50,
                usd=None,
                seconds=0.4,
                total_prompt_tokens=100,
                total_completion_tokens=50,
                total_usd=0.0,
                total_calls=1,
                unpriced_calls=1,
            ),
            NodeFinished(node="extract", summary="1 concept"),
            RunFinished(run_id="r1", ok=True),
        ]
    )

    assert "cost unavailable" in output
    assert "$0.00" not in output
    assert "could not be priced" in output
    assert "OKF_LOREMASTER_PRICE" in output


async def test_priced_run_shows_the_figure() -> None:
    output = await _render(
        [
            RunStarted(run_id="r2", prompt="a task"),
            LLMCall(
                node="extract",
                role="reasoning",
                model="claude-opus-5",
                prompt_tokens=1000,
                completion_tokens=500,
                usd=0.0525,
                seconds=0.4,
                total_prompt_tokens=1000,
                total_completion_tokens=500,
                total_usd=0.0525,
                total_calls=1,
                unpriced_calls=0,
            ),
            RunFinished(run_id="r2", ok=True),
        ]
    )

    assert "$0.0525" in output
    assert "cost unavailable" not in output
    assert "1,500" in output


async def test_warnings_and_errors_are_counted_in_the_summary() -> None:
    output = await _render(
        [
            RunStarted(run_id="r3", prompt="t"),
            WarningEvent(node="screen", message="retry 1/3"),
            WarningEvent(node="screen", message="retry 2/3"),
            RunFinished(run_id="r3", ok=True),
        ]
    )
    assert "retry 1/3" in output
    assert "warnings" in output


async def test_non_live_mode_reports_node_boundaries() -> None:
    """With no terminal there is no meter, so progress has to appear in the log itself."""
    output = await _render(
        [
            NodeStarted(node="charter"),
            NodeFinished(node="charter", summary="3 topics", seconds=1.2),
        ]
    )
    assert "charter" in output
    assert "3 topics" in output


async def test_renderer_subscribes_before_run_starts() -> None:
    """Events emitted between wiring and starting the task must not be dropped."""
    console, buffer = _console()
    bus = EventBus()
    renderer = PlainRenderer(bus, console=console, live=False)

    bus.emit(NodeFinished(node="early", summary="emitted before run()"))
    task = asyncio.create_task(renderer.run())
    bus.close()
    await task

    assert "emitted before run()" in buffer.getvalue()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"NO_COLOR": "1"}, False),
        ({"CI": "true"}, False),
        ({"TERM": "dumb"}, False),
        ({"TERM": "xterm-256color"}, True),
    ],
)
def test_live_display_opts_out_of_hostile_environments(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: bool
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert rich_enabled(_Tty()) is expected


def test_live_display_off_without_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    assert rich_enabled(io.StringIO()) is False
