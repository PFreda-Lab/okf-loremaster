"""Rich console renderer — the default UI.

Subscribes to the event bus and shows a scrolling log with a live token/cost meter
pinned underneath it. Falls back to plain sequential lines when there is no terminal
to drive: a redirected run must still produce readable output, and a live region
written to a pipe produces neither.

Everything is written to stderr so that stdout stays clean for piped data.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TextIO

from rich.console import Console
from rich.table import Table
from rich.text import Text

from okf_loremaster.events import (
    ErrorEvent,
    Event,
    EventBus,
    LLMCall,
    NodeFinished,
    NodeStarted,
    Progress,
    RunFinished,
    RunStarted,
    WarningEvent,
)
from okf_loremaster.llm.router import format_cost


def rich_enabled(stream: TextIO | None = None) -> bool:
    """Whether a live-updating display is appropriate.

    Checked in this order because each is a stronger signal than the next: an explicit
    opt-out, a CI runner, a terminal that cannot render, then finally whether there is
    a terminal at all.
    """
    target = stream if stream is not None else sys.stderr
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("CI"):
        return False
    if os.environ.get("TERM", "").lower() in {"", "dumb"}:
        return False
    isatty = getattr(target, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


class PlainRenderer:
    """Renders an event stream to a console.

    Subscribes at construction, not at `run()`, so that events emitted between wiring
    and starting the consumer task are not lost.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        console: Console | None = None,
        live: bool | None = None,
        verbose: int = 0,
    ) -> None:
        self._queue = bus.subscribe()
        self._console = console if console is not None else Console(stderr=True)
        self._live = rich_enabled() if live is None else live
        self._verbose = verbose

        self._node = ""
        self._calls = 0
        self._tokens = 0
        self._usd = 0.0
        self._unpriced = 0
        self._warnings = 0
        self._errors = 0
        self._progress: tuple[int, int] | None = None

    # --- public ------------------------------------------------------------

    async def run(self) -> None:
        """Consume events until the bus closes."""
        if self._live:
            from rich.live import Live

            with Live(
                self._meter(),
                console=self._console,
                refresh_per_second=8,
                transient=True,
            ) as live:
                await self._consume(live.console.print, live.update)
        else:
            await self._consume(self._console.print, None)

    # --- internals ---------------------------------------------------------

    async def _consume(
        self,
        emit_line: object,
        update_meter: object,
    ) -> None:
        # Typed loosely because Rich's print and update signatures differ; both are
        # only ever called with a single renderable here.
        while True:
            event = await self._queue.get()
            if event is None:
                break
            for line in self._handle(event):
                emit_line(line)  # type: ignore[operator]
            if update_meter is not None:
                update_meter(self._meter())  # type: ignore[operator]

    def _handle(self, event: Event) -> list[Text | Table]:
        match event:
            case RunStarted():
                mode = " [dry run]" if event.dry_run else ""
                return [
                    Text.from_markup(
                        f"[bold]run {event.run_id}[/bold]{mode}  {_truncate(event.prompt, 70)}"
                    )
                ]

            case NodeStarted():
                self._node = event.node
                self._progress = None
                if not self._live:
                    return [Text.from_markup(f"[dim]->[/dim] {event.node}")]
                return []

            case NodeFinished():
                self._progress = None
                detail = f"  {event.summary}" if event.summary else ""
                line = Text.from_markup(
                    f"[green]OK[/green] {event.node}{detail}  [dim]{event.seconds:.1f}s[/dim]"
                )
                if not self._live:
                    return [line, Text.from_markup(f"[dim]   {self._meter_text()}[/dim]")]
                return [line]

            case Progress():
                self._progress = (
                    (event.current, event.total)
                    if event.current is not None and event.total is not None
                    else None
                )
                if self._verbose or not self._live:
                    return [Text.from_markup(f"[dim]   {event.node}: {event.message}[/dim]")]
                return []

            case LLMCall():
                self._calls = event.total_calls
                self._tokens = event.total_prompt_tokens + event.total_completion_tokens
                self._usd = event.total_usd
                self._unpriced = event.unpriced_calls
                if self._verbose >= 2:
                    priced = "unpriced" if event.usd is None else f"${event.usd:.4f}"
                    return [
                        Text.from_markup(
                            f"[dim]   {event.role}/{event.model} "
                            f"{event.prompt_tokens}+{event.completion_tokens} tok "
                            f"{priced} {event.seconds:.1f}s[/dim]"
                        )
                    ]
                return []

            case WarningEvent():
                self._warnings += 1
                return [Text.from_markup(f"[yellow]![/yellow]  {event.node}: {event.message}")]

            case ErrorEvent():
                self._errors += 1
                tag = "[red]FATAL[/red]" if event.fatal else "[red]ERR[/red]"
                return [Text.from_markup(f"{tag} {event.node}: {event.message}")]

            case RunFinished():
                return [self._summary(event)]
        # No fallback branch: the match above covers the Event union exhaustively, and
        # mypy will flag a missing return here the moment a new event type is added
        # without a case for it.

    def _meter_text(self) -> str:
        cost = format_cost(self._usd, calls=self._calls, unpriced=self._unpriced)
        parts = [self._node or "starting", f"{self._tokens:,} tok", cost]
        if self._progress is not None:
            current, total = self._progress
            parts.insert(1, f"{current}/{total}")
        return "  ".join(parts)

    def _meter(self) -> Text:
        return Text.from_markup(f"[cyan]...[/cyan] {self._meter_text()}")

    def _summary(self, event: RunFinished) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column()

        status = "[green]complete[/green]" if event.ok else "[red]failed[/red]"
        table.add_row("run", f"{event.run_id}  {status}")
        if event.summary:
            table.add_row("", event.summary)
        table.add_row("tokens", f"{self._tokens:,}")
        table.add_row("cost", format_cost(self._usd, calls=self._calls, unpriced=self._unpriced))
        table.add_row("model calls", str(self._calls))
        if self._unpriced:
            table.add_row(
                "",
                Text.from_markup(
                    f"[yellow]{self._unpriced} call(s) could not be priced.[/yellow] "
                    "Set OKF_LOREMASTER_PRICE_<ROLE>_IN/_OUT for a USD figure."
                ),
            )
        if self._warnings:
            table.add_row("warnings", str(self._warnings))
        if self._errors:
            table.add_row("errors", str(self._errors))
        return table


async def render(bus: EventBus, **kwargs: object) -> asyncio.Task[None]:
    """Start a renderer task subscribed to `bus`. Await the task after closing the bus."""
    renderer = PlainRenderer(bus, **kwargs)  # type: ignore[arg-type]
    return asyncio.create_task(renderer.run())


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"
