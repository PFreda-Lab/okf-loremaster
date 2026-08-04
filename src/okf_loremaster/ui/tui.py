"""The `--tui` full-screen interface. A renderer, and nothing more.

Everything on screen comes off the same `EventBus` the console renderer subscribes to,
and the two decision surfaces are the same `charter_view` / `retrieve_view` /
`signoff_view` renderables the console prints, shown in a modal instead. No node knows
which is attached, and there is no second way to run the graph — `build_run` is called
exactly as the plain path calls it, with three arguments injected.

Two things are load-bearing:

**The run is a Textual worker, in the app's own event loop.** `push_screen_wait` can
only be awaited inside a worker, and the bus is an `asyncio.Queue` fanned out with
`put_nowait` — emitting from a second thread would be a data race. One loop, one thread,
no bridge.

**`q` cancels the worker; it does not kill the process.** Cancellation unwinds through
`run_build`'s `async with checkpointer(...)`, so the SQLite checkpoint closes cleanly and
the run id it was writing under is the one `--resume` wants. The id is taken off
`RunStarted` rather than passed in, because `build_run` is what mints it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, RichLog, Static

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
from okf_loremaster.graph.build import NODES
from okf_loremaster.llm.router import format_cost
from okf_loremaster.review import Signoff
from okf_loremaster.run import RunInterrupted, RunOptions, build_run
from okf_loremaster.schemas import Charter, ConceptRecord, VerificationSummary
from okf_loremaster.ui.pauses import PauseDecision, charter_view, retrieve_view
from okf_loremaster.ui.review import signoff_caption, signoff_view

if TYPE_CHECKING:
    import httpx

    from okf_loremaster.config import Settings
    from okf_loremaster.graph.state import RunState
    from okf_loremaster.llm.estimate import SpendEstimate

__all__ = ["ConfirmScreen", "LoremasterApp", "TuiPause", "TuiReviewer", "build_run_tui"]

# Node marks, deliberately the same vocabulary the console renderer uses.
PENDING = "[dim]..[/dim]"
RUNNING = "[cyan]->[/cyan]"
DONE = "[green]OK[/green]"


class ConfirmScreen(ModalScreen[bool]):
    """A scrollable body of renderables and a yes/no question about them.

    One screen for all three decisions. What differs between a charter pause, a retrieve
    pause and a sign-off is the content and the wording, never the mechanics, and giving
    each its own screen class would be three places to fix the same bug.

    `q` is re-bound here because a modal screen takes bindings over the app's, and a
    question you cannot walk away from is a trap. It means what it means everywhere else
    — stop the run and checkpoint it — not "no".
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "answer(True)", "yes", show=False),
        Binding("n", "answer(False)", "no", show=False),
        Binding("escape", "answer(False)", "no", show=False),
        Binding("q", "stop", "quit", show=False),
    ]

    def __init__(
        self,
        view: Sequence[RenderableType],
        *,
        question: str,
        yes: str = "Proceed",
        no: str = "Stop",
        default: bool = True,
    ) -> None:
        super().__init__()
        self._view = list(view)
        self._question = question
        self._yes = yes
        self._no = no
        self._default = default

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            with VerticalScroll(id="dialog-body"):
                for index, item in enumerate(self._view):
                    yield Static(item, id=f"dialog-item-{index}")
            yield Static(Text.from_markup(f"[bold]{self._question}[/bold]"), id="dialog-question")
            with Horizontal(id="dialog-buttons"):
                yield Button(f"{self._yes}  (y)", variant="success", id="yes")
                yield Button(f"{self._no}  (n)", variant="error", id="no")
            yield Static(
                Text.from_markup("[dim]q stops the run and checkpoints it[/dim]"),
                id="dialog-keys",
            )

    def on_mount(self) -> None:
        self.query_one("#yes" if self._default else "#no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, answer: bool) -> None:
        self.dismiss(answer)

    def action_stop(self) -> None:
        stop = getattr(self.app, "action_stop", None)
        if callable(stop):
            stop()


class TuiPause:
    """`ui.pauses.Pause`, answered by a modal instead of a prompt.

    `interactive=False` is an autonomous run watched full screen: the same view goes to
    the log pane instead of a modal, so what a modal would have shown is still on record
    and nothing waits for a keypress.
    """

    def __init__(self, app: LoremasterApp, *, interactive: bool = True) -> None:
        self._app = app
        self._interactive = interactive

    async def charter(self, charter: Charter) -> PauseDecision:
        return await self._decide(
            charter_view(charter),
            question="Proceed with this charter?",
            no="Stop and edit charter.yaml",
        )

    async def retrieve(self, state: RunState, *, estimate: SpendEstimate | None) -> PauseDecision:
        pool = list(state.get("pool") or [])
        return await self._decide(
            retrieve_view(state, estimate=estimate),
            question=f"Screen these {len(pool)} papers?",
        )

    async def _decide(
        self, view: list[RenderableType], *, question: str, no: str = "Stop"
    ) -> PauseDecision:
        if not self._interactive:
            for item in view:
                self._app.log_view(item)
            self._app.log_view(f"{question} — continuing without asking")
            return PauseDecision(proceed=True)
        proceed = await self._app.confirm(view, question=question, no=no)
        return PauseDecision(proceed=proceed, reason="" if proceed else "declined at pause")


class TuiReviewer:
    """`review.Reviewer`, answered by a modal instead of a prompt.

    Defaults to declining, exactly as the console reviewer does: a signature has to be
    reached for, not arrived at by pressing enter.
    """

    def __init__(self, app: LoremasterApp, signer: str) -> None:
        self._app = app
        self._signer = signer

    async def sign_off(
        self,
        records: Sequence[ConceptRecord],
        *,
        topics: dict[str, list[str]],
        verification: VerificationSummary | None,
        warnings: Sequence[str],
    ) -> Signoff:
        if not records:
            return Signoff.declined("no records")
        view = signoff_view(
            records, topics=topics, verification=verification, warnings=warnings
        )
        view.append(signoff_caption(self._signer, len(records)))
        approved = await self._app.confirm(
            view,
            question=f"Sign off on these {len(records)} documents as {self._signer}?",
            yes="Sign",
            no="Decline",
            default=False,
        )
        if not approved:
            return Signoff.declined("declined at review")
        return Signoff.granted(self._signer)


class LoremasterApp(App[None]):
    """The pipeline down the left, the event log on the right, a meter underneath."""

    CSS = """
    #body { height: 1fr; }
    #nodes { width: 30; padding: 0 1; }
    #log { width: 1fr; padding: 0 1; border-left: solid $panel; }
    #meter { height: auto; padding: 0 1; background: $panel; }
    ConfirmScreen { align: center middle; }
    #dialog { width: 90%; height: 90%; border: thick $accent; background: $surface;
              padding: 1 2; }
    #dialog-body { height: 1fr; }
    #dialog-question { height: auto; padding-top: 1; }
    #dialog-buttons { height: auto; align: center middle; padding-top: 1; }
    #dialog-buttons Button { margin: 0 2; }
    #dialog-keys { height: auto; text-align: center; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [Binding("q", "stop", "quit")]

    def __init__(
        self,
        options: RunOptions,
        *,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__()
        self._options = options
        self._settings = settings
        self._transport = transport

        # What `build_run_tui` reads back once the app has closed.
        self.outcome: tuple[RunState, Path] | None = None
        self.error: BaseException | None = None
        self.interrupted = False
        self.run_id = options.resume or ""

        self._queue: asyncio.Queue[Event | None] | None = None
        self._worker: Any = None
        self._stopping = False
        self._done = False

        self._status: dict[str, str] = dict.fromkeys(NODES, PENDING)
        self._detail: dict[str, str] = dict.fromkeys(NODES, "")
        self._node = ""
        self._calls = 0
        self._tokens = 0
        self._usd = 0.0
        self._unpriced = 0
        self._warnings = 0
        self._errors = 0
        self._progress: tuple[int, int] | None = None

    # --- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield Static(id="nodes")
            yield RichLog(id="log", wrap=True, markup=False, highlight=False, max_lines=2000)
        yield Static(id="meter")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "OKF Loremaster"
        self.sub_title = self._options.prompt or str(self._options.charter_path or "")
        self._refresh_nodes()
        self._refresh_meter()
        # `exit_on_error=False` because a failed run is a result to be reported, not a
        # crash: the exception is stored and re-raised by the caller once the app closes.
        self._worker = self.run_worker(self._drive(), name="build", exit_on_error=False)

    # --- the run -----------------------------------------------------------

    async def _drive(self) -> None:
        try:
            self.outcome = await build_run(
                self._options,
                settings=self._settings,
                transport=self._transport,
                attach=self._subscribe,
                pause=TuiPause(self, interactive=self._options.interactive),
                reviewer=self._reviewer(),
            )
        except asyncio.CancelledError:
            # The checkpoint is already flushed by `run_build`'s context manager; there
            # is nothing left to wait for and nothing to show.
            self.interrupted = True
            self.exit()
            raise
        except Exception as exc:
            self.error = exc
            self._write(Text.from_markup(f"[red]FATAL[/red] {type(exc).__name__}: {exc}"))
        self._done = True
        self._refresh_meter()

    def _reviewer(self) -> TuiReviewer | None:
        if not self._options.review:
            return None
        from okf_loremaster.config import load_settings
        from okf_loremaster.review import signer_id

        settings = self._settings if self._settings is not None else load_settings()
        return TuiReviewer(self, signer_id(settings))

    def _subscribe(self, bus: EventBus) -> asyncio.Task[None]:
        """Subscribe before the first node runs, then consume until the bus closes."""
        self._queue = bus.subscribe()
        return asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            if event is None:
                break
            self._handle(event)
            self._refresh_meter()

    # --- decisions ---------------------------------------------------------

    async def confirm(
        self,
        view: Sequence[RenderableType],
        *,
        question: str,
        yes: str = "Proceed",
        no: str = "Stop",
        default: bool = True,
    ) -> bool:
        """Put a decision on screen and wait for it. Only callable from the run worker."""
        answer = await self.push_screen_wait(
            ConfirmScreen(view, question=question, yes=yes, no=no, default=default)
        )
        return bool(answer)

    def log_view(self, item: RenderableType) -> None:
        """Send to the log pane what a modal would have shown, on an autonomous run."""
        self._write(item)

    # --- quitting ----------------------------------------------------------

    def action_stop(self) -> None:
        """`q`. Stops the run if one is going, closes the app if one is not."""
        if self._done or self._worker is None or not self._worker.is_running:
            self.exit()
            return
        if self._stopping:
            return
        self._stopping = True
        self._write(
            Text.from_markup("[yellow]stopping[/yellow] — flushing the checkpoint")
        )
        self._refresh_meter()
        self._worker.cancel()

    # --- rendering ---------------------------------------------------------

    def _handle(self, event: Event) -> None:
        match event:
            case RunStarted():
                self.run_id = event.run_id
                mode = " [dry run]" if event.dry_run else ""
                self.sub_title = f"{event.run_id}{mode}"
                self._write(Text.from_markup(f"[bold]run {event.run_id}[/bold]{mode}"))

            case NodeStarted():
                self._node = event.node
                self._progress = None
                self._status[event.node] = RUNNING
                self._refresh_nodes()

            case NodeFinished():
                self._progress = None
                self._status[event.node] = DONE
                self._detail[event.node] = f"{event.seconds:.1f}s"
                self._refresh_nodes()
                detail = f"  {event.summary}" if event.summary else ""
                self._write(
                    Text.from_markup(
                        f"[green]OK[/green] {event.node}{detail}  "
                        f"[dim]{event.seconds:.1f}s[/dim]"
                    )
                )

            case Progress():
                self._progress = (
                    (event.current, event.total)
                    if event.current is not None and event.total is not None
                    else None
                )
                if self._progress is not None:
                    current, total = self._progress
                    self._detail[event.node] = f"{current}/{total}"
                    self._refresh_nodes()
                if self._options.verbose:
                    self._write(
                        Text.from_markup(f"[dim]   {event.node}: {event.message}[/dim]")
                    )

            case LLMCall():
                self._calls = event.total_calls
                self._tokens = event.total_prompt_tokens + event.total_completion_tokens
                self._usd = event.total_usd
                self._unpriced = event.unpriced_calls
                if self._options.verbose >= 2:
                    priced = "unpriced" if event.usd is None else f"${event.usd:.4f}"
                    self._write(
                        Text.from_markup(
                            f"[dim]   {event.role}/{event.model} "
                            f"{event.prompt_tokens}+{event.completion_tokens} tok "
                            f"{priced} {event.seconds:.1f}s[/dim]"
                        )
                    )

            case WarningEvent():
                self._warnings += 1
                self._write(
                    Text.from_markup(f"[yellow]![/yellow]  {event.node}: {event.message}")
                )

            case ErrorEvent():
                self._errors += 1
                tag = "[red]FATAL[/red]" if event.fatal else "[red]ERR[/red]"
                self._write(Text.from_markup(f"{tag} {event.node}: {event.message}"))

            case RunFinished():
                status = "[green]complete[/green]" if event.ok else "[red]failed[/red]"
                self._write(Text.from_markup(f"{status}  {event.summary}"))
        # No fallback branch: the match covers the Event union, so mypy flags a new
        # event type that nothing here shows.

    def _write(self, line: RenderableType) -> None:
        self.query_one("#log", RichLog).write(line)

    def _refresh_nodes(self) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(width=5)
        table.add_column(overflow="ellipsis")
        table.add_column(justify="right", style="dim")
        for node in NODES:
            style = "bold" if node == self._node else ""
            table.add_row(
                Text.from_markup(self._status[node]),
                Text(node, style=style),
                Text(self._detail[node]),
            )
        self.query_one("#nodes", Static).update(table)

    def _refresh_meter(self) -> None:
        cost = format_cost(self._usd, calls=self._calls, unpriced=self._unpriced)
        parts = [self._node or "starting", f"{self._tokens:,} tok", cost]
        if self._progress is not None:
            current, total = self._progress
            parts.insert(1, f"{current}/{total}")
        if self._warnings:
            parts.append(f"[yellow]{self._warnings} warning(s)[/yellow]")
        if self._errors:
            parts.append(f"[red]{self._errors} error(s)[/red]")
        if self._done:
            parts = ["[green]finished[/green] — press q to close", *parts[1:]]
        elif self._stopping:
            parts = ["[yellow]stopping[/yellow]", *parts[1:]]
        self.query_one("#meter", Static).update(Text.from_markup("  ".join(parts)))


async def build_run_tui(
    options: RunOptions,
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[RunState, Path]:
    """Run a build inside the Textual app and hand back what `build_run` returned.

    Raises `RunInterrupted` when the run was stopped with `q`, so the caller can print a
    resume hint rather than a traceback or a false success.
    """
    app = LoremasterApp(options, settings=settings, transport=transport)
    await app.run_async()
    if app.error is not None:
        raise app.error
    if app.outcome is None:
        raise RunInterrupted(app.run_id)
    return app.outcome
