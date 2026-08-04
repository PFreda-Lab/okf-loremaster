"""The build step 9 gate: `--tui` works and `q` checkpoints gracefully.

Driven with Textual's `run_test()` against the same fake NCBI corpus the dry-run tests
use, so what is exercised is a real graph run answered through the real modal, not a
mock of one. Two claims are worth stating because they are the ones a refactor breaks:

**The TUI is a renderer.** It is handed to `build_run` as three arguments and changes
nothing else, so a run through the app reaches the same state a run through the console
does. `test_a_run_through_the_app_reaches_the_same_state` is that claim.

**`q` stops rather than kills.** The run task is cancelled, `run_build`'s checkpointer
closes on the way out, and the run id the app reports is the one `--resume` wants.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from okf_loremaster.cli import app as cli
from okf_loremaster.graph.build import NODES, build_graph
from okf_loremaster.graph.state import Deps
from okf_loremaster.run import RunInterrupted, RunOptions

pytest.importorskip("textual")

from okf_loremaster.ui.tui import (
    DONE,
    PENDING,
    ConfirmScreen,
    LoremasterApp,
    build_run_tui,
)
from test_dry_run import POOL_SIZE, PROMPT, charter_for

from fake_ncbi import FakeNCBI

runner = CliRunner()

# How long a pilot waits for the app to reach a state. Generous because the graph does
# real parsing behind the modal; a stuck test fails in seconds either way.
TIMEOUT = 20.0


def tui_run(
    settings_factory: Any, tmp_path: Path, **overrides: Any
) -> tuple[LoremasterApp, Any]:
    """An app wired to the fake corpus, and the settings it will use.

    `interactive=True` unless a test says otherwise: a run is autonomous by default and
    would never raise a modal, and most of what there is to test here is the modal.
    """
    charter_path = tmp_path / "given.yaml"
    charter_path.write_text(charter_for().to_yaml(), encoding="utf-8")
    options = RunOptions(
        **{
            "prompt": PROMPT,
            "charter_path": charter_path,
            "out": tmp_path / "run",
            "pool_size": POOL_SIZE,
            "target_papers": 120,
            "dry_run": True,
            "tui": True,
            "interactive": True,
            **overrides,
        }
    )
    settings = settings_factory(
        ncbi_email="test@example.org",
        http_cache_enabled=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    app = LoremasterApp(options, settings=settings, transport=FakeNCBI().transport())
    return app, settings


async def settle(pilot: Any, until: Any) -> None:
    """Wait for `until()` to hold, pumping the app in between."""
    deadline = asyncio.get_running_loop().time() + TIMEOUT
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if until():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the app never reached the expected state")


def asking(app: LoremasterApp) -> bool:
    return isinstance(app.screen, ConfirmScreen)


def log_text(app: LoremasterApp) -> str:
    """The log pane as it reads on screen.

    Segments are joined within a line, not across lines. A style change starts a new
    segment, so `[dim]->[/dim] rank` is two of them — splitting on segments would put a
    newline mid-line and quietly defeat any assertion about a phrase that spans one.
    """
    from textual.widgets import RichLog

    lines = app.query_one("#log", RichLog).lines
    return "\n".join("".join(segment.text for segment in line) for line in lines)


# --- the panel matches the graph --------------------------------------------


def test_the_node_panel_cannot_drift_from_the_pipeline(settings_factory: Any) -> None:
    """`NODES` is what the TUI draws; the graph is what actually runs.

    Two lists that must agree, so the one place they can disagree is asserted rather
    than trusted. A node added to the graph and not to `NODES` would simply never appear
    on screen, which is the kind of omission nobody notices.
    """
    from okf_loremaster.events import EventBus

    deps = Deps(settings=settings_factory(), bus=EventBus(), clients=None)  # type: ignore[arg-type]
    assert tuple(build_graph(deps).nodes) == NODES


# --- a run through the app ---------------------------------------------------


async def test_a_run_through_the_app_reaches_the_same_state(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approve both pauses from the modal and the dry run completes as it always does."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run constructed a Router — that is a model call waiting")

    monkeypatch.setattr("okf_loremaster.llm.router.Router", refuse)
    app, _ = tui_run(settings_factory, tmp_path)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: asking(app))  # charter
        await pilot.press("y")
        await settle(pilot, lambda: asking(app))  # retrieve
        await pilot.press("y")
        await settle(pilot, lambda: app.outcome is not None)
        # Read inside the context: the widgets are gone once the app has shut down.
        log = log_text(app)
        await pilot.press("q")

    assert app.error is None
    assert app.outcome is not None
    state, directory = app.outcome
    assert len(state["pool"]) == POOL_SIZE
    assert (directory / "charter.yaml").exists()

    # The panel and the log are driven off the same events the console renderer reads.
    assert app._status["rank"] == DONE
    assert app._status["screen"] == PENDING  # a dry run stops after ranking
    assert app.run_id in log
    assert "rank" in log


async def test_an_autonomous_run_logs_the_pauses_instead_of_asking(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `--interactive` no modal ever opens, and nothing is hidden either.

    The full screen is worth watching on an unattended run, so what the modal would have
    shown goes to the log instead — the same view, minus the question.
    """
    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path, interactive=False)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: app.outcome is not None)
        assert not asking(app)
        log = log_text(app)
        await pilot.press("q")

    assert app.error is None
    assert log.count("continuing without asking") == 2  # charter, then retrieve


# --- what a watcher sees while waiting ---------------------------------------


async def test_the_log_says_something_before_the_first_event_arrives(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the checkpoint and building the clients all happen before `RunStarted`.

    On a real run the first node is the charter — one reasoning-tier call for a long
    reply — so an app that writes nothing until a node finishes shows an empty pane for
    the longest stretch of the run, at the point a watcher has the least evidence that
    anything is working.
    """
    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert "starting" in log_text(app)
        await settle(pilot, lambda: asking(app))
        await pilot.press("q")


async def test_a_node_is_logged_when_it_starts_and_not_only_when_it_finishes(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pane says what is running now; the log is what gets read afterward."""
    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path, interactive=False)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: app.outcome is not None)
        log = log_text(app)
        await pilot.press("q")

    for node in ("charter", "search", "dedupe", "rank"):
        assert f"-> {node}" in log


async def test_a_progress_line_with_no_counter_is_shown_without_asking_for_verbose(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A counted tick belongs in the pipeline pane, where it overwrites itself.

    An uncounted one is a node saying what it is about to do, there are fewer than ten
    of them in a whole run, and they are exactly what someone waiting wants to read.
    `verbose` is 0 here, which is how the user runs it.
    """
    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path, interactive=False)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: app.outcome is not None)
        log = log_text(app)
        await pilot.press("q")

    assert app._options.verbose == 0
    assert "rank: citation metrics for" in log


def panel_text(app: LoremasterApp, which: str) -> str:
    """A widget as it is painted — the pipeline pane, or the meter.

    Read off the rendered lines rather than the renderable: Textual 8 wraps what a
    `Static` was given in a `Visual`, so there is no attribute holding the Rich object
    back, and the painted text is the honest thing to assert on anyway.
    """
    from textual.geometry import Region
    from textual.widgets import Static

    widget = app.query_one(f"#{which}", Static)
    lines = widget.render_lines(Region(0, 0, widget.size.width, widget.size.height))
    return "\n".join("".join(segment.text for segment in line) for line in lines)


async def test_a_node_with_nothing_to_count_still_shows_that_it_is_moving(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one place a run looks hung.

    Everything else on screen changes when a node reports something, and a model call
    reports nothing until it answers. The charter is one reasoning-tier request with a
    300-second timeout behind it, so a working run and a hung one were indistinguishable
    for up to five minutes. `_tick` is driven directly rather than by waiting on the real
    clock — what is under test is that a tick moves the display, not that Textual can
    schedule one.
    """
    from okf_loremaster.events import NodeStarted

    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: asking(app))
        # The pause sits *between* nodes, so the charter has already finished and reports
        # its duration. Started by hand to hold a node open, which a real slow call does
        # and a fake corpus never will.
        app._handle(NodeStarted(node="screen"))
        for _ in range(3):
            app._tick()
        await pilot.pause()
        panel, meter = panel_text(app, "nodes"), panel_text(app, "meter")
        await pilot.press("q")

    assert "3s" in panel, "the running node does not say how long it has been running"
    assert "0m03s" in meter


async def test_the_clock_stops_when_the_run_does(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished run that keeps counting reads as one that is still going."""
    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path, interactive=False)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: app.outcome is not None)
        settled = app._elapsed
        for _ in range(3):
            app._tick()
        await pilot.press("q")

    assert app._elapsed == settled


# --- getting the run out of the terminal --------------------------------------


async def test_the_log_is_kept_as_text_so_it_can_be_pasted_rather_than_photographed(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full-screen app's output is not in the scrollback, and is gone when it closes.

    Selecting on screen works and is now advertised, but it goes through a clipboard
    escape the terminal is free to ignore, and it cannot reach lines the pane has already
    scrolled past. The file can.
    """
    from okf_loremaster.run import TRANSCRIPT_FILENAME

    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path, interactive=False)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: app.outcome is not None)
        await pilot.press("q")

    assert app.outcome is not None
    written = (app.outcome[1] / TRANSCRIPT_FILENAME).read_text(encoding="utf-8")
    assert app.run_id in written
    assert "-> rank" in written
    assert "\x1b[" not in written, "written to be pasted, so no color codes"


async def test_the_transcript_ends_with_what_the_run_cost(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tokens and dollars live on the meter, which is a widget and not a log line.

    So the one number somebody opens a transcript to find was the one thing the transcript
    did not have. A real run's `run.log` held a pre-run estimate table and no total, which
    is worse than nothing: the estimate reads like an actual.
    """
    from okf_loremaster.run import TRANSCRIPT_FILENAME

    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path, interactive=False)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: app.outcome is not None)
        await pilot.press("q")

    assert app.outcome is not None
    last = (app.outcome[1] / TRANSCRIPT_FILENAME).read_text(encoding="utf-8").strip()
    assert "tok" in last.rsplit("\n", 1)[-1]


def test_copying_a_selection_is_in_the_footer_where_it_can_be_found() -> None:
    """Textual has selected on drag and copied on ctrl+c since 7.0, bound with
    `show=False`. Undiscoverable is the same as missing when the reason the log pane
    exists is that a warning can be quoted somewhere else."""
    copies = [b for b in LoremasterApp.BINDINGS if getattr(b, "action", "") == "copy"]

    assert copies, "nothing in the footer says a selection can be copied"
    assert all(b.show for b in copies)


async def test_pressing_copy_with_nothing_selected_says_so_instead_of_nothing(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Textual's `screen.copy_text` raises `SkipAction` on an empty selection, so the key
    advertised in the footer did nothing at all and the app looked broken. It is a missing
    drag, and the message that says so also names the file that needs no drag."""
    from okf_loremaster.run import TRANSCRIPT_FILENAME

    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path, interactive=False)
    said: list[str] = []

    async with app.run_test() as pilot:
        await settle(pilot, lambda: app.outcome is not None)
        monkeypatch.setattr(app, "notify", lambda message, **kw: said.append(message))
        await pilot.press("c")
        await pilot.press("q")

    assert said, "pressing c with no selection said nothing"
    assert "nothing selected" in said[0]
    assert TRANSCRIPT_FILENAME in said[0]


async def test_declining_the_charter_stops_the_run_without_failing_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`n` at the first pause is the intended way to go and edit charter.yaml.

    Escape rather than the `n` key, so the third way out of the modal is exercised too.
    """
    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, _ = tui_run(settings_factory, tmp_path)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: asking(app))
        await pilot.press("escape")
        await settle(pilot, lambda: app.outcome is not None)
        await pilot.press("q")

    assert app.error is None
    assert app.outcome is not None
    state, directory = app.outcome
    assert not state.get("pool")
    # The charter is on disk, which is the whole reason declining here is useful.
    assert (directory / "charter.yaml").exists()


async def test_q_stops_the_run_and_leaves_a_checkpoint_to_resume_from(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate. `q` cancels the run task; it does not kill the process.

    The checkpoint database is what makes the id it reports worth anything — without one
    `--resume <id>` would name a run nothing could be resumed from.
    """
    monkeypatch.setattr(
        "okf_loremaster.llm.router.Router", lambda *a, **k: pytest.fail("no model")
    )
    app, settings = tui_run(settings_factory, tmp_path)

    async with app.run_test() as pilot:
        await settle(pilot, lambda: asking(app))
        await pilot.press("q")
        await settle(pilot, lambda: app.interrupted)

    assert app.interrupted is True
    assert app.outcome is None
    assert app.error is None
    assert app.run_id  # taken off RunStarted, and it is what --resume wants
    assert (settings.cache_dir / "checkpoints.sqlite").exists()


# --- what the caller gets back -----------------------------------------------


async def test_a_stopped_run_is_reported_as_resumable_not_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stopped(self: LoremasterApp, *args: Any, **kwargs: Any) -> None:
        self.interrupted = True
        self.run_id = "20260803-101010-abcd"

    monkeypatch.setattr(LoremasterApp, "run_async", stopped)

    with pytest.raises(RunInterrupted) as caught:
        await build_run_tui(RunOptions(prompt="anything"))

    assert caught.value.run_id == "20260803-101010-abcd"


async def test_a_failed_run_raises_what_it_failed_with(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app catches so it can show; the caller still gets the real exception."""

    async def failed(self: LoremasterApp, *args: Any, **kwargs: Any) -> None:
        self.error = ValueError("no prompt: pass one as an argument or use --charter")

    monkeypatch.setattr(LoremasterApp, "run_async", failed)

    with pytest.raises(ValueError, match="no prompt"):
        await build_run_tui(RunOptions())


# --- the flag ----------------------------------------------------------------


def test_tui_and_json_are_refused_rather_than_ranked() -> None:
    result = runner.invoke(cli, ["build", "a prompt", "--tui", "--json"])

    assert result.exit_code == 1, result.output
    assert "--tui cannot be combined with --json" in result.output


def _intercept(monkeypatch: pytest.MonkeyPatch, target: str) -> list[RunOptions]:
    """Replace a runner with one that records its options and stops the command."""
    seen: list[RunOptions] = []

    async def stop(options: RunOptions, **kwargs: Any) -> Any:
        seen.append(options)
        raise ValueError("reached the runner")

    monkeypatch.setattr(target, stop)
    return seen


def test_a_dry_run_keeps_its_printed_plan_instead_of_taking_the_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _intercept(monkeypatch, "okf_loremaster.run.build_run")

    result = runner.invoke(cli, ["build", "a prompt", "--tui", "--dry-run"])

    assert "--dry-run prints its plan" in result.output
    assert result.exit_code == 1  # the intercept, not the flag
    assert seen and seen[0].tui is False


def test_no_terminal_falls_back_to_the_console_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("okf_loremaster.ui.plain.rich_enabled", lambda *a, **k: False)
    seen = _intercept(monkeypatch, "okf_loremaster.run.build_run")

    result = runner.invoke(cli, ["build", "a prompt", "--tui"])

    assert "no terminal to drive" in result.output
    assert seen and seen[0].tui is False


def test_a_terminal_gets_the_full_screen_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("okf_loremaster.ui.plain.rich_enabled", lambda *a, **k: True)
    seen = _intercept(monkeypatch, "okf_loremaster.ui.tui.build_run_tui")

    result = runner.invoke(cli, ["build", "a prompt", "--tui"])

    assert "falling back" not in result.output
    assert seen and seen[0].tui is True


def test_a_missing_extra_is_named_before_the_screen_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the name survives being printed.

    The fix is `pip install 'okf-loremaster[tui]'`, and Rich reads `[tui]` as a markup
    tag — so the one word the user has to type was being swallowed. The same was true of
    the `[vectors]` hint, which is why the escaping lives in the shared error handler.
    """
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr("okf_loremaster.ui.plain.rich_enabled", lambda *a, **k: True)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "textual" else real(name, *a, **k),
    )

    result = runner.invoke(cli, ["build", "a prompt", "--tui"])

    assert result.exit_code == 1, result.output
    assert "okf-loremaster[tui]" in _uncolored(result.output)


def _uncolored(output: str) -> str:
    """CliRunner keeps the color codes; the assertions are about the words."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)
