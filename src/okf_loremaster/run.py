"""Assembling a run: settings, event bus, renderer, clients, router, pauses.

The CLI declares flags; this decides what they mean. Kept out of `cli.py` so the same
assembly can be driven from a test with an injected transport and injected settings,
which is what makes a `--dry-run` regression test possible without a network.

Two of the choices here are the load-bearing ones:

**`router=None` is what "zero LLM calls" means.** A dry run is not a run with the model
calls skipped by convention — there is no router to call. Anything that tried would
raise rather than quietly spend.

**A declined pause still writes `charter.yaml`.** Declining at the charter pause is the
intended way to correct a taxonomy or a missing vocabulary, and that is only useful if
the file to edit is already on disk when the run stops.

**The renderer, the pause and the reviewer are injectable.** `--tui` supplies all three
and nothing else changes: a Textual app is a second subscriber to the same bus and a
second implementation of the same two protocols, not a second way to run the graph.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from okf_loremaster.curation import MAX_ROUNDS
from okf_loremaster.schemas import (
    DEFAULT_TARGET_PAPERS,
    DEFAULT_TOPIC_MAX,
    DEFAULT_TOPIC_MIN,
    Charter,
)

if TYPE_CHECKING:  # imported lazily below; litellm and httpx are slow to import
    import httpx
    from rich.console import Console

    from okf_loremaster.config import Settings
    from okf_loremaster.emitters.vectors import Embedder
    from okf_loremaster.events import EventBus
    from okf_loremaster.graph.state import RunState
    from okf_loremaster.llm.router import Router
    from okf_loremaster.review import Reviewer
    from okf_loremaster.ui.pauses import Pause

__all__ = [
    "RunInterrupted",
    "RunOptions",
    "build_run",
    "draft_charter",
    "embedder",
    "new_run_id",
    "parse_vocab",
    "require_textual",
    "run_directory",
]

CHARTER_FILENAME = "charter.yaml"


class RunInterrupted(Exception):
    """A run stopped on purpose, part way. Carries the id needed to resume it.

    Distinct from a failure: nothing went wrong, and the checkpoint is intact. Defined
    here rather than in `ui.tui` so the CLI can catch it without importing textual.
    """

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id} stopped before finishing")
        self.run_id = run_id


def new_run_id() -> str:
    """A sortable, unique run id. It is also the checkpoint thread id and the run
    directory name, so it has to be filesystem-safe and orderable."""
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:4]}"


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Everything the CLI decided, in one place."""

    prompt: str = ""
    charter_path: Path | None = None
    out: Path | None = None
    pool_size: int = 800
    screen_budget: int = 400
    target_papers: int = DEFAULT_TARGET_PAPERS
    topic_min: int = DEFAULT_TOPIC_MIN
    topic_max: int = DEFAULT_TOPIC_MAX
    max_queries: int = 12
    max_rounds: int = MAX_ROUNDS
    vocab: list[str] = field(default_factory=list)
    # Stop at the charter and the retrieved pool and ask. Off by default: a run is
    # autonomous end to end unless someone asks to be in it. The two pauses still print
    # what they would have asked about, so an unattended run is not a silent one.
    interactive: bool = False
    # Ask for human sign-off before the bundle is written. Refused with `--dry-run` or
    # `--json` by the CLI: auto-signing would attribute `human:<id>` to someone who
    # never looked.
    review: bool = False
    # Build the vector index once the bundle is written. Off by default because it
    # downloads an embedding model on first use, and a bundle is complete without one.
    index: bool = False
    dry_run: bool = False
    resume: str | None = None
    # Full-screen Textual interface. Refused with `--json` by the CLI, and quietly
    # declined for `--dry-run` or a terminal that cannot drive it.
    tui: bool = False
    json_out: bool = False
    verbose: int = 0


def parse_vocab(raw: str | None) -> list[str]:
    """`--vocab icd10,atc,loinc` to a list. Empty entries dropped, order kept."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_directory(options: RunOptions, settings: Settings, run_id: str) -> Path:
    """Where this run's bundle goes.

    Without `-o` the run id names the folder, which is unique and sortable but not
    memorable. With `-o` the name is the user's, resolved under the same output
    directory — see `Settings.resolve_output`, which is also what `export` uses.
    """
    if options.out is None:
        return settings.output_dir / run_id
    return settings.resolve_output(options.out)


# --- the build command ------------------------------------------------------


async def build_run(
    options: RunOptions,
    *,
    console: Console | None = None,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    attach: Callable[[EventBus], asyncio.Task[None]] | None = None,
    pause: Pause | None = None,
    reviewer: Reviewer | None = None,
) -> tuple[RunState, Path]:
    """Run the whole graph, `charter` through `validate`, plus any re-query round it
    asks for. Returns the state and the run directory.

    Returns rather than raises on a declined pause: a run stopped after ranking has
    still produced a real query plan and a real pool, and the caller wants to report
    them.

    `attach`, `pause` and `reviewer` override the console surfaces. `attach` is handed
    the bus and returns a task that must end when the bus closes — the same contract
    `_start_renderer` satisfies, so the lifecycle below does not change shape.
    """
    from okf_loremaster.clients import build_clients
    from okf_loremaster.config import load_settings
    from okf_loremaster.events import EventBus
    from okf_loremaster.graph.build import run_build
    from okf_loremaster.graph.state import Deps

    resolved = settings if settings is not None else load_settings()
    # Checked before anything is built, so a missing model name fails in a sentence
    # rather than three nodes deep. A dry run needs none of it.
    if not options.dry_run:
        resolved.require_llm()

    charter = _load_charter(options.charter_path)
    prompt = options.prompt or (charter.prompt if charter is not None else "")
    if not prompt:
        raise ValueError("no prompt: pass one as an argument or use --charter")

    run_id = options.resume or new_run_id()
    # Decided here, before the graph starts, and carried on `Deps`. The emit node must
    # not work it out for itself: a resumed run would then land somewhere else than the
    # run it resumed.
    directory = run_directory(options, resolved, run_id)
    bus = EventBus()
    task = attach(bus) if attach is not None else _start_renderer(bus, options, console)

    clients = build_clients(resolved, bus=bus, transport=transport)
    router = None if options.dry_run else _router(resolved, bus)
    deps = Deps(
        settings=resolved,
        bus=bus,
        clients=clients,
        router=router,
        bundle_dir=directory,
        reviewer=reviewer if reviewer is not None else _reviewer(options, resolved, console),
        embedder=embedder(resolved) if options.index else None,
        pool_size=options.pool_size,
        screen_budget=options.screen_budget,
        max_queries=options.max_queries,
        max_rounds=options.max_rounds,
        target_papers=options.target_papers,
        topic_min=options.topic_min,
        topic_max=options.topic_max,
    )

    try:
        state = await run_build(
            run_id=run_id,
            prompt=prompt,
            deps=deps,
            pause=pause if pause is not None else _pause(options, console),
            charter=charter,
            vocab_override=options.vocab,
            dry_run=options.dry_run,
            resume=options.resume is not None,
        )
    finally:
        await clients.aclose()
        bus.close()
        await task

    settled = state.get("charter")
    if settled is not None:
        # Written again here rather than only by the emitter: a run that stopped at a
        # pause never reached `emit_okf`, and the charter is the file you go and edit.
        _write_charter(settled, directory / CHARTER_FILENAME)
    return state, directory


# --- the charter command ----------------------------------------------------


async def draft_charter(
    prompt: str,
    *,
    out: Path,
    vocab: list[str] | None = None,
    target_papers: int = DEFAULT_TARGET_PAPERS,
    topic_min: int = DEFAULT_TOPIC_MIN,
    topic_max: int = DEFAULT_TOPIC_MAX,
    console: Console | None = None,
    settings: Settings | None = None,
    verbose: int = 0,
) -> Charter:
    """One reasoning-tier call, then write the result. No search, no graph.

    The charter node is invoked directly rather than through the graph: there is no
    second node to run, and a checkpoint thread for a single call would be scaffolding
    around nothing.
    """
    from okf_loremaster.clients import build_clients
    from okf_loremaster.config import load_settings
    from okf_loremaster.events import EventBus, RunFinished, RunStarted
    from okf_loremaster.graph.nodes import charter_node
    from okf_loremaster.graph.state import Deps, initial_state

    resolved = settings if settings is not None else load_settings()
    resolved.require_llm()

    bus = EventBus()
    task = _start_renderer(bus, RunOptions(verbose=verbose), console)
    # The charter node itself never touches the network, but `Deps` carries the clients
    # for every other node and building them here keeps one code path.
    clients = build_clients(resolved, bus=bus)
    deps = Deps(
        settings=resolved,
        bus=bus,
        clients=clients,
        router=_router(resolved, bus),
        target_papers=target_papers,
        topic_min=topic_min,
        topic_max=topic_max,
    )

    run_id = new_run_id()
    bus.emit(RunStarted(run_id=run_id, prompt=prompt))
    try:
        state = initial_state(run_id, prompt, vocab_override=vocab)
        update = await charter_node(state, deps)
        charter: Charter = update["charter"]
        bus.emit(
            RunFinished(
                run_id=run_id,
                ok=True,
                summary=f"{len(charter.topic_taxonomy)} topics -> {out}",
            )
        )
    finally:
        await clients.aclose()
        bus.close()
        await task

    _write_charter(charter, out)
    return charter


# --- shared -----------------------------------------------------------------


def _router(settings: Settings, bus: EventBus) -> Router:
    from okf_loremaster.llm.router import Router

    return Router(settings, bus)


def embedder(settings: Settings) -> Embedder:
    """The configured embedder, with the `[vectors]` extra checked first.

    The check is here rather than in the node because the node runs last: without it, a
    run that took an hour would reach the final step and only then discover that the
    thing it was asked to build cannot be built. Nothing is loaded — the import is a
    presence test, and the model itself downloads on first use.
    """
    from okf_loremaster.config import ConfigError
    from okf_loremaster.emitters.vectors import SentenceTransformerEmbedder

    for package in ("chromadb", "sentence_transformers"):
        if importlib.util.find_spec(package) is None:
            raise ConfigError(
                f"vector indexing needs {package}, which is not installed — "
                f"`pip install 'okf-loremaster[vectors]'`, or drop --index"
            )
    return SentenceTransformerEmbedder(settings.embed_model, settings.embed_revision)


def require_textual() -> None:
    """Fail before the screen is cleared, not after.

    Same shape as the `[vectors]` check above and for the same reason: a missing extra
    should read as one sentence naming the fix, not as an ImportError from a UI module.
    """
    from okf_loremaster.config import ConfigError

    if importlib.util.find_spec("textual") is None:
        raise ConfigError(
            "the full-screen interface needs textual, which is not installed — "
            "`pip install 'okf-loremaster[tui]'`, or drop --tui"
        )


def _pause(options: RunOptions, console: Console | None) -> Pause:
    """Which confirmation surface to use.

    A run is autonomous unless `--interactive` asks for the pauses, so the default
    surface prints each decision and continues. `--json` is the one override: a
    machine-readable stream has nobody to ask, and printing tables into it would corrupt
    the thing the flag exists to produce.

    `--dry-run` is deliberately *not* an override. Interactivity used to be the default
    and a dry run opted out of it, because approving a run that costs nothing is noise.
    Now that asking has to be requested, honoring the request is the simpler rule and
    the one both renderers can share — and a dry run is exactly where rehearsing the
    decisions is cheap.
    """
    from okf_loremaster.ui.pauses import AutoApprove, ConsolePause

    if options.json_out:
        return AutoApprove()
    return ConsolePause(console, interactive=options.interactive)


def _reviewer(options: RunOptions, settings: Settings, console: Console | None) -> Reviewer | None:
    """The sign-off surface, or None when nobody asked for one.

    None is not a decline. It is the review node's signal that the question was never
    put, which is why a run without `--review` carries no warning about being unverified
    — an unsigned bundle is the normal case, not a shortfall.
    """
    if not options.review:
        return None
    from okf_loremaster.review import signer_id
    from okf_loremaster.ui.review import ConsoleReviewer

    return ConsoleReviewer(signer_id(settings), console=console)


def _start_renderer(
    bus: EventBus, options: RunOptions, console: Console | None
) -> asyncio.Task[None]:
    """Subscribe a renderer and start consuming. Await the task after closing the bus.

    The renderer subscribes in its constructor, so nothing emitted between here and the
    first `await` is lost; only the returned task needs keeping.
    """
    from okf_loremaster.ui.jsonl import JsonlRenderer
    from okf_loremaster.ui.plain import PlainRenderer

    renderer: JsonlRenderer | PlainRenderer
    if options.json_out:
        renderer = JsonlRenderer(bus)
    else:
        renderer = PlainRenderer(bus, console=console, verbose=options.verbose)
    return asyncio.create_task(renderer.run())


def _load_charter(path: Path | None) -> Charter | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"no charter at {path}")
    return Charter.from_yaml(path.read_text(encoding="utf-8"))


def _write_charter(charter: Charter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(charter.to_yaml(), encoding="utf-8")
