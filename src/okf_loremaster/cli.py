"""Command-line interface for OKF Loremaster.

Heavy modules are imported inside the commands that need them. `litellm` and `chromadb`
each cost seconds to import, and `--help` should not pay for either. The commands that
touch a bundle rather than build one — `validate`, `export`, `inspect` — import nothing
from the graph at all, so they run against a directory on a machine with no API key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from okf_loremaster import DISPLAY_NAME, __version__

console = Console(stderr=True)

app = typer.Typer(
    name="okf-loremaster",
    help=f"{DISPLAY_NAME} — build a task-scoped biomedical literature bundle "
    "from PubMed/PMC in Open Knowledge Format.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@contextmanager
def _reported() -> Iterator[None]:
    """Turn the failures a user can act on into one line and an exit code.

    Config, file and interrupt errors are the user's to fix and a traceback tells them
    nothing. Everything else propagates: a bug in a node should look like one.
    """
    from okf_loremaster.config import ConfigError
    from okf_loremaster.run import RunInterrupted

    try:
        yield
    except RunInterrupted as exc:
        # Not a failure: the user asked it to stop and the checkpoint is intact. 130
        # anyway, because to whatever ran us this is the same event as a Ctrl-C.
        console.print(
            f"[yellow]stopped[/yellow] — resume with "
            f"[cyan]okf-loremaster build --resume {exc.run_id}[/cyan]"
        )
        raise typer.Exit(code=130) from exc
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        # Escaped, because the fix these messages name is often an extra —
        # `okf-loremaster[tui]` — and Rich reads the brackets as a markup tag and drops
        # the one word the user needs to type.
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt as exc:
        console.print("[yellow]interrupted[/yellow] — rerun with --resume <run-id> to continue")
        raise typer.Exit(code=130) from exc


def _version_callback(value: bool) -> None:
    if value:
        # stdout, not stderr: `--version` output is meant to be piped.
        typer.echo(f"{DISPLAY_NAME} {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Build and inspect Open Knowledge Format literature bundles."""


@app.command()
def init(
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing .env.")] = False,
) -> None:
    """Write a .env from the template and check the environment is usable."""
    from rich.table import Table

    from okf_loremaster.config import (
        ENV_PREFIX,
        ConfigError,
        Role,
        env_file_candidates,
        load_settings,
    )

    template = Path(".env.example")
    target = Path(".env")
    if template.exists() and (force or not target.exists()):
        target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        console.print(f"[green]wrote[/green] {target} from {template} — fill it in, then rerun")
    elif not template.exists() and not target.exists():
        console.print(f"[yellow]neither {template} nor {target} found[/yellow]")

    try:
        settings = load_settings()
    except ConfigError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    found = [str(path) for path in env_file_candidates() if path.exists()]
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_row("env files", ", ".join(found) if found else "[yellow]none found[/yellow]")

    missing = set(settings.missing_for_llm())
    for role in Role:
        name = f"{ENV_PREFIX}MODEL_{role.value.upper()}"
        value = {
            Role.FAST: settings.model_fast,
            Role.BALANCED: settings.model_balanced,
            Role.REASONING: settings.model_reasoning,
        }[role]
        table.add_row(role.value, value if value else f"[red]unset[/red] ({name})")

    table.add_row(
        "api key",
        "[red]unset[/red] (ANTHROPIC_API_KEY)" if "ANTHROPIC_API_KEY" in missing else "set",
    )
    if settings.api_base:
        table.add_row("api base", settings.api_base)
    table.add_row(
        "NCBI email",
        settings.ncbi_email or f"[yellow]unset[/yellow] ({ENV_PREFIX}NCBI_EMAIL)",
    )
    table.add_row(
        "NCBI key",
        "set (10 req/s)" if settings.ncbi_api_key else "[dim]unset (3 req/s)[/dim]",
    )
    revision = settings.embed_revision or "[yellow]unpinned[/yellow]"
    table.add_row("embeddings", f"{settings.embed_model} @ {revision}")
    table.add_row("HF_HOME", str(settings.hf_home) if settings.hf_home else "[dim]default[/dim]")
    table.add_row("cache dir", str(settings.cache_dir))
    table.add_row("output dir", str(settings.output_dir))
    console.print(table)

    warning = settings.hf_home_warning()
    if warning:
        console.print(f"[yellow]![/yellow]  {warning}")

    unpriced = settings.unpriced_roles()
    if unpriced:
        names = ", ".join(role.value for role in unpriced)
        console.print(
            f"[dim]note[/dim]  no price override for: {names}. If the provider is not in "
            f"LiteLLM's price map, those calls report as 'cost unavailable' rather than a "
            f"USD figure. Set {ENV_PREFIX}PRICE_<ROLE>_IN/_OUT to get one."
        )

    if missing:
        console.print(f"[red]not ready[/red] — {len(missing)} required variable(s) unset")
        raise typer.Exit(code=1)
    console.print("[green]ready[/green]")


@app.command(hidden=True)
def selftest(
    live: Annotated[
        bool | None, typer.Option("--live/--no-live", help="Force or suppress the live meter.")
    ] = None,
    verbose: Annotated[int, typer.Option("-v", "--verbose", count=True)] = 0,
) -> None:
    """Exercise events, routing, retries, and cost accounting against a fake model."""
    from okf_loremaster.selftest import run_selftest

    code = asyncio.run(run_selftest(live=live, verbose=verbose, console=console))
    raise typer.Exit(code=code)


@app.command()
def charter(
    prompt: Annotated[str, typer.Argument(help="Plain-language description of the task.")],
    out: Annotated[
        Path, typer.Option("-o", "--out", help="Where to write the charter.")
    ] = Path("charter.yaml"),
    vocab: Annotated[
        str | None, typer.Option("--vocab", help="Comma-separated vocabularies; overrides charter.")
    ] = None,
    target_papers: Annotated[int, typer.Option(help="Target retained paper count.")] = 180,
    topic_min: Annotated[int, typer.Option(help="Minimum papers per topic.")] = 8,
    topic_max: Annotated[int, typer.Option(help="Maximum papers per topic.")] = 40,
    verbose: Annotated[int, typer.Option("-v", "--verbose", count=True, help="Verbosity.")] = 0,
) -> None:
    """Draft a charter — topic taxonomy, vocabularies, query plan — without building."""
    from okf_loremaster.run import draft_charter, parse_vocab
    from okf_loremaster.ui.pauses import render_charter

    with _reported():
        drafted = asyncio.run(
            draft_charter(
                prompt,
                out=out,
                vocab=parse_vocab(vocab),
                target_papers=target_papers,
                topic_min=topic_min,
                topic_max=topic_max,
                console=console,
                verbose=verbose,
            )
        )

    # The same surface the build pauses on, so what is reviewed here is what is
    # reviewed there — minus the question, because there is nothing here to approve.
    render_charter(console, drafted)
    console.print(f"[green]wrote[/green] {out}")
    console.print(
        "[dim]Edit it — especially [bold]vocabularies[/bold] — then run "
        f"[cyan]okf-loremaster build --charter {out}[/cyan].[/dim]"
    )


@app.command()
def build(
    prompt: Annotated[
        str | None, typer.Argument(help="Task description. Omit when using --charter.")
    ] = None,
    charter_file: Annotated[
        Path | None, typer.Option("--charter", help="Build from an existing charter.yaml.")
    ] = None,
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Bundle output path.")] = None,
    pool_size: Annotated[int, typer.Option(help="Candidate pool before screening.")] = 800,
    screen_budget: Annotated[int, typer.Option(help="Max abstracts sent to the screener.")] = 400,
    target_papers: Annotated[int, typer.Option(help="Target retained paper count.")] = 180,
    topic_min: Annotated[int, typer.Option(help="Minimum papers per topic.")] = 8,
    topic_max: Annotated[int, typer.Option(help="Maximum papers per topic.")] = 40,
    max_rounds: Annotated[
        int, typer.Option(help="Search rounds, including the first. 1 disables re-query.", min=1)
    ] = 2,
    vocab: Annotated[
        str | None, typer.Option("--vocab", help="Comma-separated vocabularies; overrides charter.")
    ] = None,
    index: Annotated[
        bool, typer.Option("--index/--no-index", help="Build the vector index.")
    ] = False,
    review: Annotated[bool, typer.Option("--review", help="Human sign-off before emit.")] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive", "-i", help="Stop at the charter and the pool, and ask before going on."
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Plan and cost the run. Makes zero LLM calls.")
    ] = False,
    resume: Annotated[str | None, typer.Option("--resume", help="Resume a run by id.")] = None,
    tui: Annotated[bool, typer.Option("--tui", help="Full-screen Textual interface.")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Emit machine-readable events.")] = False,
    verbose: Annotated[int, typer.Option("-v", "--verbose", count=True, help="Verbosity.")] = 0,
) -> None:
    """Build a bundle end to end."""
    from okf_loremaster.curation import MAX_ROUNDS
    from okf_loremaster.run import RunOptions, build_run, parse_vocab, require_textual
    from okf_loremaster.ui.plain import rich_enabled
    from okf_loremaster.ui.summary import render_bundle, render_extraction, render_topics

    if tui and json_out:
        # Refused rather than degraded: a full-screen app writes escape sequences over
        # the stream --json exists to keep parsable.
        console.print(
            "[red]--tui cannot be combined with --json[/red] — one paints a terminal, "
            "the other feeds a program."
        )
        raise typer.Exit(code=1)
    if index and dry_run:
        # A dry run writes no bundle, and the index is built by reading one back.
        console.print("[red]--index cannot be combined with --dry-run[/red] — a dry run "
                      "writes no bundle to index.")
        raise typer.Exit(code=1)
    if review and (dry_run or json_out):
        # Not a usability nicety. `--json` has nobody to ask and a dry run writes no
        # bundle — signing under either would stamp `verified: human:<id>` on work no
        # human looked at, which is the one claim in the format that has to be true.
        # `--review` is independent of `--interactive`: signing off on a finished bundle
        # and steering the search are different moments, and either can be wanted alone.
        blocking = ", ".join(
            flag for flag, on in (("--dry-run", dry_run), ("--json", json_out)) if on
        )
        console.print(
            f"[red]--review cannot be combined with {blocking}[/red] — sign-off has to be "
            "given by a person who saw the bundle."
        )
        raise typer.Exit(code=1)
    if max_rounds > MAX_ROUNDS:
        # A cap, not a preference. A third round re-screens a pool to ask the question
        # the second one already failed to answer.
        console.print(
            f"[red]--max-rounds {max_rounds} exceeds the hard cap of {MAX_ROUNDS}[/red]"
        )
        raise typer.Exit(code=1)

    # Declined rather than refused, because neither is the user asking for something
    # incoherent. A dry run's deliverable *is* the printed plan, and a full-screen app
    # takes the screen back when it closes; a terminal that cannot drive a live region
    # cannot drive an app either.
    full_screen = tui
    if tui and dry_run:
        console.print("[dim]note[/dim]  --dry-run prints its plan, so --tui is not used here.")
        full_screen = False
    elif tui and not rich_enabled():
        console.print("[dim]note[/dim]  no terminal to drive — falling back from --tui.")
        full_screen = False
    if full_screen:
        # Before the screen is cleared, so a missing extra is readable.
        with _reported():
            require_textual()

    options = RunOptions(
        prompt=prompt or "",
        charter_path=charter_file,
        out=out,
        pool_size=pool_size,
        screen_budget=screen_budget,
        target_papers=target_papers,
        topic_min=topic_min,
        topic_max=topic_max,
        max_rounds=max_rounds,
        vocab=parse_vocab(vocab),
        interactive=interactive,
        review=review,
        index=index,
        dry_run=dry_run,
        resume=resume,
        tui=full_screen,
        json_out=json_out,
        verbose=verbose,
    )

    with _reported():
        if full_screen:
            from okf_loremaster.ui.tui import build_run_tui

            state, directory = asyncio.run(build_run_tui(options))
        else:
            state, directory = asyncio.run(build_run(options, console=console))

    if json_out:
        return
    render_topics(console, state)
    render_extraction(console, state)
    render_bundle(console, state)
    console.print(f"[dim]run directory[/dim]  {directory}")
    if not state.get("pool"):
        raise typer.Exit(code=1)
    # A bundle that failed the gate is still on disk and still worth reading; the exit
    # code is what says so to whatever ran us.
    if state.get("bundle") and not state.get("validated"):
        raise typer.Exit(code=1)


@app.command()
def index(
    bundle: Annotated[Path, typer.Argument(help="Bundle directory to index.")],
) -> None:
    """Build the vector index from an existing bundle.

    The store is written to `<bundle>.chroma`, beside the bundle rather than inside it,
    and an existing collection there is replaced. Re-running this is how a bundle edited
    by hand gets an index that matches it.
    """
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from okf_loremaster.config import load_settings
    from okf_loremaster.emitters.vectors import build_index
    from okf_loremaster.run import embedder

    with _reported():
        settings = load_settings()
        warning = settings.hf_home_warning()
        if warning:
            console.print(f"[yellow]![/yellow]  {warning}")
        model = embedder(settings)
        revision = settings.embed_revision or "unpinned"
        console.print(f"[dim]embedding with[/dim] {settings.embed_model} @ {revision}")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            # Total is unknown until the bundle has been walked, which happens inside
            # `build_index`; the first progress call supplies it.
            task = progress.add_task("embedding", total=None)

            def advance(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total)

            result = asyncio.run(build_index(bundle, embedder=model, on_progress=advance))

    for note in result.warnings:
        console.print(f"[yellow]![/yellow]  {note}")
    if not result.chunks:
        raise typer.Exit(code=1)

    verb = "rebuilt" if result.replaced else "wrote"
    console.print(f"[green]{verb}[/green] {result.path}  [dim]{result.summary()}[/dim]")
    console.print(
        f"[dim]collection[/dim] {result.collection}  [dim]distance[/dim] {result.distance}  "
        f"[dim]dimensions[/dim] {result.dimensions}"
    )
    console.print(
        "[dim]The three per-row keys — timing, confidence, evidence_type — are empty on a "
        'concept chunk. Filter on them with chunk_level == "predictor".[/dim]'
    )


@app.command()
def validate(
    bundle: Annotated[Path, typer.Argument(help="Bundle directory to check.")],
) -> None:
    """Check a bundle against the OKF contract."""
    from okf_loremaster.config import load_settings
    from okf_loremaster.okf.validate import Severity, validate_bundle

    with _reported():
        settings = load_settings()
        report = validate_bundle(bundle, embed_model=settings.embed_model)

    for finding in report.findings:
        color = "red" if finding.severity is Severity.ERROR else "yellow"
        console.print(
            f"[{color}]{finding.severity.value}[/{color}]  "
            f"{finding.line(relative_to=bundle)}"
        )

    console.print(report.summary())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def export(
    bundle: Annotated[Path, typer.Argument(help="Bundle directory to export.")],
    out: Annotated[Path, typer.Option("-o", "--out", help="Destination directory.")],
    permissive_only: Annotated[
        bool, typer.Option("--permissive-only", help="Only redistributable-licensed documents.")
    ] = False,
) -> None:
    """Copy a bundle out, optionally filtered to redistributable documents.

    The copy is a bundle in its own right — its own indexes, catalog and descriptor id —
    so it validates and attaches on its own. The vector index is not copied: it embeds
    every document in the source, including any the filter just removed. Rebuild it with
    `okf-loremaster index <the copy>`.
    """
    from okf_loremaster.emitters.export import export_bundle

    with _reported():
        result = export_bundle(bundle, out, permissive_only=permissive_only)

    for note in result.warnings:
        console.print(f"[yellow]![/yellow]  {escape(note)}")

    console.print(
        f"[green]wrote[/green] {result.path}  [dim]{result.documents} document(s) across "
        f"{result.topics} topic/topics[/dim]"
    )
    if result.permissive_only:
        console.print(
            f"[dim]left behind[/dim]  {result.omitted_count} document(s) under a license "
            f"that does not permit redistribution"
        )
    console.print(
        f"[dim]check it with[/dim] [cyan]okf-loremaster validate {result.path}[/cyan]"
    )


@app.command()
def inspect(
    bundle: Annotated[Path, typer.Argument(help="Bundle directory to summarize.")],
) -> None:
    """Summarize a bundle: topic sizes, designs, coverage, cost."""
    from okf_loremaster.okf.overview import read_overview
    from okf_loremaster.ui.overview import render_overview

    with _reported():
        overview = read_overview(bundle)

    render_overview(console, overview)
    if not overview.documents:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
