"""The two confirmation pauses, and what they show.

The graph stops after `charter` and after `rank` because those are the last moments a
person can act cheaply. Everything before the charter pause is one model call;
everything after the retrieve pause is thousands. A charter with a wrong topic taxonomy
or a missing vocabulary is fixable in ten seconds at the first pause and expensive to
notice at the last.

**A run is autonomous unless `--interactive` asks otherwise**, so both moments are
printed on every run and only *asked* on an interactive one. The graph is interrupted
either way: the interrupt is what writes the checkpoint that `--resume` needs, and
skipping the question is not the same as skipping the stop.

This is UI, not a node — nodes never print, and these do nothing else. `Pause` is a
protocol so the orchestrator can be driven by a console, by an autonomous run, by the
TUI, or by a test without knowing which.

**What a pause shows is built as renderables, not printed.** `charter_view` and
`retrieve_view` return a list; `ConsolePause` prints it and the Textual pause screen
scrolls it. Two surfaces asking the same question have to be showing the same thing, and
the only way to guarantee that is for there to be one thing.

**Asking is `async` even though `Confirm.ask` is not.** A decision surface is I/O, and
the TUI answers by awaiting a modal screen — a synchronous protocol would leave it no
way to do that but a second event loop on a second thread, with the event bus straddling
the two. The console implementation still blocks the loop while the prompt is up, which
is what it did before and is correct: nothing else needs to run while a person types.

Query terms are printed through `rich.markup.escape`: a PubMed query is full of
`[tiab]` and `[All Fields]`, which Rich would otherwise read as markup tags and
silently swallow — turning the one string the pause exists to show into a blank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rich.console import Console, RenderableType
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from okf_loremaster.graph.state import RunState
from okf_loremaster.llm.estimate import SpendEstimate
from okf_loremaster.schemas import Charter, ExecutedQuery, ScoredCandidate

__all__ = [
    "AutoApprove",
    "ConsolePause",
    "Pause",
    "PauseDecision",
    "charter_view",
    "render_charter",
    "retrieve_view",
]

# How many titles the retrieve pause prints. Enough to tell a well-aimed corpus from a
# badly aimed one at a glance; short enough to read.
TOP_TITLES = 20


@dataclass(frozen=True, slots=True)
class PauseDecision:
    proceed: bool
    reason: str = ""


class Pause(Protocol):
    """What the orchestrator needs from a confirmation surface.

    Async because the TUI answers by awaiting a modal screen; see the module docstring.
    """

    async def charter(self, charter: Charter) -> PauseDecision: ...

    async def retrieve(
        self, state: RunState, *, estimate: SpendEstimate | None
    ) -> PauseDecision: ...


class AutoApprove:
    """Proceeds without asking or printing. Used by tests and by `--json`."""

    async def charter(self, charter: Charter) -> PauseDecision:
        return PauseDecision(proceed=True)

    async def retrieve(self, state: RunState, *, estimate: SpendEstimate | None) -> PauseDecision:
        return PauseDecision(proceed=True)


def charter_view(charter: Charter) -> list[RenderableType]:
    """Everything a person needs to accept or reject a charter, as renderables.

    Returned rather than printed so the console pause and the Textual pause screen show
    the same thing without either owning it.
    """
    view: list[RenderableType] = [Rule("[bold]charter[/bold]")]

    facts = Table.grid(padding=(0, 2))
    facts.add_column(justify="right", style="dim")
    facts.add_column()
    facts.add_row("task", charter.task or "[dim]none[/dim]")
    if charter.population:
        facts.add_row("population", charter.population)
    if charter.outcome:
        facts.add_row("outcome", charter.outcome)
    facts.add_row("target", f"{charter.target_papers} papers")
    facts.add_row("per topic", f"{charter.topic_min}-{charter.topic_max}")
    if charter.min_year:
        facts.add_row("from", str(charter.min_year))
    facts.add_row("languages", ", ".join(charter.languages) or "[dim]any[/dim]")
    view.append(facts)

    if charter.inclusion or charter.exclusion:
        criteria = Table.grid(padding=(0, 2))
        criteria.add_column(justify="right", style="dim")
        criteria.add_column()
        for label, items in (("include", charter.inclusion), ("exclude", charter.exclusion)):
            for index, item in enumerate(items):
                criteria.add_row(label if index == 0 else "", item)
        view.append(criteria)

    view.append(_topic_table(charter))

    # Directly under the taxonomy, deliberately: this is the field that gates every
    # later extraction and it is the one a reader skims past.
    if charter.vocabularies:
        view.append(
            Text.from_markup("[bold]vocabularies[/bold]  " + ", ".join(charter.vocabularies))
        )
    else:
        view.append(Text.from_markup("[yellow]vocabularies  none[/yellow]"))
    view.append(
        Text.from_markup(
            "[dim]  these gate what every extraction may record. Override with "
            "--vocab a,b,c, or edit `vocabularies` in charter.yaml.[/dim]"
        )
    )

    view.extend(Text.from_markup(f"[yellow]![/yellow]  {p}") for p in charter.problems())
    return view


def render_charter(console: Console, charter: Charter) -> None:
    """Print a charter for review.

    A free function because the `charter` command shows exactly this and has nothing to
    ask about; the pause is the same rendering with a question after it.
    """
    for item in charter_view(charter):
        console.print(item)


def _topic_table(charter: Charter) -> Table:
    table = Table(title="topic_taxonomy", title_style="bold", show_lines=False)
    table.add_column("slug", style="cyan", no_wrap=True)
    table.add_column("title")
    table.add_column("scope", overflow="fold")
    table.add_column("seed terms", overflow="fold", style="dim")
    for topic in charter.topic_taxonomy:
        table.add_row(topic.slug, topic.title, topic.scope, ", ".join(topic.seed_terms))
    if not charter.topic_taxonomy:
        table.add_row("[yellow]none[/yellow]", "", "", "")
    return table


def retrieve_view(state: RunState, *, estimate: SpendEstimate | None) -> list[RenderableType]:
    """Everything a person needs to accept or reject the pool, as renderables."""
    executed = list(state.get("executed") or [])
    unique = list(state.get("unique") or [])
    pool = list(state.get("pool") or [])
    dropped = dict(state.get("dropped") or {})

    view: list[RenderableType] = [Rule("[bold]retrieved[/bold]"), _query_table(executed)]

    totals = Table.grid(padding=(0, 2))
    totals.add_column(justify="right", style="dim")
    totals.add_column()
    totals.add_row("hits", f"{sum(q.count for q in executed):,} across {len(executed)} queries")
    totals.add_row("retrieved", f"{sum(q.retrieved for q in executed):,}")
    totals.add_row("unique", f"{len(unique):,}")
    drops = ", ".join(f"{count} {reason}" for reason, count in dropped.items() if count)
    if drops:
        totals.add_row("dropped", drops)
    totals.add_row("pool", f"{len(pool):,} to be screened")
    view.append(totals)

    view.append(_titles(pool))
    view.extend(_comparison(state))
    view.extend(_estimate(estimate))
    view.extend(
        Text.from_markup(f"[yellow]![/yellow]  {w}") for w in list(state.get("warnings") or [])
    )
    return view


def _query_table(executed: list[ExecutedQuery]) -> Table:
    table = Table(title="queries", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("term", overflow="fold")
    table.add_column("hits", justify="right")
    table.add_column("got", justify="right", style="dim")
    for index, query in enumerate(executed, start=1):
        flag = " [yellow]suspect[/yellow]" if query.suspect else ""
        table.add_row(
            str(index),
            escape(query.term) + flag,
            f"{query.count:,}",
            str(query.retrieved),
        )
    return table


def _titles(pool: list[ScoredCandidate]) -> Table:
    table = Table(title=f"top {min(TOP_TITLES, len(pool))} of the pool", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right", style="dim")
    table.add_column("pmid", style="cyan", no_wrap=True)
    table.add_column("title", overflow="fold")
    table.add_column("source", style="dim", overflow="fold")
    for index, item in enumerate(pool[:TOP_TITLES], start=1):
        table.add_row(
            str(index),
            f"{item.score:.3f}",
            item.pmid,
            escape(item.candidate.title),
            escape(item.candidate.citation()),
        )
    return table


def _comparison(state: RunState) -> list[RenderableType]:
    comparison = state.get("comparison")
    if comparison is None:
        return []
    view: list[RenderableType] = [
        Text.from_markup(f"[bold]diversification[/bold]  {comparison.summary()}")
    ]
    topics = sorted(set(comparison.pure_by_topic) | set(comparison.diversified_by_topic))
    if not topics:
        return view
    table = Table(title="pool by topic affinity", title_style="dim")
    table.add_column("topic", style="cyan")
    table.add_column("pure rank", justify="right")
    table.add_column("MMR + quota", justify="right")
    table.add_column("delta", justify="right")
    for topic in topics:
        before = comparison.pure_by_topic.get(topic, 0)
        after = comparison.diversified_by_topic.get(topic, 0)
        delta = after - before
        color = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        table.add_row(topic, str(before), str(after), f"[{color}]{delta:+d}[/{color}]")
    view.append(table)
    return view


def _estimate(estimate: SpendEstimate | None) -> list[RenderableType]:
    if estimate is None:
        return []
    table = Table(title="projected spend", title_style="bold")
    table.add_column("node", style="cyan")
    table.add_column("role", style="dim")
    table.add_column("calls", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("basis", overflow="fold", style="dim")
    for node in estimate.nodes:
        table.add_row(
            node.node,
            node.role.value,
            f"{node.calls:,}",
            f"{node.tokens:,}",
            node.format_usd(),
            node.basis,
        )
    table.add_section()
    table.add_row(
        "total", "", f"{estimate.calls:,}", f"{estimate.tokens:,}", estimate.format_usd(), ""
    )
    view: list[RenderableType] = [table]
    view.extend(Text.from_markup(f"[dim]  {note}[/dim]") for note in estimate.notes)
    return view


class ConsolePause:
    """Prints the decision surface, and asks unless told not to.

    `interactive=False` is the default surface for an autonomous run, and what a dry run
    uses. It still prints everything: the information is the point, and a run whose
    output is being piped into a log wants it more, not less.
    """

    def __init__(self, console: Console | None = None, *, interactive: bool = True) -> None:
        self._console = console if console is not None else Console(stderr=True)
        self._interactive = interactive

    async def charter(self, charter: Charter) -> PauseDecision:
        self._show(charter_view(charter))
        return self._ask("Proceed with this charter?")

    async def retrieve(self, state: RunState, *, estimate: SpendEstimate | None) -> PauseDecision:
        self._show(retrieve_view(state, estimate=estimate))
        pool = list(state.get("pool") or [])
        return self._ask(f"Screen these {len(pool)} papers?")

    # --- shared ------------------------------------------------------------

    def _show(self, view: list[RenderableType]) -> None:
        for item in view:
            self._console.print(item)

    def _ask(self, question: str) -> PauseDecision:
        """Blocking on purpose — nothing else should run while a person is typing."""
        if not self._interactive:
            self._console.print(
                f"[dim]{question} — continuing without asking; --interactive stops here[/dim]"
            )
            return PauseDecision(proceed=True)
        from rich.prompt import Confirm

        proceed = Confirm.ask(question, console=self._console, default=True)
        return PauseDecision(proceed=proceed, reason="" if proceed else "declined at pause")
