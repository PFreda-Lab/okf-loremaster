"""Printing what `inspect` read. UI, not a reader.

Separate from `okf/overview.py` for the reason every renderer in this package is separate
from what it renders: the counting has to be testable without a terminal, and the reader
has to stay importable by anything that wants the numbers rather than the picture.

The order is the order someone reads it in: where and how big, then the topics, then
what the corpus is made of, then the run behind it. Verification gets its own line rather
than a column, because "of 412 predictor rows, 388 carry a magnitude and 6 read
`unverified`" is the sentence that says how quotable this bundle is, and a number in a
table is not a sentence.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from okf_loremaster.okf.layout import vector_store_path
from okf_loremaster.okf.overview import BundleOverview

__all__ = ["render_overview"]

# Problems printed in full before the rest are counted. A bundle mangled by a sync client
# produces one per file, and a screen of them buries the summary they are attached to.
MAX_PROBLEMS = 8

# Facts lifted off the root index, in the order they are printed. The labels are the ones
# `emitters/okf.py` writes; a bundle from another tool simply has fewer of them.
_RUN_FACTS = ("Run id", "Built", "Duration", "Tool", "Models", "Signed off by", "Stale after")
_COST_FACTS = ("Model calls", "Tokens", "Spend")


def render_overview(console: Console, overview: BundleOverview) -> None:
    """Print a whole bundle summary."""
    console.print(f"[bold cyan]{escape(overview.name)}[/bold cyan]")
    console.print(f"[dim]{escape(str(overview.path))}[/dim]")
    if overview.resource_id:
        console.print(f"[dim]resource id[/dim]  {escape(overview.resource_id)}")

    _render_topics(console, overview)
    _render_corpus(console, overview)
    _render_designs(console, overview)
    _render_vocabularies(console, overview)
    _render_run(console, overview)
    _render_vectors(console, overview)
    _render_problems(console, overview)


def _render_topics(console: Console, overview: BundleOverview) -> None:
    if not overview.topics:
        console.print("[yellow]![/yellow]  no topics — the bundle holds nothing")
        return

    table = Table(title="topics", title_style="bold")
    table.add_column("topic", style="cyan", no_wrap=True)
    table.add_column("title", overflow="fold")
    table.add_column("papers", justify="right")
    table.add_column("full text", justify="right")
    table.add_column("predictors", justify="right")
    table.add_column("permissive", justify="right")

    for topic in overview.topics:
        count = str(topic.documents)
        if not topic.documents:
            count = f"[yellow]{count}[/yellow]"
        table.add_row(
            topic.slug,
            escape(topic.title),
            count,
            str(topic.full_text),
            str(topic.predictors),
            str(topic.exportable),
        )
    table.add_section()
    table.add_row(
        "total",
        "",
        str(overview.documents),
        str(overview.full_text),
        str(overview.predictors),
        str(overview.exportable),
    )
    console.print(table)


def _render_corpus(console: Console, overview: BundleOverview) -> None:
    console.rule("[bold]corpus[/bold]")
    console.print(
        f"  {overview.documents} paper(s): {overview.full_text} read from full text, "
        f"{overview.abstract_only} from the abstract only"
    )
    median = overview.median_n
    if median is not None:
        console.print(f"  median reported sample size: {median:,}")
    console.print(
        f"  {overview.predictors} predictor row(s); "
        f"{overview.reporting_nulls} paper(s) report a null or non-significant finding"
    )

    # The verification line. `unverified` is a magnitude the numeric check removed, which
    # is a different claim from a paper that reported none — and the only place the
    # difference is visible is here.
    stated = overview.with_effect + overview.unverified
    unstated = overview.predictors - stated
    line = f"  effect sizes: {overview.with_effect} verified"
    if overview.unverified:
        line += f", [yellow]{overview.unverified} unverified[/yellow]"
    console.print(f"{line}, {unstated} row(s) reported none")

    console.print(
        f"  {overview.exportable} of {overview.documents} carry a license that permits "
        f"redistribution"
    )
    if overview.untagged:
        console.print(
            f"[yellow]![/yellow]  {overview.untagged} document(s) have no tags — retrieval "
            f"matches over title, tags and journal, so they are findable by title alone"
        )
    for note in overview.notes:
        console.print(f"[yellow]![/yellow]  {escape(note)}")


def _render_designs(console: Console, overview: BundleOverview) -> None:
    if not overview.designs:
        return
    table = Table(title="study designs", title_style="bold", box=None, pad_edge=False)
    table.add_column("design", overflow="fold")
    table.add_column("papers", justify="right")
    for design, count in overview.designs:
        table.add_row(escape(design), str(count))
    console.print(table)


def _render_vocabularies(console: Console, overview: BundleOverview) -> None:
    """Which coding vocabularies the corpus actually carries hints for.

    The charter asks for a set of them; this is how many papers yielded anything under
    each, which is the difference between a vocabulary being requested and being useful.
    """
    if not overview.vocabularies:
        return
    table = Table(title="vocabulary hints", title_style="bold", box=None, pad_edge=False)
    table.add_column("key", style="cyan", overflow="fold")
    table.add_column("papers", justify="right")
    for key, count in overview.vocabularies:
        table.add_row(escape(key), str(count))
    console.print(table)


def _render_run(console: Console, overview: BundleOverview) -> None:
    facts = overview.index_facts
    shown = [(label, facts[label]) for label in (*_RUN_FACTS, *_COST_FACTS) if facts.get(label)]
    if not shown:
        return
    console.rule("[bold]run[/bold]")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="dim")
    grid.add_column(overflow="fold")
    for label, value in shown:
        grid.add_row(label.lower(), escape(value))
    console.print(grid)


def _render_vectors(console: Console, overview: BundleOverview) -> None:
    store = overview.vectors
    if not store:
        console.print(
            f"[dim]no vector index — build one with "
            f"[cyan]okf-loremaster index {escape(str(overview.path))}[/cyan][/dim]"
        )
        return
    revision = escape(str(store.get("embedding_revision") or "")) or "[yellow]unpinned[/yellow]"
    detail = ", ".join(
        part
        for part in (
            f"{store['chunks']} chunks" if store.get("chunks") else "",
            f"{store['dimensions']}d" if store.get("dimensions") else "",
            escape(str(store.get("distance") or "")),
        )
        if part
    )
    console.print(
        f"  vectors [cyan]{escape(vector_store_path(overview.path).name)}[/cyan]  "
        f"[dim]{escape(str(store.get('embedding_model') or 'unnamed'))} @ {revision}"
        + (f", {detail}" if detail else "")
        + "[/dim]"
    )


def _render_problems(console: Console, overview: BundleOverview) -> None:
    if not overview.problems:
        return
    console.print(f"[red]{len(overview.problems)} file(s) could not be read:[/red]")
    for path, why in overview.problems[:MAX_PROBLEMS]:
        console.print(f"[red]      {escape(path.name)}: {escape(why)}[/red]")
    if len(overview.problems) > MAX_PROBLEMS:
        console.print(f"[dim]      and {len(overview.problems) - MAX_PROBLEMS} more[/dim]")
