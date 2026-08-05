"""What a finished run printed at the end. UI, not a node.

Separate from `pauses.py` because nothing here asks anything: a pause is a decision
surface and this is a report. The two would otherwise grow into one module that both
blocks and summarizes, and the summary is the part that later steps keep adding to.

The topic table is the first place a person sees whether the run worked. A topic at its
floor with a `missing` line beside it is a finding about the literature, not a defect,
and it is printed as such — the alternative is a bundle whose thin topics are only
discoverable by browsing it.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from okf_loremaster.graph.state import RunState

__all__ = ["render_bundle", "render_extraction", "render_topics"]

# Warnings printed in full. A structurally broken run can accumulate one per document,
# and a wall of them is a wall nobody reads; the rest are in the bundle's own log.
MAX_WARNINGS = 12

# Named casualties printed under the verification line, across every check. The line
# above them already carries the counts, so these are here to show the shape of the
# problem rather than to enumerate it.
MAX_EXAMPLES = 5


def render_topics(console: Console, state: RunState) -> None:
    """Print the final placement, and what came up short."""
    topics = dict(state.get("topics") or {})
    if not topics:
        return

    charter = state.get("charter")
    floor = charter.topic_paper_min if charter is not None else 0
    ceiling = charter.topic_paper_max if charter is not None else 0
    titles = {topic.slug: topic.title for topic in charter.topic_taxonomy} if charter else {}
    curation = state.get("curation")
    gaps = {gap.topic: gap for gap in (curation.gaps if curation is not None else [])}

    console.rule("[bold]curated[/bold]")
    table = Table(title=f"topics ({floor}-{ceiling} papers each)", title_style="bold")
    table.add_column("topic", style="cyan", no_wrap=True)
    table.add_column("title", overflow="fold")
    table.add_column("papers", justify="right")
    table.add_column("still missing", overflow="fold", style="dim")

    for slug, pmids in topics.items():
        gap = gaps.get(slug)
        count = str(len(pmids))
        if gap is not None and gap.shortfall > 0:
            count = f"[yellow]{count}[/yellow]"
        table.add_row(
            slug,
            escape(titles.get(slug, "")),
            count,
            escape(gap.missing) if gap is not None else "",
        )

    kept = sum(len(pmids) for pmids in topics.values())
    table.add_section()
    table.add_row("total", "", str(kept), "")
    console.print(table)

    rounds = int(state.get("rounds") or 1)
    if rounds > 1:
        console.print(
            f"[dim]  {rounds} search rounds: a topic came up short and was refilled[/dim]"
        )

    short = [slug for slug, gap in gaps.items() if gap.shortfall > 0]
    if short:
        console.print(
            f"[yellow]![/yellow]  under the floor of {floor}: {', '.join(sorted(short))} — "
            "the searches found what there was to find"
        )


def render_extraction(console: Console, state: RunState) -> None:
    """Print what was read, and what checking the numbers found.

    Verification gets its own lines rather than a count in a table. A run that silently
    removed a dozen effect sizes and a run that removed none look identical from the
    record count alone, and the difference is the whole reason the check exists.
    """
    records = state.get("records") or []
    texts = state.get("texts") or {}
    if not records and not texts:
        return

    console.rule("[bold]extracted[/bold]")
    full = sum(1 for source in texts.values() if source.is_full_text)
    truncated = sum(1 for source in texts.values() if source.truncated)
    rows = sum(len(record.extraction.predictors) for record in records)
    nulls = sum(1 for record in records if record.extraction.reports_null_findings)
    exportable = sum(1 for record in records if record.export_safe)

    console.print(
        f"  {len(records)} concept(s): {rows} predictor row(s), "
        f"{nulls} reporting a null finding"
    )
    console.print(
        f"  read from full text: {full} of {len(texts)}"
        + (f" ({truncated} truncated to the prompt budget)" if truncated else "")
        + f"; {exportable} carry a permissive license"
    )

    summary = state.get("verification")
    if summary is None:
        return
    if summary.clean:
        console.print(f"[green]ok[/green]  numeric verification: {summary.line()}")
        return
    console.print(f"[yellow]![/yellow]  numeric verification: {summary.line()}")
    for example in summary.examples[:MAX_EXAMPLES]:
        console.print(f"[dim]      {escape(example.note)}[/dim]")


def render_bundle(console: Console, state: RunState) -> None:
    """Print where the bundle went, whether it passed the gate, and what it warned about.

    The warnings block is printed here rather than dripped out as each node produced it.
    A warning seen four minutes before the run ends is a warning nobody acts on; the same
    line beside the bundle path is the moment someone might.
    """
    location = str(state.get("bundle") or "")
    if not location:
        return

    console.rule("[bold]bundle[/bold]")
    console.print(f"  [cyan]{escape(location)}[/cyan]")

    signer = str(state.get("verified_by") or "")
    if signer:
        console.print(f"  verified by [green]{escape(signer)}[/green]")
    else:
        console.print(
            "  [dim]unverified — no human sign-off; rerun with --review to add one[/dim]"
        )

    errors = list(state.get("validation_errors") or [])
    if errors:
        console.print(f"[red]invalid[/red]  {len(errors)} error(s) against the OKF contract:")
        for note in errors[:MAX_WARNINGS]:
            console.print(f"[red]      {escape(note)}[/red]")
        if len(errors) > MAX_WARNINGS:
            console.print(f"[dim]      and {len(errors) - MAX_WARNINGS} more[/dim]")
        console.print(
            f"[dim]  full report: [cyan]okf-loremaster validate {escape(location)}[/cyan][/dim]"
        )
    elif state.get("validated"):
        console.print("[green]ok[/green]  passes the OKF contract")

    store = str(state.get("vector_index") or "")
    if store:
        chunks = int(state.get("vector_chunks") or 0)
        console.print(f"  vectors [cyan]{escape(store)}[/cyan] [dim]({chunks} chunks)[/dim]")

    _render_warnings(console, state, location)


def _render_warnings(console: Console, state: RunState, location: str) -> None:
    # The summary line the validate node appends duplicates the error block above.
    warnings = [
        note
        for note in (state.get("warnings") or [])
        if not note.startswith("the bundle failed validation")
    ]
    if not warnings:
        return
    console.print(f"[yellow]![/yellow]  {len(warnings)} warning(s):")
    for note in warnings[:MAX_WARNINGS]:
        # A rerun command arrives on its own indented line and is worth keeping legible.
        for part in note.splitlines():
            console.print(f"[yellow]      {escape(part)}[/yellow]")
    if len(warnings) > MAX_WARNINGS:
        remaining = len(warnings) - MAX_WARNINGS
        log = Path(location) / "log.md"
        console.print(f"[dim]      and {remaining} more — all of them in {escape(str(log))}[/dim]")
