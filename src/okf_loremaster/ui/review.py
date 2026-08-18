"""The `--review` sign-off surface. UI, not a node.

The one design decision worth stating: this shows a whole rendered concept file before
it asks. Counts and a verification line say the run behaved; they say nothing about
whether the files are any good, and a sign-off on something nobody looked at is a
signature on a blank page. The specimen is chosen rather than sampled — the paper with
the most predictor rows shows the most of the format in one screen, and a run whose
tables are wrong is wrong there first.

`signoff_view` returns renderables and `ConsoleReviewer` prints them, so the Textual
review screen shows the same specimen and the same tables rather than a second opinion
about what matters.

Everything is printed to stderr like the rest of the interactive surface, so a bundle
path on stdout stays pipeable.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console, RenderableType
from rich.markup import escape
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from okf_loremaster.emitters.okf import document_for
from okf_loremaster.review import Signoff
from okf_loremaster.schemas import ConceptRecord, TextBasis, VerificationSummary

__all__ = ["ConsoleReviewer", "signoff_caption", "signoff_view"]

# Warnings shown before the question. All of them would bury the question itself; the
# rest are in the bundle's own `log.md` and root index, which is where they live anyway.
MAX_WARNINGS = 12

# Named casualties printed under the verification line, across every check. The line
# above them already carries the counts.
MAX_EXAMPLES = 5


def signoff_view(
    records: Sequence[ConceptRecord],
    *,
    topics: dict[str, list[str]],
    verification: VerificationSummary | None,
    warnings: Sequence[str],
    abstracts: bool = True,
) -> list[RenderableType]:
    """What a reviewer looks at before signing. Empty `records` yields the rule alone.

    `abstracts` is `--no-abstract`, and it is here rather than left at its default so the
    specimen is the file that will be written. Showing a section the bundle will not carry
    would make the one thing this screen promises — that a signature is given to something
    somebody saw — untrue in the only way that matters.
    """
    view: list[RenderableType] = [Rule("[bold]review[/bold]")]
    if not records:
        view.append(
            Text.from_markup(
                "[yellow]nothing was extracted — there is nothing to sign off[/yellow]"
            )
        )
        return view

    view.append(_topic_table(records, topics))
    view.extend(_verification(verification))
    view.extend(_warnings(warnings))
    view.extend(_specimen(records, abstracts=abstracts))
    return view


def signoff_caption(signer: str, count: int) -> Text:
    """What signing actually does to the files, said before the question is asked."""
    return Text.from_markup(
        f'[dim]Signing writes [bold]verified: [{{by: "{escape(signer)}", at: ...}}][/bold] '
        f"into all {count} files. Declining still emits the bundle, at the unverified "
        f"tier.[/dim]"
    )


class ConsoleReviewer:
    """Shows the bundle-to-be and asks whether to attest to it."""

    def __init__(
        self, signer: str, console: Console | None = None, *, abstracts: bool = True
    ) -> None:
        self._signer = signer
        self._console = console if console is not None else Console(stderr=True)
        self._abstracts = abstracts

    async def sign_off(
        self,
        records: Sequence[ConceptRecord],
        *,
        topics: dict[str, list[str]],
        verification: VerificationSummary | None,
        warnings: Sequence[str],
    ) -> Signoff:
        console = self._console
        for item in signoff_view(
            records,
            topics=topics,
            verification=verification,
            warnings=warnings,
            abstracts=self._abstracts,
        ):
            console.print(item)

        if not records:
            return Signoff.declined("no records")

        console.print(signoff_caption(self._signer, len(records)))
        from rich.prompt import Confirm

        approved = Confirm.ask(
            f"Sign off on these {len(records)} documents as {self._signer}?",
            console=console,
            default=False,
        )
        if not approved:
            return Signoff.declined("declined at review")
        return Signoff.granted(self._signer)


# --- what a reviewer looks at ----------------------------------------------


def _topic_table(records: Sequence[ConceptRecord], topics: dict[str, list[str]]) -> Table:
    by_topic: dict[str, list[ConceptRecord]] = {slug: [] for slug in topics}
    for record in records:
        by_topic.setdefault(record.domain, []).append(record)

    table = Table(title="what would be written", title_style="bold")
    table.add_column("topic", style="cyan", no_wrap=True)
    table.add_column("papers", justify="right")
    table.add_column("full text", justify="right", style="dim")
    table.add_column("predictor rows", justify="right")
    table.add_column("null findings", justify="right")
    table.add_column("permissive", justify="right", style="dim")

    for slug, items in by_topic.items():
        full = sum(1 for r in items if r.text_basis is TextBasis.FULL_TEXT)
        rows = sum(len(r.extraction.predictors) for r in items)
        nulls = sum(1 for r in items if r.extraction.reports_null_findings)
        table.add_row(
            slug,
            str(len(items)),
            str(full),
            str(rows),
            str(nulls),
            str(sum(1 for r in items if r.export_safe)),
        )
    table.add_section()
    table.add_row(
        "total",
        str(len(records)),
        str(sum(1 for r in records if r.text_basis is TextBasis.FULL_TEXT)),
        str(sum(len(r.extraction.predictors) for r in records)),
        str(sum(1 for r in records if r.extraction.reports_null_findings)),
        str(sum(1 for r in records if r.export_safe)),
    )
    return table


def _verification(verification: VerificationSummary | None) -> list[RenderableType]:
    if verification is None:
        return []
    mark = "[green]ok[/green]" if verification.clean else "[yellow]![/yellow]"
    view: list[RenderableType] = [
        Text.from_markup(f"{mark}  numeric verification: {verification.line()}")
    ]
    view.extend(
        Text.from_markup(f"[dim]      {escape(example.note)}[/dim]")
        for example in verification.examples[:MAX_EXAMPLES]
    )
    return view


def _warnings(warnings: Sequence[str]) -> list[RenderableType]:
    view: list[RenderableType] = [
        Text.from_markup(f"[yellow]![/yellow]  {escape(warning)}")
        for warning in list(warnings)[:MAX_WARNINGS]
    ]
    remaining = len(warnings) - MAX_WARNINGS
    if remaining > 0:
        view.append(
            Text.from_markup(f"[dim]      and {remaining} more, all of them in log.md[/dim]")
        )
    return view


def _specimen(records: Sequence[ConceptRecord], *, abstracts: bool = True) -> list[RenderableType]:
    specimen = max(records, key=lambda record: len(record.extraction.predictors))
    return [
        Rule(f"[dim]{specimen.domain}/{specimen.filename}[/dim]"),
        Syntax(
            document_for(specimen, abstracts=abstracts),
            "markdown",
            theme="ansi_dark",
            word_wrap=True,
        ),
    ]
