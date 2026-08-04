"""Writing the markdown a bundle is made of — the inverse of `reader.py`.

Here rather than in the emitter because there are now two writers. `emitters/okf.py`
renders a bundle from a finished run and `emitters/export.py` renders a filtered copy of
one from files on disk, and both have to produce tables that `reader.markdown_table`
reads back cell for cell. Two private copies of `_cell` would be two places for the
escaping to drift, and the drift would be invisible: a table that round-trips wrong still
renders fine.

Three rules, all of them load-bearing for the round trip:

- **A cell is one line, pipes escaped.** A newline would end the row and a bare pipe
  would open a column that is not there.
- **An empty cell is `NONE_CELL`, never blank.** A row's shape has to survive a reader
  that collapses whitespace, and "nothing here" reads better as something deliberate.
- **A fact line is `- **Label** — value`.** `reader.fact_list` matches exactly that, so
  the em dash is syntax rather than typography.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from okf_loremaster.okf.layout import NONE_CELL

__all__ = ["cell", "facts", "inline", "table_row", "table_rule"]


def inline(text: str) -> str:
    """Collapse a value onto one line. Markdown treats a blank line as a paragraph."""
    return " ".join(str(text).split())


def cell(text: str) -> str:
    """One table cell: single line, pipes escaped, never empty."""
    collapsed = inline(text).replace("|", "\\|")
    return collapsed or NONE_CELL


def table_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cell(value) for value in cells) + " |"


def table_rule(columns: int) -> str:
    return "|" + "---|" * columns


def facts(pairs: Iterable[tuple[str, str]]) -> list[str]:
    """`- **Label** — value` lines, skipping the pairs with nothing to say."""
    return [f"- **{label}** — {inline(value)}" for label, value in pairs if value]
