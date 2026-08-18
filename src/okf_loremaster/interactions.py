"""Reciprocity between predictor rows, computed. No model call.

An interaction is a claim about two variables, and a paper states it once — from
whichever side its sentence happened to start on. A model asked to state both halves
states them inconsistently and charges for the second one, so the second half is derived
here instead: if row 1 says *maternal age is correlated with parity* and parity is
row 3 of the same table, row 3 gains *correlated with maternal age*, marked `mirrored`.

Three rules keep that from inventing anything.

**Only inside one paper.** The other side of a mirror must be a predictor row in the same
document. A name that matches nothing in the table is left exactly as the extraction
wrote it — it is still a real variable the study used, it just is not a row, so there is
nowhere to mirror it to.

**Matching is exact after folding.** Case, punctuation and whitespace collapse and
nothing else does. The lexical clustering in `recurrence.py` is documented as timid for
the same reason and this is timider still: a wrong merge here does not blur an index
entry, it attributes a coefficient to a variable the paper never measured it against.

**A directional relationship flips.** `modifies` mirrors as `modified by`, `derived
from` as `derives`. Correlation and mutual exclusivity are symmetric and mirror as
themselves. Keeping the forward label on the mirrored half would reverse the claim, which
is the one failure that would make this worse than not mirroring at all.

The banding itself is not here — it is `schemas.common.band_interaction`, beside the
vocabulary it produces, because it is a property of the measure rather than of the paper.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from okf_loremaster.schemas import Extraction, Interaction, PredictorRow

__all__ = [
    "fold_variable",
    "interaction_rows",
    "mirror_interactions",
    "same_variable",
    "variable_rows",
]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def same_variable(left: str, right: str) -> str:
    """The folded form two names share, or `""` if they are not the same name.

    Returned rather than a boolean so a caller can key a lookup on it without folding
    twice, and so the empty string — a name with no alphanumeric content — can never
    match another empty one.
    """
    folded = fold_variable(left)
    return folded if folded and folded == fold_variable(right) else ""


def fold_variable(name: str) -> str:
    """A variable name reduced to what matching is allowed to see.

    Public because the emitter folds too — it resolves a mirrored interaction back to the
    row that stated it, and a second, slightly different folding rule there would print a
    pointer to the wrong row. One rule, one place.
    """
    return " ".join(_NON_ALNUM.sub(" ", name.casefold()).split())


def variable_rows(rows: Sequence[PredictorRow]) -> dict[str, int]:
    """Folded predictor name to the index of the first row carrying it.

    First row wins. Two rows for the same predictor against different outcomes are
    normal, and an interaction belongs to one of them rather than to both — the second is
    reachable from the first through the `#` column either way.
    """
    found: dict[str, int] = {}
    for index, row in enumerate(rows):
        found.setdefault(fold_variable(row.predictor), index)
    return found


def mirror_interactions(extraction: Extraction) -> Extraction:
    """Give every interaction its other half, where the other half is a row here.

    Idempotent: a mirror that already exists is not added twice, and running this over an
    extraction it has already processed changes nothing. That matters because reconcile
    runs it after verification, and a resumed run re-reconciles from the checkpoint.
    """
    rows = extraction.predictors
    if len(rows) < 2:
        return extraction

    by_name = variable_rows(rows)

    added: dict[int, list[Interaction]] = {}
    for index, row in enumerate(rows):
        for interaction in row.interacts_with:
            target = by_name.get(fold_variable(interaction.feature))
            if target is None or target == index:
                continue
            mirror = Interaction(
                feature=row.predictor,
                kind=interaction.kind.inverse,
                measure=interaction.measure,
                value=interaction.value,
                measure_raw=interaction.measure_raw,
                mirrored=True,
            )
            if _already_stated(rows[target].interacts_with, added.get(target, []), mirror):
                continue
            added.setdefault(target, []).append(mirror)

    if not added:
        return extraction
    return extraction.model_copy(
        update={
            "predictors": [
                row
                if index not in added
                else row.model_copy(
                    update={"interacts_with": [*row.interacts_with, *added[index]]}
                )
                for index, row in enumerate(rows)
            ]
        }
    )


def _already_stated(
    existing: Sequence[Interaction], pending: Sequence[Interaction], mirror: Interaction
) -> bool:
    """Whether this row already names that variable, by any relationship.

    Deliberately blind to `kind`. A row that already says something about a variable is a
    row where the extraction had its say, and adding a second, derived line about the same
    pair beside it would read as two findings where the paper reported one.
    """
    folded = fold_variable(mirror.feature)
    return any(fold_variable(other.feature) == folded for other in (*existing, *pending))


def interaction_rows(row: PredictorRow) -> list[Interaction]:
    """One row's interactions, stated ones first.

    Ordering only. What the paper said about *this* variable outranks what was derived
    from what it said about another, and a reader scanning the section should not have to
    check the `mirrored` flag to see which is which.
    """
    return sorted(row.interacts_with, key=lambda entry: entry.mirrored)
