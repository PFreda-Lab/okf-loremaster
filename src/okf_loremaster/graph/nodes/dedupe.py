"""The dedupe node: drop what should never reach the screener. All code, no judgment.

Search already merges across queries — one paper is fetched once, whatever found it —
so identity duplicates cannot arrive here. What can, and what this removes:

- **Retracted papers.** Dropped outright rather than flagged. A retracted finding in a
  bundle is worse than a missing one, and the per-concept `status` field that would let
  a reader see the flag was deliberately left out of the frontmatter.
- **Papers with no abstract.** Screening reads a title and an abstract; with no
  abstract there is nothing to screen and extraction has nothing to read either.
- **Near-duplicate titles.** The same study reaches PubMed twice often enough to
  matter: a preprint and its journal version, a conference abstract and the paper, a
  corrected reprint. Different PMIDs, so identity does not catch them.

Every drop is counted by reason, and the counts reach the retrieve pause. A corpus that
lost a third of itself here is worth knowing about before anything is spent on it.
"""

from __future__ import annotations

from typing import Any

from okf_loremaster.graph.state import Deps, RunState, span
from okf_loremaster.schemas import Candidate

__all__ = ["dedupe_node"]

NODE = "dedupe"


async def dedupe_node(state: RunState, deps: Deps) -> dict[str, Any]:
    candidates: list[Candidate] = list(state.get("candidates") or [])
    warnings = list(state.get("warnings") or [])

    with span(deps, NODE) as report:
        kept, dropped = _filter(candidates)

        lost = sum(dropped.values())
        if candidates and lost > len(candidates) // 3:
            note = (
                f"dedupe dropped {lost} of {len(candidates)} retrieved records "
                f"({', '.join(f'{n} {reason}' for reason, n in dropped.items() if n)})"
            )
            warnings.append(note)
            deps.warn(NODE, note)

        detail = ", ".join(f"{count} {reason}" for reason, count in dropped.items() if count)
        report["summary"] = f"{len(kept)} kept" + (f"; dropped {detail}" if detail else "")

    return {"unique": kept, "dropped": dropped, "warnings": warnings}


def _filter(candidates: list[Candidate]) -> tuple[list[Candidate], dict[str, int]]:
    dropped = {"retracted": 0, "no abstract": 0, "duplicate title": 0}
    kept: list[Candidate] = []
    # Normalized title to the index in `kept` that claimed it.
    by_title: dict[str, int] = {}

    for candidate in candidates:
        if candidate.is_retracted:
            dropped["retracted"] += 1
            continue
        if not candidate.has_abstract:
            dropped["no abstract"] += 1
            continue

        title = candidate.normalized_title
        if not title:
            kept.append(candidate)
            continue

        existing = by_title.get(title)
        if existing is None:
            by_title[title] = len(kept)
            kept.append(candidate)
            continue

        dropped["duplicate title"] += 1
        # Keep the better sighting and fold the loser's provenance into it, so a paper
        # indexed twice does not lose the queries that found the copy we discarded.
        kept[existing] = _better(kept[existing], candidate)

    return kept, dropped


def _better(left: Candidate, right: Candidate) -> Candidate:
    """Merge two records of the same study, keeping the stronger one's identity.

    Stronger means: found by more queries, then ranked higher, then carrying a PMC id —
    in that order, because provenance is evidence about relevance while a PMC id is
    only evidence about what can be read later.
    """
    def strength(candidate: Candidate) -> tuple[int, int, int]:
        return (len(candidate.found_by), -candidate.best_rank, int(candidate.may_have_full_text))

    winner, loser = (left, right) if strength(left) >= strength(right) else (right, left)
    return winner.merged_with(loser)
