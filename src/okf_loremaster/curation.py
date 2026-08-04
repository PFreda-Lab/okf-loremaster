"""Making the shelves fit, once the curator has judged the papers.

Curation is two things, kept strictly apart. The judgment — does this paper belong on
this shelf — is the model's, one call per shelf. Everything in this module is the
arithmetic that follows: the ceiling, the floor, the global target, and an honest record
of what could not be filled. That split is why the size of every shelf is reproducible
across runs even though the judgment behind it is not.

Order comes from the caller as a single total order over PMIDs, so that "the best paper
on this shelf" means the same thing to the trim, the backfill and the final listing.
The curate node builds it from screening relevance first, then position in the ranked
pool.

The three bounds conflict, and the order they are applied in is what decides which one
yields:

1. `shelf_max` is a hard trim. A shelf past its ceiling stops being browsable, which is
   the failure this whole package exists to avoid.
2. `shelf_min` is a backfill from the reserve — papers the screener saw and the curator
   did not keep. Cheaper and better informed than another search round, and it is tried
   first for exactly that reason.
3. `target_papers` is a trim of the largest shelves, and the only bound allowed to fail.
   A floor is a statement that a shelf below it is not worth having at all, so when the
   floors cannot fit inside the target, the target gives way — and says so.

Nothing here names a shelf, a condition, or a vocabulary. Every slug arrives from the
charter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from okf_loremaster.schemas import Charter, ShelfGap

__all__ = ["MAX_ROUNDS", "Placement", "enforce_bounds"]

# How many times a curation gap may send a run back to search. Two: the first round is
# the plan and the second is the correction. A third would be paying to re-screen a pool
# in order to ask for the same thing the second round already failed to find.
MAX_ROUNDS = 2


@dataclass(frozen=True, slots=True)
class Placement:
    """The final shelves, and what the bounds cost to reach them."""

    shelves: dict[str, list[str]]
    # Shelves still under their floor after the backfill. Carries the curator's own
    # account of what is missing, which is what the re-query round searches on.
    gaps: tuple[ShelfGap, ...] = ()
    warnings: tuple[str, ...] = ()
    trimmed: int = 0
    backfilled: int = 0
    duplicates: int = 0

    @property
    def total(self) -> int:
        return sum(len(pmids) for pmids in self.shelves.values())

    @property
    def filled(self) -> int:
        return sum(1 for pmids in self.shelves.values() if pmids)

    def summary(self) -> str:
        parts = [f"{self.total} papers across {self.filled} of {len(self.shelves)} shelves"]
        if self.backfilled:
            parts.append(f"{self.backfilled} backfilled")
        if self.trimmed:
            parts.append(f"{self.trimmed} trimmed")
        if self.gaps:
            parts.append(f"{len(self.gaps)} under floor")
        return ", ".join(parts)


def enforce_bounds(
    charter: Charter,
    *,
    kept: Mapping[str, Sequence[str]],
    rank: Mapping[str, int],
    reserve: Mapping[str, Sequence[str]] | None = None,
    missing: Mapping[str, str] | None = None,
) -> Placement:
    """Fit the curator's keeps inside the charter's bounds.

    `kept` is PMIDs per shelf slug, as decided. `rank` is the total order — lower is
    better — and every PMID that could be placed must appear in it; one that does not
    sorts last rather than raising, because losing a run over a missing sort key would
    be a poor trade for a paper we can simply place at the end.

    `reserve` is the fallback queue per shelf, **consumed in the order given**. The
    caller decides what "nearest miss" means; this only decides how many are taken.

    Every shelf in the charter appears in the result, including the empty ones. A shelf
    that ended up with nothing is a finding about the search, and dropping it from the
    mapping would hide it.
    """
    order = _orderer(rank)
    slugs = list(charter.slugs)
    reserve_map = reserve or {}
    missing_map = missing or {}
    warnings: list[str] = []

    placed: dict[str, list[str]] = {slug: [] for slug in slugs}
    # PMID to the shelf holding it. A paper appears on exactly one shelf: the bundle
    # has one file per paper, and a second copy under another folder would be a second
    # document claiming to be the same one.
    taken: dict[str, str] = {}

    duplicates = _place(slugs, kept, placed, taken)
    unknown = sorted(set(kept) - set(placed))
    if unknown:
        warnings.append(
            "curation named shelf(s) the charter does not have, and their papers were "
            "dropped: " + ", ".join(unknown)
        )
    if duplicates:
        warnings.append(
            f"{duplicates} paper(s) were kept on more than one shelf; each stayed on the "
            "first shelf in the charter's order"
        )

    trimmed = _apply_ceiling(charter, placed, taken, order)
    backfilled = _apply_floor(charter, slugs, placed, taken, reserve_map)

    # Computed before the target trim, and still accurate afterward: that trim never
    # takes a shelf below its floor, so it cannot create or deepen a gap.
    gaps = tuple(
        ShelfGap(
            shelf=slug,
            kept=len(placed[slug]),
            floor=charter.shelf_min,
            missing=missing_map.get(slug, ""),
        )
        for slug in slugs
        if len(placed[slug]) < charter.shelf_min
    )

    trimmed += _apply_target(charter, slugs, placed, taken, warnings)

    for pmids in placed.values():
        pmids.sort(key=order)

    return Placement(
        shelves=placed,
        gaps=gaps,
        warnings=tuple(warnings),
        trimmed=trimmed,
        backfilled=backfilled,
        duplicates=duplicates,
    )


# --- the four passes --------------------------------------------------------


def _place(
    slugs: list[str],
    kept: Mapping[str, Sequence[str]],
    placed: dict[str, list[str]],
    taken: dict[str, str],
) -> int:
    """Assign each kept paper to one shelf. Returns how many were claimed twice.

    Iterates the charter's slug order rather than the mapping's, so a paper two shelves
    both wanted lands on the same one every run.
    """
    duplicates = 0
    for slug in slugs:
        for pmid in kept.get(slug, ()):
            if pmid in taken:
                duplicates += 1
                continue
            taken[pmid] = slug
            placed[slug].append(pmid)
    return duplicates


def _apply_ceiling(
    charter: Charter,
    placed: dict[str, list[str]],
    taken: dict[str, str],
    order: Callable[[str], tuple[int, str]],
) -> int:
    """Trim every shelf to `shelf_max`, worst-ranked first."""
    trimmed = 0
    for pmids in placed.values():
        pmids.sort(key=order)
        if len(pmids) <= charter.shelf_max:
            continue
        for pmid in pmids[charter.shelf_max :]:
            del taken[pmid]
        trimmed += len(pmids) - charter.shelf_max
        del pmids[charter.shelf_max :]
    return trimmed


def _apply_floor(
    charter: Charter,
    slugs: list[str],
    placed: dict[str, list[str]],
    taken: dict[str, str],
    reserve: Mapping[str, Sequence[str]],
) -> int:
    """Fill shelves under `shelf_min` from their reserve, in the order given."""
    backfilled = 0
    for slug in slugs:
        pmids = placed[slug]
        for pmid in reserve.get(slug, ()):
            if len(pmids) >= charter.shelf_min:
                break
            if pmid in taken:
                continue
            taken[pmid] = slug
            pmids.append(pmid)
            backfilled += 1
    return backfilled


def _apply_target(
    charter: Charter,
    slugs: list[str],
    placed: dict[str, list[str]],
    taken: dict[str, str],
    warnings: list[str],
) -> int:
    """Trim the largest shelves down to `target_papers`, never below a floor.

    The bound that yields. `target_papers` is a browsability ceiling for the bundle as a
    whole; `shelf_min` is a judgment about whether a shelf earns a heading. When they
    disagree the shelves win, because a bundle slightly over its target still reads,
    while one with a two-paper shelf in it reads as broken.
    """
    trimmed = 0
    total = sum(len(pmids) for pmids in placed.values())
    while total > charter.target_papers:
        widest = max(
            (slug for slug in slugs if len(placed[slug]) > charter.shelf_min),
            key=lambda slug: (len(placed[slug]), slug),
            default="",
        )
        if not widest:
            break
        del taken[placed[widest].pop()]
        trimmed += 1
        total -= 1

    if total > charter.target_papers:
        warnings.append(
            f"kept {total} papers against a target of {charter.target_papers}: trimming "
            f"further would put a shelf below its floor of {charter.shelf_min}, and a "
            "shelf below its floor is not worth having"
        )
    return trimmed


def _orderer(rank: Mapping[str, int]) -> Callable[[str], tuple[int, str]]:
    """Sort key over PMIDs: the caller's order, then the PMID itself.

    The PMID breaks ties rather than list position, so that two shelves holding the same
    paper resolve it identically and a run repeated on the same corpus sorts the same
    way. A PMID absent from `rank` sorts after everything ranked.
    """
    fallback = len(rank) + 1

    def key(pmid: str) -> tuple[int, str]:
        return (rank.get(pmid, fallback), pmid)

    return key
