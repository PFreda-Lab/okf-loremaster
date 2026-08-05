"""Ceilings on the caches: what a budget deletes, and in what order.

Every store this tool writes used to grow forever, in three different ways. The
checkpoints were the visible one — three gigabytes in two days — and they are pinned in
`test_resume.py` beside the rest of the resume machinery. This file is the other two and
the arithmetic all three share: delete oldest first, stop as soon as it fits, and never
raise, because the worst thing a cache miss costs is doing the work again.
"""

from __future__ import annotations

import os
from pathlib import Path

from okf_loremaster.extraction_cache import ExtractionCache
from okf_loremaster.retention import directory_bytes, trim_to_budget


def entry(root: Path, name: str, *, size: int, age: float) -> Path:
    """One cache entry of a known size and a known age, sharded the way both caches do.

    Age is set explicitly rather than relying on the order files are written: on a
    filesystem with one-second mtime granularity, four files written in a loop are all
    the same age and a test of "oldest first" would pass on any order at all.
    """
    path = root / name[:2] / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size, encoding="utf-8")
    stamp = 1_700_000_000.0 - age * 86400
    os.utime(path, (stamp, stamp))
    return path


def names(root: Path) -> set[str]:
    return {path.stem for path in root.rglob("*.json")}


# --- the budget ---------------------------------------------------------------


def test_a_budget_deletes_the_oldest_first(tmp_path: Path) -> None:
    """Age is the only ranking available that is not a guess. A cache holds answers, and
    an answer nobody has needed in a month is the one whose loss costs least."""
    for index, age in enumerate([40.0, 1.0, 10.0, 25.0]):
        entry(tmp_path, f"e{index}0000", size=1000, age=age)

    files, freed = trim_to_budget(tmp_path, max_bytes=2500)

    assert (files, freed) == (2, 2000)
    assert names(tmp_path) == {"e10000", "e20000"}, "the two most recently written"


def test_a_budget_stops_as_soon_as_it_fits(tmp_path: Path) -> None:
    """Not a purge down to empty, and not a high-water mark that clears half. Deleting
    one more file than necessary is one more paper somebody pays to read again."""
    for index in range(5):
        entry(tmp_path, f"e{index}0000", size=100, age=float(index))

    assert trim_to_budget(tmp_path, max_bytes=350) == (2, 200)
    assert directory_bytes(tmp_path) == 300


def test_a_cache_already_under_its_budget_is_left_alone(tmp_path: Path) -> None:
    entry(tmp_path, "aa0000", size=100, age=900.0)

    assert trim_to_budget(tmp_path, max_bytes=1000) == (0, 0)
    assert names(tmp_path) == {"aa0000"}


def test_a_budget_of_zero_is_no_budget(tmp_path: Path) -> None:
    """How a ceiling is turned off. Zero has to mean unlimited rather than "delete
    everything", because the two readings differ by the entire cache."""
    entry(tmp_path, "aa0000", size=100, age=900.0)

    assert trim_to_budget(tmp_path, max_bytes=0) == (0, 0)
    assert names(tmp_path) == {"aa0000"}


def test_a_cache_that_was_never_written_is_not_an_error(tmp_path: Path) -> None:
    """The first build on a new machine sweeps before it caches anything."""
    missing = tmp_path / "not-there"

    assert trim_to_budget(missing, max_bytes=100) == (0, 0)
    assert directory_bytes(missing) == 0


def test_a_budget_counts_every_shard(tmp_path: Path) -> None:
    """Both caches fan out one level, so a sweep that only looked at the top would
    measure a cache of tens of thousands of files as empty."""
    entry(tmp_path, "aa0000", size=1000, age=9.0)
    entry(tmp_path, "zz0000", size=1000, age=1.0)

    assert directory_bytes(tmp_path) == 2000
    assert trim_to_budget(tmp_path, max_bytes=1500) == (1, 1000)
    assert names(tmp_path) == {"zz0000"}


# --- readings -----------------------------------------------------------------


def test_the_extraction_cache_is_capped(tmp_path: Path) -> None:
    """The one store where a deletion costs real money, so it is the one with the
    highest ceiling — but a ceiling all the same, since what accumulates here is entries
    nothing can reach: every prompt revision leaves its predecessors behind under a
    fingerprint that will never be asked for again."""
    cache = ExtractionCache(tmp_path)
    for index, age in enumerate([100.0, 2.0, 50.0]):
        entry(tmp_path, f"1234{index}-abcdef0123456789", size=1000, age=age)

    files, freed = cache.sweep(max_bytes=1500)

    assert (files, freed) == (2, 2000)
    assert names(tmp_path) == {"12341-abcdef0123456789"}


def test_the_extraction_cache_has_no_ceiling_by_default(tmp_path: Path) -> None:
    """`sweep()` with nothing said is a no-op, so a caller that forgets to pass a budget
    cannot silently start deleting readings somebody paid for."""
    cache = ExtractionCache(tmp_path)
    entry(tmp_path, "12345-abcdef0123456789", size=1000, age=999.0)

    assert cache.sweep() == (0, 0)
    assert names(tmp_path) == {"12345-abcdef0123456789"}


def test_a_reading_that_survives_the_sweep_is_still_a_hit(tmp_path: Path) -> None:
    """A trimmed cache has to be a working cache: the point of a budget is that the
    entries under it keep answering."""
    from okf_loremaster.extraction_cache import fingerprint
    from okf_loremaster.schemas import Extraction

    cache = ExtractionCache(tmp_path)
    key = fingerprint("prompt", "text")
    cache.put("12345", key, Extraction(bottom_line="read once, kept"))
    entry(tmp_path, "99999-0000000000000000", size=5000, age=400.0)

    cache.sweep(max_bytes=1000)

    kept = cache.get("12345", key)
    assert kept is not None and kept.bottom_line == "read once, kept"
