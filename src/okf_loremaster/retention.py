"""Ceilings on everything this tool leaves behind.

Three stores grow, for three different reasons, and none of them used to stop. Checkpoints
are the big one — the whole run state serialized once per node — and they are pure scratch.
The HTTP cache and the extraction cache are not scratch: they are what makes a rerun cheap,
and the extraction cache is what makes it *free*, since extract is the only node that pays
per paper. So the checkpoint store is trimmed hard and the other two only at a ceiling they
are not expected to reach.

Every budget here is applied when a build starts, not while it runs. A build writes on top
of whatever survived, so the true peak is a budget plus one run, and a cap of a gigabyte is
a promise about what you come back to rather than a limit that can never be crossed. Saying
otherwise would be a number that quietly does not hold.

Byte budgets are on content, and files are deleted oldest first. Age is the only ranking
available that is not a guess: what a cache holds is answers, and an answer nobody has
needed in a month is the one whose loss costs least.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from okf_loremaster.config import Settings

__all__ = [
    "MB",
    "directory_bytes",
    "extraction_cache_path",
    "http_cache_path",
    "trim_to_budget",
]

# Budgets are configured in megabytes because that is the unit a person thinks about
# disk in, and applied in bytes. One conversion, in one place.
MB = 1_048_576


def http_cache_path(settings: Settings) -> Path:
    """Where cached HTTP responses live."""
    return settings.cache_dir / "http"


def extraction_cache_path(settings: Settings) -> Path:
    """Where papers already read live."""
    return settings.cache_dir / "extractions"


def directory_bytes(root: Path) -> int:
    """Bytes of ordinary files under `root`. 0 if it is not there."""
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def trim_to_budget(root: Path, *, max_bytes: int) -> tuple[int, int]:
    """Delete the oldest files under `root` until what is left fits. Returns files, bytes.

    `max_bytes <= 0` means no ceiling, which is how a budget is turned off.

    Oldest first, by modification time. Entries are written once and renamed into place,
    never touched again, so mtime is when the answer was obtained — and one nobody has
    needed since is the one whose loss costs least. This is a cache: the worst case of
    deleting the wrong file is paying for it again.

    Best effort on every file. A cache that cannot be tidied is disk, not a failed run,
    and a single unreadable entry must not stop the rest from being reclaimed.
    """
    if max_bytes <= 0:
        return (0, 0)

    entries: list[tuple[float, int, Path]] = []
    total = 0
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
        total += stat.st_size

    if total <= max_bytes:
        return (0, 0)

    files = 0
    freed = 0
    for _, size, path in sorted(entries):
        if total <= max_bytes:
            break
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        files += 1
        freed += size
    return (files, freed)
