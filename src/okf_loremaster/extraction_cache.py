"""Papers already read, kept on disk between runs.

Extract is the only node that makes a call per paper, so it is where a run's money goes,
and until this existed it was also the only expensive thing a run could be made to do
twice. `extractions` is checkpointed keyed by PMID, and the node skips a PMID already in
it — but LangGraph checkpoints a node's output when the node *returns*, and this node
returns once, after two hundred papers. Interrupt it at paper a hundred and ninety and
the checkpoint still says zero. `--resume` then re-reads, and re-buys, all of them.

So a reading is written down as it happens, outside the checkpoint. The key is a
fingerprint of the exact bytes that were sent to the model, which makes a hit mean "this
request was already answered" rather than "some run once looked at this PMID". Edit the
extraction prompt, retrieve a longer full text, or ask a different question, and the
fingerprint moves and the paper is read again — which is the behavior you want from a
cache you are allowed to forget about.

Failures are never cached. A paper that timed out or came back unparsable is a paper
nobody has read, and remembering that would make one bad afternoon permanent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from okf_loremaster.retention import trim_to_budget
from okf_loremaster.schemas import Extraction

__all__ = ["ExtractionCache", "fingerprint"]

# Enough of the digest to name a file with. A collision needs two different requests
# whose hex prefixes agree in sixteen places, and the cost of one is a single wrong
# extraction in a single bundle rather than anything structural.
KEY_CHARS = 16


def fingerprint(*parts: str) -> str:
    """A stable key for the exact request that produced an extraction."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        # A separator the inputs cannot themselves contain, so that ("ab", "c") and
        # ("a", "bc") are different requests here as well as everywhere else.
        digest.update(b"\x00")
    return digest.hexdigest()[:KEY_CHARS]


class ExtractionCache:
    """Readings kept between runs, one JSON file each.

    Every failure mode is a miss. A cache that cannot be read from, written to, or
    parsed is a cache that does not save money, and none of those is a reason to stop a
    run that can still do the work itself.
    """

    __slots__ = ("root",)

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, pmid: str, key: str) -> Extraction | None:
        path = self._path(pmid, key)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return Extraction.model_validate_json(raw)
        except ValueError:
            # A file written before a schema change, or one from a version that spelled
            # a field differently. Deleted rather than repaired: the model can produce
            # another reading and this code cannot.
            path.unlink(missing_ok=True)
            return None

    def put(self, pmid: str, key: str, extraction: Extraction) -> None:
        path = self._path(pmid, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written beside and renamed, because a few dozen papers are in flight at
            # once and a process killed mid-write would otherwise leave a truncated file
            # that looks like a hit and parses as nothing.
            temporary = path.with_suffix(".part")
            temporary.write_text(extraction.model_dump_json(), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            return

    def sweep(self, *, max_bytes: int = 0) -> tuple[int, int]:
        """Delete the oldest readings until the cache fits `max_bytes`. Files, bytes.

        No TTL here, unlike the HTTP cache, because an entry does not go stale: the key
        is the request, so a reading is either still the answer to a question somebody is
        asking or it is unreachable already. What accumulates is the unreachable half —
        every prompt revision, every full text that got longer, left behind under a
        fingerprint nothing will ever ask for again. A budget is the only thing that
        clears those, since they are indistinguishable from live entries by content.

        Which makes deleting by age exactly right here: the entries nobody can reach are
        the old ones, and a live entry that goes with them costs one paper re-read.
        """
        return trim_to_budget(self.root, max_bytes=max_bytes)

    def _path(self, pmid: str, key: str) -> Path:
        # Fanned out one level, because a flat directory of a hundred thousand files is
        # slow to list and miserable to look through. The PMID leads the filename so a
        # person can still find a paper by eye.
        return self.root / key[:2] / f"{pmid}-{key}.json"
