"""Machine-readable renderer — one JSON object per line, on stdout.

The counterpart to `ui/plain.py` for `--json`. Everything human goes to stderr, so a
piped run yields a clean event stream on stdout and still shows its errors in the
terminal.

Events are dataclasses, so `asdict` is the whole serializer; only `datetime` needs
help. The `kind` key is the dataclass name, which is what makes the stream matchable
without a schema.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime
from typing import Any, TextIO

from okf_loremaster.events import Event, EventBus

__all__ = ["JsonlRenderer"]


def _encode(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class JsonlRenderer:
    """Writes every event as one line of JSON.

    Subscribes at construction, like `PlainRenderer`, so events emitted before the
    consumer task starts are not lost.
    """

    def __init__(self, bus: EventBus, *, stream: TextIO | None = None) -> None:
        self._queue = bus.subscribe()
        self._stream = stream if stream is not None else sys.stdout

    async def run(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            self._write(event)
        self._stream.flush()

    def _write(self, event: Event) -> None:
        payload: dict[str, Any] = {"kind": type(event).__name__}
        payload |= dataclasses.asdict(event)
        self._stream.write(json.dumps(payload, default=_encode) + "\n")
        # Flushed per line: the point of this renderer is that something else can read
        # it while the run is still going.
        self._stream.flush()
