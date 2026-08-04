"""Typed events emitted by graph nodes and consumed by renderers.

Nodes never print. They emit events onto a bus and renderers subscribe, which is what
lets the Rich renderer, the Textual TUI, and the JSONL stream be interchangeable
without any node knowing which one is attached.

`LLMCall` carries both per-call and cumulative figures. The redundancy is deliberate:
every renderer can show a running meter without keeping its own accumulator, and a
JSONL line is self-describing to whatever reads it later.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeAlias


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    prompt: str
    dry_run: bool = False
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class RunFinished:
    run_id: str
    ok: bool
    summary: str = ""
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class NodeStarted:
    node: str
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class NodeFinished:
    node: str
    summary: str = ""
    seconds: float = 0.0
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class Progress:
    node: str
    message: str
    current: int | None = None
    total: int | None = None
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One completed model call.

    `usd` is None when the call could not be priced. That is not the same as free, and
    renderers must not display it as $0.00.
    """

    node: str
    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    usd: float | None
    seconds: float
    # Cumulative across the run, as of this call.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_usd: float = 0.0
    total_calls: int = 0
    unpriced_calls: int = 0
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class WarningEvent:
    node: str
    message: str
    at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    node: str
    message: str
    fatal: bool = False
    at: datetime = field(default_factory=_now)


Event: TypeAlias = (
    RunStarted
    | RunFinished
    | NodeStarted
    | NodeFinished
    | Progress
    | LLMCall
    | WarningEvent
    | ErrorEvent
)


class EventBus:
    """Fan-out of events to any number of subscribed renderers.

    `emit` is synchronous and never blocks, so a node can call it without awaiting and
    without caring whether anyone is listening. Queues are unbounded: a slow renderer
    costs memory, never correctness, and a run emits on the order of thousands of
    events.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[Event | None]] = []
        self._closed = False

    def subscribe(self) -> asyncio.Queue[Event | None]:
        """Return a queue receiving every subsequent event, then None at close."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._queues.append(queue)
        if self._closed:
            queue.put_nowait(None)
        return queue

    def emit(self, event: Event) -> None:
        for queue in self._queues:
            queue.put_nowait(event)

    def close(self) -> None:
        """Signal end of stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for queue in self._queues:
            queue.put_nowait(None)

    @property
    def closed(self) -> bool:
        return self._closed
