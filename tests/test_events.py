"""The event bus: fan-out, close semantics, and emitting with nobody listening."""

from __future__ import annotations

import asyncio

from okf_loremaster.events import EventBus, NodeStarted, Progress, RunStarted


async def _drain(queue: asyncio.Queue[object]) -> list[object]:
    received: list[object] = []
    while True:
        item = await queue.get()
        if item is None:
            return received
        received.append(item)


async def test_fan_out_to_every_subscriber() -> None:
    bus = EventBus()
    first, second = bus.subscribe(), bus.subscribe()

    bus.emit(RunStarted(run_id="r1", prompt="p"))
    bus.emit(NodeStarted(node="charter"))
    bus.close()

    assert len(await _drain(first)) == 2
    assert len(await _drain(second)) == 2


async def test_emit_without_subscribers_is_harmless() -> None:
    """Nodes emit unconditionally; nothing may depend on a renderer being attached."""
    bus = EventBus()
    bus.emit(Progress(node="screen", message="x"))
    bus.close()
    assert bus.closed


async def test_close_is_idempotent() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    bus.close()
    bus.close()
    assert await queue.get() is None
    assert queue.empty()


async def test_late_subscriber_gets_the_sentinel() -> None:
    """Otherwise a renderer started after a fast run would await a None that never comes."""
    bus = EventBus()
    bus.close()
    assert await bus.subscribe().get() is None


async def test_subscriber_misses_only_events_before_it_subscribed() -> None:
    bus = EventBus()
    bus.emit(NodeStarted(node="early"))
    late = bus.subscribe()
    bus.emit(NodeStarted(node="late"))
    bus.close()

    received = await _drain(late)
    assert [event.node for event in received] == ["late"]  # type: ignore[attr-defined]
