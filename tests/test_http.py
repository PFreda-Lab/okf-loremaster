"""Rate limiting, caching, retries, and credential hygiene."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from okf_loremaster.clients import ncbi_rate
from okf_loremaster.clients._http import (
    NCBI_CEILING_WITH_KEY,
    NCBI_CEILING_WITHOUT_KEY,
    DiskCache,
    HttpClient,
    HttpError,
    RateLimiter,
    _redact,
)
from okf_loremaster.events import EventBus, WarningEvent

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


class FakeClock:
    """A clock that only advances when something sleeps.

    Lets pacing be asserted exactly, in no wall-clock time at all.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


# --- rate limiting ---------------------------------------------------------


async def test_burst_is_capped_then_paced() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2.0, capacity=2.0, clock=clock, sleep=clock.sleep)

    for _ in range(2):
        await limiter.acquire()
    assert clock.now == 1000.0, "the bucket starts full, so the burst is free"

    for _ in range(4):
        await limiter.acquire()
    # Four more at 2/second.
    assert clock.now == pytest.approx(1002.0)


async def test_tokens_refill_over_time() -> None:
    clock = FakeClock()
    limiter = RateLimiter(4.0, capacity=4.0, clock=clock, sleep=clock.sleep)

    for _ in range(4):
        await limiter.acquire()
    clock.now += 10.0  # idle long enough to refill past capacity

    start = clock.now
    for _ in range(4):
        await limiter.acquire()
    assert clock.now == start, "a full bucket must not be charged for waiting"


async def test_rate_never_exceeds_the_published_ceiling(settings_factory: Any) -> None:
    """The one number that gets an IP blocked. Config must not be able to raise it."""
    assert ncbi_rate(settings_factory(ncbi_api_key="k")) <= NCBI_CEILING_WITH_KEY
    assert ncbi_rate(settings_factory()) <= NCBI_CEILING_WITHOUT_KEY
    assert ncbi_rate(settings_factory(ncbi_api_key="k")) > ncbi_rate(settings_factory())


def test_rate_limiter_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0.0)


# --- cache -----------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_days=30)
    key = DiskCache.key("GET", URL, {"term": "x"})
    cache.put(key, status=200, text="payload", content_type="application/json")

    hit = cache.get(key)
    assert hit is not None
    assert hit.text == "payload"


def test_credentials_are_excluded_from_the_cache_key() -> None:
    """Rotating a key must not orphan the cache, and must never name a file."""
    with_key = DiskCache.key("GET", URL, {"term": "x", "api_key": "secret"})
    without = DiskCache.key("GET", URL, {"term": "x"})
    assert with_key == without
    assert "secret" not in with_key


def test_caller_identity_is_excluded_from_the_cache_key() -> None:
    """`tool` and `email` identify us to NCBI but never change the response."""
    a = DiskCache.key("GET", URL, {"term": "x", "tool": "a", "email": "a@b.c"})
    b = DiskCache.key("GET", URL, {"term": "x", "tool": "z", "email": "z@y.x"})
    assert a == b


def test_different_queries_get_different_keys() -> None:
    assert DiskCache.key("GET", URL, {"term": "a"}) != DiskCache.key(
        "GET", URL, {"term": "b"}
    )


def test_parameter_order_does_not_matter() -> None:
    assert DiskCache.key("GET", URL, {"a": 1, "b": 2}) == DiskCache.key(
        "GET", URL, {"b": 2, "a": 1}
    )


def test_expired_entries_are_ignored(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_days=1)
    key = DiskCache.key("GET", URL, {"term": "x"})
    cache.put(key, status=200, text="stale", content_type="")

    path = next(tmp_path.rglob("*.json"))
    payload = json.loads(path.read_text())
    payload["stored_at"] = time.time() - 86400 * 2
    path.write_text(json.dumps(payload))

    assert cache.get(key) is None


def test_a_disabled_cache_never_answers(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_days=30, enabled=False)
    key = DiskCache.key("GET", URL, {"term": "x"})
    cache.put(key, status=200, text="payload", content_type="")
    assert cache.get(key) is None
    assert not list(tmp_path.rglob("*.json"))


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_days=30)
    key = DiskCache.key("GET", URL, {"term": "x"})
    cache.put(key, status=200, text="payload", content_type="")
    next(tmp_path.rglob("*.json")).write_text("{ truncated")
    assert cache.get(key) is None


# --- redaction -------------------------------------------------------------


def test_redaction_hides_the_key_and_keeps_the_rest() -> None:
    redacted = _redact(f"{URL}?db=pubmed&api_key=SECRET&term=x")
    assert "SECRET" not in redacted
    assert "db=pubmed" in redacted
    assert "term=x" in redacted


# --- retries ---------------------------------------------------------------


def _client(handler: Any, tmp_path: Path, bus: EventBus | None = None) -> HttpClient:
    return HttpClient(
        limiter=RateLimiter(1000.0),
        cache=DiskCache(tmp_path, ttl_days=30, enabled=False),
        transport=httpx.MockTransport(handler),
        bus=bus,
        max_retries=3,
    )


async def test_transient_status_is_retried_then_succeeds(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text="ok")

    bus = EventBus()
    queue = bus.subscribe()
    client = _client(handler, tmp_path, bus)

    assert await client.get_text(URL, cacheable=False) == "ok"
    assert calls["n"] == 3
    assert client.stats.retries == 2

    bus.close()
    warnings = []
    while (event := queue.get_nowait()) is not None:
        if isinstance(event, WarningEvent):
            warnings.append(event)
    assert len(warnings) == 2, "each retry must be visible"


async def test_client_error_is_not_retried(tmp_path: Path) -> None:
    """A 400 means the request is wrong; sending it again just wastes the budget."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = _client(handler, tmp_path)
    with pytest.raises(HttpError, match="400"):
        await client.get_text(URL, cacheable=False)
    assert calls["n"] == 1


async def test_retries_are_exhausted_then_raised(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    client = _client(handler, tmp_path)
    with pytest.raises(HttpError, match="503"):
        await client.get_text(URL, cacheable=False)


async def test_error_messages_never_leak_the_key(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad")

    client = _client(handler, tmp_path)
    with pytest.raises(HttpError) as excinfo:
        await client.get_text(URL, params={"api_key": "SECRET"}, cacheable=False)
    assert "SECRET" not in str(excinfo.value)


async def test_a_cache_hit_skips_the_network(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text="ok")

    client = HttpClient(
        limiter=RateLimiter(1000.0),
        cache=DiskCache(tmp_path, ttl_days=30),
        transport=httpx.MockTransport(handler),
    )
    assert await client.get_text(URL, params={"term": "x"}) == "ok"
    assert await client.get_text(URL, params={"term": "x"}) == "ok"
    assert calls["n"] == 1
    assert client.stats.cache_hits == 1
    assert client.stats.requests == 1
