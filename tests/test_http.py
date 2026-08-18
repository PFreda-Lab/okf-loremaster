"""Rate limiting, caching, retries, and credential hygiene."""

from __future__ import annotations

import json
import os
import ssl
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
    assert DiskCache.key("GET", URL, {"term": "a"}) != DiskCache.key("GET", URL, {"term": "b"})


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


def expire(path: Path, days: float) -> None:
    """Age one entry, both ways it is read: `stored_at` is what `get` checks, mtime is
    what `sweep` checks, and an entry aged in only one of them tests neither."""
    payload = json.loads(path.read_text())
    payload["stored_at"] = time.time() - 86400 * days
    path.write_text(json.dumps(payload))
    os.utime(path, (payload["stored_at"], payload["stored_at"]))


def test_sweeping_deletes_what_the_ttl_already_expired(tmp_path: Path) -> None:
    """The TTL expired reads and nothing expired the files. An entry past it was ignored
    on lookup and then stayed forever, because the only thing that overwrites one is the
    same URL being asked for again — so a directory that looked like a month of cache was
    every response the tool had ever seen."""
    cache = DiskCache(tmp_path, ttl_days=1)
    for term in ("old", "new"):
        cache.put(DiskCache.key("GET", URL, {"term": term}), status=200, text=term, content_type="")
    stale = next(p for p in tmp_path.rglob("*.json") if json.loads(p.read_text())["text"] == "old")
    expire(stale, 2)

    files, freed = cache.sweep()

    assert files == 1
    assert freed > 0
    assert not stale.exists()
    fresh = DiskCache.key("GET", URL, {"term": "new"})
    assert cache.get(fresh) is not None, "sweeping took an entry still inside the TTL"


def test_sweeping_a_disabled_cache_deletes_nothing(tmp_path: Path) -> None:
    """Disabling the cache is how a run is made to re-fetch everything, not how its disk
    is reclaimed — the entries are somebody else's to keep."""
    enabled = DiskCache(tmp_path, ttl_days=1)
    enabled.put(DiskCache.key("GET", URL, {"term": "x"}), status=200, text="x", content_type="")
    expire(next(tmp_path.rglob("*.json")), 2)

    assert DiskCache(tmp_path, ttl_days=1, enabled=False).sweep() == (0, 0)
    assert list(tmp_path.rglob("*.json"))


def test_a_cache_with_no_ttl_sweeps_nothing(tmp_path: Path) -> None:
    """`ttl_days=0` is "these never go stale", and nothing expired cannot be swept."""
    cache = DiskCache(tmp_path, ttl_days=0)
    cache.put(DiskCache.key("GET", URL, {"term": "x"}), status=200, text="x", content_type="")
    expire(next(tmp_path.rglob("*.json")), 3650)

    assert cache.sweep() == (0, 0)
    assert list(tmp_path.rglob("*.json"))


def test_a_budget_bounds_a_cache_the_ttl_cannot(tmp_path: Path) -> None:
    """A TTL is not a ceiling. Every entry here is a day old against a month-long
    lifetime, so the sweep has nothing to expire — and enough traffic inside one lifetime
    is still unbounded growth. The two answer different questions: the TTL is about an
    answer being stale, the budget is about disk."""
    cache = DiskCache(tmp_path, ttl_days=30)
    for index in range(4):
        key = DiskCache.key("GET", URL, {"term": f"t{index}"})
        cache.put(key, status=200, text="x" * 500, content_type="")
        # Aged by hand: four files written in a loop share an mtime on a filesystem with
        # one-second granularity, and "oldest first" would then pass on any order at all.
        expire(tmp_path / key[:2] / f"{key}.json", index + 1)

    assert cache.sweep() == (0, 0), "nothing is past a 30-day TTL yet"
    files, freed = cache.sweep(max_bytes=1200)
    assert files == 2 and freed > 0
    assert cache.get(DiskCache.key("GET", URL, {"term": "t0"})) is not None, "newest first"
    assert cache.get(DiskCache.key("GET", URL, {"term": "t3"})) is None, "oldest went"


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


async def test_a_certificate_failure_is_not_retried_and_names_the_cause(
    tmp_path: Path,
) -> None:
    """A certificate does not become trusted between attempts.

    Found on a corporate network that TLS-intercepts one NIH host and not another: the
    run reported `ConnectError` three times, backed off between them, and concluded the
    service was unavailable. The service was fine. Both halves of that are fixed here —
    one attempt, and a message that says trust failed rather than that NIH is down.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate"
        ) from ssl.SSLCertVerificationError("certificate verify failed")

    client = _client(handler, tmp_path)
    with pytest.raises(HttpError, match="CA_BUNDLE"):
        await client.get_text(URL, cacheable=False)
    assert calls["n"] == 1, "a retry cannot make a certificate trusted"
    assert client.stats.retries == 0


async def test_a_certificate_failure_quotes_openssl_rather_than_guessing_the_cause(
    tmp_path: Path,
) -> None:
    """Two causes look identical from here, and the message used to assert one of them.

    It said the failure was "almost always a proxy on your own network" and pointed at
    `OKF_LOREMASTER_CA_BUNDLE`. It cannot know that: `self-signed certificate` is what a
    corporate root looks like and equally what a service answering on a misconfigured
    host looks like, and one NIH host fails exactly that way while the others in the same
    run succeed. A reader followed the message and went looking for a CA file that could
    not have helped.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("certificate verify failed") from (
            ssl.SSLCertVerificationError("self-signed certificate")
        )

    client = _client(handler, tmp_path)
    with pytest.raises(HttpError) as excinfo:
        await client.get_text(URL, cacheable=False)

    message = str(excinfo.value)
    assert "self-signed certificate" in message, "OpenSSL's own words reach the reader"
    assert "OKF_LOREMASTER_CA_BUNDLE" in message, "still named, for the case where it is the fix"
    # Both causes, and the one thing that separates them: whether the run's other hosts
    # failed too. Asserting either one is what this test exists to prevent.
    assert "other hosts" in message and "this host failed alone" in message


async def test_an_ordinary_connect_error_is_still_retried(tmp_path: Path) -> None:
    """The narrow rule stays narrow: only a TLS trust failure is treated as permanent."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, text="ok")

    client = _client(handler, tmp_path)
    assert await client.get_text(URL, cacheable=False) == "ok"
    assert calls["n"] == 3


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
