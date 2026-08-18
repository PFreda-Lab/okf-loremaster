"""Shared HTTP plumbing: rate limiting, disk cache, retries.

Every outbound request in the package goes through `HttpClient`. That is what makes
the NCBI rate limit enforceable — the limit is per IP address across *all* of
E-utilities, BioC and PubTator, so a limiter attached to any single client would be
quietly wrong the moment two of them run concurrently.

Nothing here knows what a paper is. Parsing lives in the per-API modules.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import ssl
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from okf_loremaster.events import EventBus, WarningEvent
from okf_loremaster.retention import trim_to_budget

# NCBI publishes 3 requests/second without an API key and 10 with one. We run under
# both ceilings on purpose: the limit is enforced per IP, and a shared institutional
# address may already be carrying traffic we cannot see. Being throttled costs a whole
# run; being 20% slower costs nothing that matters.
NCBI_RPS_WITH_KEY = 8.0
NCBI_RPS_WITHOUT_KEY = 2.5
NCBI_CEILING_WITH_KEY = 10.0
NCBI_CEILING_WITHOUT_KEY = 3.0

# iCite publishes no rate limit. This is politeness, not compliance.
ICITE_RPS = 5.0

# Query parameters that must never reach a cache key or a recorded fixture.
_SECRET_PARAMS = frozenset({"api_key", "apikey", "key"})
# Parameters that identify the caller but never change the response body. Excluded
# from cache keys so that changing a contact address does not invalidate the cache.
_IDENTITY_PARAMS = frozenset({"tool", "email"})

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class HttpError(RuntimeError):
    """A request failed in a way no retry will fix."""


@dataclass
class HttpStats:
    """Counters for the run manifest. Cheap to keep, tedious to reconstruct later."""

    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    bytes_downloaded: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.requests + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{self.requests} request(s), {self.cache_hits} cache hit(s) "
            f"({self.cache_hit_rate:.0%}), {self.retries} retry(ies), "
            f"{self.bytes_downloaded / 1024:.0f} KiB"
        )


class RateLimiter:
    """Token bucket. `acquire()` returns when a request may be sent.

    The clock and sleep function are injectable so that tests can assert pacing
    without spending real seconds on it.
    """

    def __init__(
        self,
        rate: float,
        *,
        capacity: float | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleeper | None = None,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate
        # A burst of one second's worth absorbs jitter without ever exceeding the
        # average, which is what the published limit actually constrains.
        self._capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self._capacity
        self._clock = clock
        self._sleep: Sleeper = sleep if sleep is not None else asyncio.sleep
        self._updated = clock()
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    async def acquire(self) -> None:
        # The lock is held across the sleep deliberately: it serializes waiters into
        # arrival order and makes the emitted rate the configured rate rather than a
        # thundering herd that all wake at once and fire together.
        async with self._lock:
            while True:
                now = self._clock()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await self._sleep((1.0 - self._tokens) / self._rate)


@dataclass(frozen=True, slots=True)
class CachedBody:
    status: int
    text: str
    content_type: str
    stored_at: float


class DiskCache:
    """Content-addressed response cache.

    Keys are derived from the redacted request, so a credential can never appear in a
    filename, and rotating one does not orphan the cache.
    """

    def __init__(self, root: Path, *, ttl_days: int, enabled: bool = True) -> None:
        self._root = root
        self._ttl = ttl_days * 86400.0
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def key(method: str, url: str, params: Mapping[str, Any] | None) -> str:
        canonical = f"{method.upper()} {url}?{_canonical_query(params)}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Sharded: a corpus build produces tens of thousands of entries, and some
        # filesystems degrade badly on a single flat directory that size.
        return self._root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedBody | None:
        if not self._enabled:
            return None
        path = self._path(key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        stored_at = float(raw.get("stored_at", 0.0))
        if self._ttl > 0 and time.time() - stored_at > self._ttl:
            return None
        return CachedBody(
            status=int(raw["status"]),
            text=str(raw["text"]),
            content_type=str(raw.get("content_type", "")),
            stored_at=stored_at,
        )

    def put(self, key: str, *, status: int, text: str, content_type: str) -> None:
        if not self._enabled:
            return
        path = self._path(key)
        payload = {
            "status": status,
            "text": text,
            "content_type": content_type,
            "stored_at": time.time(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: an interrupted run must not leave a truncated entry
            # that later reads as a valid short response.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # A cache that cannot be written is a slow run, not a failed one.
            return

    def sweep(self, *, max_bytes: int = 0) -> tuple[int, int]:
        """Delete entries past the TTL, then whatever still does not fit `max_bytes`.

        The TTL expired reads and nothing ever expired the files. An entry past it is
        ignored on lookup and then stays on disk indefinitely, because the only thing
        that overwrites one is the same URL being requested again — so a directory that
        looks like a month of cache is really every response this tool has ever seen.
        This is what makes the configured lifetime mean what it says.

        The TTL alone is not a ceiling, though: enough traffic inside one lifetime is
        still unbounded. `max_bytes` is the ceiling, and the two answer different
        questions — the TTL is about an answer being stale, the budget is about disk.

        Judged by modification time rather than by the `stored_at` inside each entry, so
        a sweep is one stat per file instead of a parse. They agree: an entry is written
        once, under a temporary name, and renamed into place without being touched again.

        Best effort throughout, on the same grounds as `put` — a cache that cannot be
        tidied is disk, not a failed run.
        """
        if not self._enabled:
            return (0, 0)

        files = 0
        freed = 0
        if self._ttl > 0:
            cutoff = time.time() - self._ttl
            for path in self._root.glob("*/*.json"):
                try:
                    stat = path.stat()
                    if stat.st_mtime > cutoff:
                        continue
                    path.unlink()
                except OSError:
                    continue
                files += 1
                freed += stat.st_size

        trimmed, reclaimed = trim_to_budget(self._root, max_bytes=max_bytes)
        return (files + trimmed, freed + reclaimed)


@dataclass
class HttpClient:
    """Rate-limited, cached, retrying HTTP. One instance per rate-limit domain."""

    limiter: RateLimiter
    cache: DiskCache
    transport: httpx.AsyncBaseTransport | None = None
    bus: EventBus | None = None
    timeout: float = 30.0
    max_retries: int = 4
    user_agent: str = "okf-loremaster"
    ca_bundle: Path | None = None
    stats: HttpStats = field(default_factory=HttpStats)
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
                verify=str(self.ca_bundle) if self.ca_bundle is not None else True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        node: str = "http",
        cacheable: bool = True,
    ) -> str:
        """Fetch a URL as text, through the cache and the rate limiter."""
        key = DiskCache.key("GET", url, params)
        if cacheable:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.cache_hits += 1
                return hit.text

        response = await self._request_with_retries(url, params=params, node=node)
        text = response.text
        self.stats.requests += 1
        self.stats.bytes_downloaded += len(response.content)
        if cacheable and response.status_code == 200:
            self.cache.put(
                key,
                status=response.status_code,
                text=text,
                content_type=response.headers.get("content-type", ""),
            )
        return text

    async def _request_with_retries(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        node: str,
    ) -> httpx.Response:
        attempts = max(1, self.max_retries)
        client = self._ensure()
        for attempt in range(1, attempts + 1):
            await self.limiter.acquire()
            reason: str
            retry_after: float | None = None
            try:
                response = await client.get(url, params=dict(params or {}))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                trust_failure = _certificate_failure(exc)
                if trust_failure is not None:
                    raise HttpError(
                        f"{_redact(url)}: {_certificate_advice(trust_failure)}"
                    ) from exc
                if attempt == attempts:
                    raise HttpError(f"{_redact(url)}: {type(exc).__name__}") from exc
                reason = type(exc).__name__
            else:
                if response.status_code not in _RETRY_STATUS:
                    if response.status_code >= 400:
                        raise HttpError(
                            f"{_redact(url)}: HTTP {response.status_code} {response.text[:200]!r}"
                        )
                    return response
                if attempt == attempts:
                    raise HttpError(
                        f"{_redact(url)}: HTTP {response.status_code} after {attempts} attempt(s)"
                    )
                reason = f"HTTP {response.status_code}"
                retry_after = _retry_after(response)

            delay = retry_after if retry_after is not None else _backoff(attempt)
            self.stats.retries += 1
            if self.bus is not None:
                self.bus.emit(
                    WarningEvent(
                        node=node,
                        message=(
                            f"{reason} from {_host(url)}, "
                            f"retry {attempt}/{attempts - 1} in {delay:.1f}s"
                        ),
                    )
                )
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")


def _certificate_advice(exc: ssl.SSLError) -> str:
    """What a TLS trust failure means, without asserting which cause it had.

    This message used to say the failure was "almost always a proxy on your own network"
    and to point at `OKF_LOREMASTER_CA_BUNDLE`. It cannot know that. `icite.od.nih.gov`
    fails here with OpenSSL reporting `self-signed certificate`, which is what a proxy's
    own root looks like *and* what a service answering on a misconfigured host looks
    like. The two are indistinguishable from inside one request, and asserting the first
    sent a reader off to obtain a corporate CA file that could not have helped.

    So the reason is quoted rather than interpreted, and the discriminator named instead:
    whether the other hosts in this run also failed. They share a network and they do not
    share an operator, so one host failing alone is that host's certificate and all of
    them failing together is the network's. A reader has that in front of them already —
    the rest of the warnings — which is why this does not go looking for it.
    """
    reason = str(getattr(exc, "verify_message", "") or exc).strip() or "no reason given"
    return (
        f"TLS certificate verification failed: {reason}. Retrying cannot help — a "
        "certificate does not become trusted between attempts. If the run's other hosts "
        "failed the same way, a proxy on your own network is terminating TLS: point "
        "OKF_LOREMASTER_CA_BUNDLE at your organization's CA file. If this host failed "
        "alone, the certificate is the service's own and nothing here configures it away."
    )


def _certificate_failure(exc: BaseException) -> ssl.SSLError | None:
    """The TLS trust failure under a transport error, or None if it is something else.

    The error itself rather than a boolean, because the advice quotes OpenSSL's reason
    and only this object carries it.

    Worth telling apart from every other `ConnectError` for two reasons, both learned
    from a run that reported `ConnectError from icite.od.nih.gov` three times and then
    ranked without citation metrics. Retrying is pointless — a certificate does not
    become trusted between attempts — so this raises on the first one instead of
    sleeping through a backoff schedule that cannot help. And `ConnectError` on its own
    reads as "the service is down", which sends a reader to a status page that will say
    everything is fine; what actually failed is trust, and only this branch can say so.

    Read off the `__cause__` chain rather than the exception type: httpx reports this as
    a plain `ConnectError`, and only the `ssl.SSLError` underneath it says what happened.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _backoff(attempt: int) -> float:
    """Exponential with full jitter, matching the LLM router's policy."""
    return min(30.0, 2.0**attempt) * random.random()


def _retry_after(response: httpx.Response) -> float | None:
    """Honor `Retry-After` when the server sends one; it knows better than we do."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, min(120.0, float(raw)))
    except ValueError:
        # The HTTP-date form. Rare from these APIs; fall back to our own backoff.
        return None


def _canonical_query(params: Mapping[str, Any] | None) -> str:
    """Sorted query string with credentials and caller identity removed."""
    if not params:
        return ""
    keep = {
        k: v
        for k, v in params.items()
        if k.lower() not in _SECRET_PARAMS and k.lower() not in _IDENTITY_PARAMS
    }
    return urlencode(sorted((k, str(v)) for k, v in keep.items()))


def _redact(url: str) -> str:
    """A URL safe to put in an exception, a log line, or a fixture."""
    if "?" not in url:
        return url
    base, _, query = url.partition("?")
    parts = []
    for chunk in query.split("&"):
        name, _, _value = chunk.partition("=")
        parts.append(f"{name}=REDACTED" if name.lower() in _SECRET_PARAMS else chunk)
    return f"{base}?{'&'.join(parts)}"


def _host(url: str) -> str:
    try:
        return httpx.URL(url).host
    except (httpx.InvalidURL, ValueError):
        return url
