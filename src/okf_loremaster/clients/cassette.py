"""Record/replay HTTP fixtures, so the test suite never touches the network.

A cassette is a JSONL file of interactions, one per line. It plugs in as an httpx
transport, which puts it *below* the cache and the rate limiter — so replayed tests
exercise the real client code path rather than a stub of it.

In replay mode an unrecorded request raises `CassetteMiss` rather than falling through
to the network. That is the whole point: a test that silently starts making live calls
is a test that passes for the wrong reason, and it will fail on an airplane.

Credentials are redacted on write. Fixtures are committed; API keys are not.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

from okf_loremaster.clients._http import _canonical_query, _redact


class CassetteMode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"


class CassetteMiss(RuntimeError):
    """A request was made in replay mode that the cassette does not contain."""


def interaction_key(method: str, url: httpx.URL) -> str:
    """Identity of a request, ignoring credentials and caller identity.

    Deliberately not the raw URL: a fixture recorded with an API key must replay for
    a developer who has none, and vice versa.
    """
    params = dict(url.params)
    return f"{method.upper()} {url.scheme}://{url.host}{url.path}?{_canonical_query(params)}"


class CassetteTransport(httpx.AsyncBaseTransport):
    """httpx transport that records to, or replays from, a JSONL cassette."""

    def __init__(
        self,
        path: Path,
        mode: CassetteMode,
        *,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._path = path
        self._mode = mode
        self._inner = inner
        self._index: dict[str, dict[str, Any]] = {}
        if mode is CassetteMode.REPLAY:
            self._index = _load(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def interactions(self) -> int:
        return len(self._index)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = interaction_key(request.method, request.url)

        if self._mode is CassetteMode.REPLAY:
            entry = self._index.get(key)
            if entry is None:
                raise CassetteMiss(
                    f"No recorded interaction for:\n  {key}\n"
                    f"Cassette: {self._path} ({len(self._index)} interaction(s))\n"
                    "Re-record with `python scripts/record_fixtures.py`."
                )
            return httpx.Response(
                status_code=int(entry["status"]),
                headers={"content-type": str(entry.get("content_type", ""))},
                text=str(entry["text"]),
                request=request,
            )

        inner = self._inner or httpx.AsyncHTTPTransport()
        response = await inner.handle_async_request(request)
        await response.aread()
        self._append(
            key,
            {
                "key": key,
                "url": _redact(str(request.url)),
                "status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "text": response.text,
            },
        )
        return response

    def _append(self, key: str, entry: dict[str, Any]) -> None:
        # Last write wins in the index; re-recording appends rather than rewrites, and
        # `_load` keeps the final occurrence.
        self._index[key] = entry
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise CassetteMiss(
            f"Cassette not found: {path}\nRecord it with `python scripts/record_fixtures.py`."
        )
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                index[str(entry["key"])] = entry
    return index


@contextmanager
def cassette(path: Path, mode: CassetteMode) -> Iterator[CassetteTransport]:
    transport = CassetteTransport(path, mode)
    try:
        yield transport
    finally:
        pass
