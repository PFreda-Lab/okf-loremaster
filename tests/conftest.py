"""Shared fixtures.

Every test runs against a known-empty environment. Without this a developer's own
`.env` leaks into assertions and the suite passes or fails depending on whose machine
it runs on.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from okf_loremaster.clients import Clients, build_clients
from okf_loremaster.clients.cassette import CassetteMode, CassetteTransport
from okf_loremaster.config import ENV_PREFIX, Settings

CASSETTE = Path(__file__).parent / "fixtures" / "ncbi.jsonl"

# LiteLLM fetches its model-price map over HTTP the first time it is imported, falling
# back to the copy bundled in the wheel. Set before any import of it, so the suite
# neither depends on that call nor pays for it. Production keeps the live fetch: a
# fresher price map means fewer calls we have to report as unpriced.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_ALSO_CLEARED = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "HF_HOME",
    "NO_COLOR",
    "CI",
    "TERM",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX) or key in _ALSO_CLEARED:
            monkeypatch.delenv(key, raising=False)

    # Clearing the variables is only half of it: `load_settings()` also reads `./.env`,
    # and the suite runs from the repository root where a working `.env` normally sits.
    # Two CLI tests passed for exactly that reason and failed on a machine without one —
    # the first `pytest` after a fresh clone. Patched on the module, so the name
    # `test_config.py` imported directly still refers to the real function and can go on
    # asserting the real search order.
    from okf_loremaster import config

    def no_env_files() -> tuple[Path, ...]:
        return ()

    monkeypatch.setattr(config, "env_file_candidates", no_env_files)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a live HTTP request from a test impossible, not merely unlikely.

    Replaying fixtures is only worth something if nothing can quietly bypass them. A
    test that reaches the real API passes for the wrong reason, is slow, depends on
    someone else's uptime, and fails on a plane. Blocking the transport turns that
    from a convention into a guarantee — including for tests written later.

    Cassette replay never constructs a real transport, so nothing legitimate trips it.
    """

    def refuse(host: str) -> RuntimeError:
        return RuntimeError(
            f"A test tried to reach {host} over the network.\n"
            "Tests replay recorded fixtures. Add the interaction with "
            "`python scripts/record_fixtures.py --email you@example.org`."
        )

    async def blocked_async(self: object, request: httpx.Request) -> httpx.Response:
        raise refuse(request.url.host)

    # Separate sync blocker on purpose. An async function patched over a synchronous
    # method does not raise — it returns a coroutine nobody awaits, so the call
    # silently succeeds-ish and only surfaces as a RuntimeWarning.
    def blocked_sync(self: object, request: httpx.Request) -> httpx.Response:
        raise refuse(request.url.host)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked_sync)


@pytest.fixture
def llm_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete set of model tiers and a key, for tests about a later check than that one.

    `run_build` calls `require_llm()` before it looks at the prompt or the resume id, so a
    test asserting on either reads back a config error unless it supplies these. They used
    to arrive from whichever `.env` happened to be in the working directory, which is why
    two such tests passed here and failed on a machine that had never been configured.
    Names no real provider: nothing in these tests reaches a model.
    """
    for role in ("FAST", "BALANCED", "REASONING"):
        monkeypatch.setenv(f"{ENV_PREFIX}MODEL_{role}", f"fake/{role.lower()}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")


@pytest.fixture
def settings_factory() -> Any:
    """Build Settings that ignore any .env on disk."""

    def build(**overrides: Any) -> Settings:
        return Settings(_env_file=None, **overrides)

    return build


@pytest.fixture
async def replay_clients(settings_factory: Any, tmp_path: Path) -> AsyncIterator[Clients]:
    """All four clients wired to the recorded cassette.

    The disk cache is off: a cache hit would satisfy the request before it ever
    reached the cassette, so the very code path under test would be skipped.
    """
    settings = settings_factory(
        ncbi_email="test@example.org",
        http_cache_enabled=False,
        cache_dir=tmp_path / "cache",
    )
    transport = CassetteTransport(CASSETTE, CassetteMode.REPLAY)
    clients = build_clients(settings, transport=transport)
    try:
        yield clients
    finally:
        await clients.aclose()
