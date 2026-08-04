"""Record/replay, and the guarantee that the suite cannot reach the network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from okf_loremaster.clients import build_clients
from okf_loremaster.clients.cassette import (
    CassetteMiss,
    CassetteMode,
    CassetteTransport,
    interaction_key,
)

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def _recorder(path: Path, body: str = "recorded") -> CassetteTransport:
    inner = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    return CassetteTransport(path, CassetteMode.RECORD, inner=inner)


async def test_record_then_replay(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"

    async with httpx.AsyncClient(transport=_recorder(path)) as client:
        assert (await client.get(URL, params={"term": "x"})).text == "recorded"

    replay = CassetteTransport(path, CassetteMode.REPLAY)
    async with httpx.AsyncClient(transport=replay) as client:
        assert (await client.get(URL, params={"term": "x"})).text == "recorded"


async def test_an_unrecorded_request_raises_in_replay(tmp_path: Path) -> None:
    """The whole point: a test that quietly starts calling live APIs is not a test."""
    path = tmp_path / "c.jsonl"
    async with httpx.AsyncClient(transport=_recorder(path)) as client:
        await client.get(URL, params={"term": "x"})

    replay = CassetteTransport(path, CassetteMode.REPLAY)
    async with httpx.AsyncClient(transport=replay) as client:
        with pytest.raises(CassetteMiss, match="No recorded interaction"):
            await client.get(URL, params={"term": "something else"})


def test_a_missing_cassette_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(CassetteMiss, match="record_fixtures"):
        CassetteTransport(tmp_path / "absent.jsonl", CassetteMode.REPLAY)


async def test_credentials_are_never_written_to_a_cassette(tmp_path: Path) -> None:
    """Fixtures are committed to the repo. Keys are not."""
    path = tmp_path / "c.jsonl"
    async with httpx.AsyncClient(transport=_recorder(path)) as client:
        await client.get(URL, params={"term": "x", "api_key": "SECRET"})

    assert "SECRET" not in path.read_text()


async def test_a_fixture_replays_with_or_without_a_key(tmp_path: Path) -> None:
    """Recorded with a key, replayed by a developer who has none."""
    path = tmp_path / "c.jsonl"
    async with httpx.AsyncClient(transport=_recorder(path)) as client:
        await client.get(URL, params={"term": "x", "api_key": "SECRET"})

    replay = CassetteTransport(path, CassetteMode.REPLAY)
    async with httpx.AsyncClient(transport=replay) as client:
        assert (await client.get(URL, params={"term": "x"})).text == "recorded"


def test_interaction_key_ignores_credentials_and_identity() -> None:
    with_secrets = interaction_key(
        "GET", httpx.URL(URL, params={"term": "x", "api_key": "s", "email": "a@b.c"})
    )
    without = interaction_key("GET", httpx.URL(URL, params={"term": "x"}))
    assert with_secrets == without


async def test_re_recording_wins_over_the_old_entry(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    async with httpx.AsyncClient(transport=_recorder(path, "first")) as client:
        await client.get(URL, params={"term": "x"})
    async with httpx.AsyncClient(transport=_recorder(path, "second")) as client:
        await client.get(URL, params={"term": "x"})

    replay = CassetteTransport(path, CassetteMode.REPLAY)
    async with httpx.AsyncClient(transport=replay) as client:
        assert (await client.get(URL, params={"term": "x"})).text == "second"


async def test_clients_in_replay_mode_cannot_reach_the_network(
    settings_factory: Any, tmp_path: Path
) -> None:
    """An empty cassette must make every client fail loudly rather than dial out."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    clients = build_clients(
        settings_factory(
            ncbi_email="test@example.org",
            http_cache_enabled=False,
            cache_dir=tmp_path / "cache",
        ),
        transport=CassetteTransport(path, CassetteMode.REPLAY),
    )
    try:
        with pytest.raises(CassetteMiss):
            await clients.eutils.esearch("anything")
        with pytest.raises(CassetteMiss):
            await clients.icite.metrics(["1"])
    finally:
        await clients.aclose()
