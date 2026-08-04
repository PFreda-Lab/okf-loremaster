"""Picking a run back up: finding its id, and paying for nothing twice.

`--resume` was mechanically correct long before it was usable. It takes a run id, the id
was printed once when the run stopped, and it named the output folder only when `-o` was
not given — so for anyone who passed `-o` and closed the terminal, a resumable run was
unreachable and its checkpoint was disk. The store knew the id the whole time; nothing
asked it. That is what `runs` and `find_run` are for, and what most of this file pins.

The rest is about what a resume costs. There are two mechanisms and they cover different
accidents, which is the part worth stating in a test: `extractions` is checkpointed, so a
run resumed after the extract node returned skips it whole; the extraction cache is
written per paper, so a run interrupted *inside* that node keeps what it already bought.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from okf_loremaster.run import RunOptions, build_run, find_run, list_runs, started_at

from graph_runs import PROMPT, full_run, run_settings


async def resume(settings_factory: Any, tmp_path: Path, run_id: str, **overrides: Any) -> Any:
    """Pick a run back up the way the CLI does: an id, and no question retyped."""
    from fake_ncbi import FakeNCBI

    options = RunOptions(resume=run_id, **{"out": tmp_path / "run", **overrides})
    state, _ = await build_run(
        options,
        console=Console(file=io.StringIO(), width=160, no_color=True),
        settings=run_settings(settings_factory, tmp_path),
        transport=FakeNCBI().transport(),
    )
    return state


async def resume_directory(
    settings_factory: Any, tmp_path: Path, run_id: str, **overrides: Any
) -> Path:
    """Where a resume decided to write, which is not the same question as where the run
    it resumed wrote — a finished run replays and its checkpointed `bundle` never moves,
    so asserting on that would pass whatever the directory logic did."""
    from fake_ncbi import FakeNCBI

    options = RunOptions(resume=run_id, **{"out": tmp_path / "run", **overrides})
    _, directory = await build_run(
        options,
        console=Console(file=io.StringIO(), width=160, no_color=True),
        settings=run_settings(settings_factory, tmp_path),
        transport=FakeNCBI().transport(),
    )
    return directory


# --- finding a run again ------------------------------------------------------


async def test_a_finished_run_is_listed_with_the_question_it_answered(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    listed = await list_runs(run_settings(settings_factory, tmp_path))

    assert [item.run_id for item in listed] == [run.state["run_id"]]
    assert listed[0].prompt == PROMPT
    assert listed[0].finished is True
    assert listed[0].reached == "emit_okf"


async def test_an_id_that_was_never_run_is_not_a_run(
    settings_factory: Any, tmp_path: Path
) -> None:
    assert await find_run(run_settings(settings_factory, tmp_path), "20200101-000000-dead") is None


async def test_resuming_an_id_that_was_never_run_says_so_rather_than_starting_one(
    settings_factory: Any, tmp_path: Path
) -> None:
    """A typo in an id used to begin a fresh run under that name — the checkpoint was
    empty, so the graph entered at `charter` and the user paid for a whole new run while
    believing they were finishing an old one."""
    with pytest.raises(ValueError, match="no run 20200101-000000-dead"):
        await resume(settings_factory, tmp_path, "20200101-000000-dead")


def test_a_run_id_carries_its_own_timestamp() -> None:
    """Read back out of the id rather than stored beside it, because `new_run_id` puts it
    there and a second copy is a second thing to disagree."""
    assert started_at("20260804-111902-b537") == datetime(2026, 8, 4, 11, 19, 2, tzinfo=UTC)


def test_an_id_with_no_date_in_it_is_still_a_run() -> None:
    """Typed by hand, or made by an older version. This feeds a listing, and a run with
    no date beside it is still a run somebody can resume."""
    assert started_at("my-run") is None


# --- what a resume costs ------------------------------------------------------


async def test_resuming_needs_only_the_id_and_re_reads_nothing(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point. The question is read back out of the checkpoint — not retyped,
    which would invite a version of it that differs from the one the run was built on —
    and every node that already finished stays finished."""
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    already = (len(run.scripted.screened), len(run.scripted.curated), len(run.scripted.extracted))
    assert all(already), "the run under test never reached the nodes this is about"

    state = await resume(settings_factory, tmp_path, run.state["run_id"])

    assert state["prompt"] == PROMPT
    assert (
        len(run.scripted.screened),
        len(run.scripted.curated),
        len(run.scripted.extracted),
    ) == already
    assert set(state["extractions"]) == set(run.state["extractions"])


async def test_a_resumed_run_writes_the_same_bundle_to_the_same_place(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed run must land where the run it resumed did. Deciding the directory
    inside the emit node rather than before the graph starts is how that breaks, and it
    breaks quietly: two folders, each holding half a bundle."""
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    state = await resume(settings_factory, tmp_path, run.state["run_id"])

    assert state["bundle"] == run.state["bundle"]
    assert state["run_id"] == run.state["run_id"]
    assert state["validated"] is True


async def test_a_resume_that_omits_the_name_lands_where_the_name_put_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-o` names the run, not the invocation.

    Resumed without it, the run used to fall back to the id, finish in a second folder
    called `20260804-111902-b537`, and leave the named one holding a half-built bundle
    that nobody was watching. The name is checkpointed with the run and read back, the
    same way the prompt is.
    """
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    landed = await resume_directory(settings_factory, tmp_path, run.state["run_id"], out=None)

    assert landed == tmp_path / "run"
    assert str(run.state["run_id"]) not in str(landed)


async def test_a_resume_that_names_somewhere_else_is_obeyed(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remembering the name must not make it impossible to change."""
    run = await full_run(settings_factory, tmp_path, monkeypatch)

    landed = await resume_directory(
        settings_factory, tmp_path, run.state["run_id"], out=tmp_path / "elsewhere"
    )

    assert landed == tmp_path / "elsewhere"


async def test_a_second_run_of_the_same_question_re_reads_no_papers(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not resume — a whole new run, new id, empty checkpoint. The extraction cache is
    keyed by the request rather than by the run, so the expensive node still costs
    nothing the second time. This is what covers a run killed *inside* extract, where the
    checkpoint holds nothing at all."""
    first = await full_run(settings_factory, tmp_path, monkeypatch)
    assert first.scripted.extracted

    second = await full_run(settings_factory, tmp_path, monkeypatch)

    assert second.scripted.extracted == [], "the second run paid to read the same papers"
    assert set(second.state["extractions"]) == set(first.state["extractions"])
    assert second.state["run_id"] != first.state["run_id"]
