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
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from okf_loremaster.run import (
    RunOptions,
    build_run,
    find_run,
    list_runs,
    prune_checkpoints,
    started_at,
)

from graph_runs import PROMPT, charter_for, full_run, run_settings


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


async def test_an_id_that_was_never_run_is_not_a_run(settings_factory: Any, tmp_path: Path) -> None:
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


# --- what the store keeps -----------------------------------------------------


async def seed_runs(settings: Any, *run_ids: str, size: int = 0) -> None:
    """Put threads in the store without paying for runs to make them.

    Retention is SQL over two tables, and what it has to get right is which thread_ids
    it picks — not anything about what a checkpoint contains. Three real runs to prove
    that would be minutes of test time to assert on an `ORDER BY`.

    `size` pads the checkpoint blob, for the rule that is about bytes rather than count.
    A real run is 100 to 350 MB and the ceiling has to be tested against something; these
    are the same arithmetic three orders of magnitude down.
    """
    from okf_loremaster.graph.build import checkpointer

    async with checkpointer(settings) as saver:
        await saver.setup()
        for run_id in run_ids:
            await saver.conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                "type, checkpoint, metadata) VALUES (?, '', 'c1', 'x', ?, ?)",
                (run_id, b"{}" + b" " * size, b"{}"),
            )
            await saver.conn.execute(
                "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, "
                "idx, channel, type, value) VALUES (?, '', 'c1', 't', 0, 'ch', 'x', ?)",
                (run_id, b"{}"),
            )
        await saver.conn.commit()


async def threads(settings: Any) -> tuple[list[str], list[str]]:
    """Every thread id in each table, newest first. Both, because a run deleted from one
    and left in the other is a store that disagrees with itself."""
    from okf_loremaster.graph.build import checkpointer

    async with checkpointer(settings) as saver:
        await saver.setup()
        out = []
        for table in ("checkpoints", "writes"):
            cursor = await saver.conn.execute(
                f"SELECT DISTINCT thread_id FROM {table} ORDER BY thread_id DESC"
            )
            out.append([str(row[0]) for row in await cursor.fetchall()])
        return (out[0], out[1])


async def test_retention_keeps_the_newest_runs_and_drops_the_rest(
    settings_factory: Any, tmp_path: Path
) -> None:
    """Two days of ordinary use reached 3 GB before anything dropped a finished run: a
    build costs 100 to 350 MB of checkpoints, because the whole state is serialized once
    per node and by then it holds abstracts, full texts and extractions."""
    settings = run_settings(settings_factory, tmp_path)
    ids = [f"2026080{day}-120000-aaaa" for day in range(1, 6)]
    await seed_runs(settings, *ids)

    result = await prune_checkpoints(settings, keep=2)

    assert result.runs == 3
    kept, kept_writes = await threads(settings)
    assert kept == ids[-1:-3:-1], "kept by recency, not by insertion order"
    assert kept_writes == kept, "a thread left in `writes` is a store disagreeing with itself"


async def test_retention_is_by_count_and_not_by_age(settings_factory: Any, tmp_path: Path) -> None:
    """What makes a checkpoint worth keeping is being recent relative to the others.
    Somebody resuming picks from the last few runs, not the last few days, and a
    fortnight away from the tool should not mean coming back to an empty store."""
    settings = run_settings(settings_factory, tmp_path)
    await seed_runs(settings, "20200101-120000-aaaa", "20200102-120000-bbbb")

    assert (await prune_checkpoints(settings, keep=5)).runs == 0
    kept, _ = await threads(settings)
    assert len(kept) == 2


async def test_retention_can_be_turned_off(settings_factory: Any, tmp_path: Path) -> None:
    settings = run_settings(settings_factory, tmp_path)
    ids = [f"2026080{day}-120000-aaaa" for day in range(1, 6)]
    await seed_runs(settings, *ids)

    assert (await prune_checkpoints(settings, keep=0)).runs == 0
    kept, _ = await threads(settings)
    assert len(kept) == len(ids)


async def test_a_run_survives_retention_still_resumable(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting other threads must not damage the one kept. Whole runs only, and never
    the newest ones — a retention pass that left a resumable run half-deleted would fail
    at the point of resuming it, long after the pass that broke it."""
    settings = run_settings(settings_factory, tmp_path)
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    await seed_runs(settings, "20200101-120000-aaaa", "20200102-120000-bbbb")

    assert (await prune_checkpoints(settings, keep=1)).runs == 2

    found = await find_run(settings, run.state["run_id"])
    assert found is not None
    assert found.prompt == PROMPT


async def test_a_resumed_run_prunes_nothing(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retention keeps the newest runs and a resumed one is, by definition, not the
    newest — so a build that pruned on the way in could delete the very thread it was
    about to replay."""
    from fake_ncbi import FakeNCBI

    run = await full_run(settings_factory, tmp_path, monkeypatch)
    settings = run_settings(settings_factory, tmp_path).model_copy(
        update={"checkpoint_keep_runs": 1}
    )
    await seed_runs(settings, "20200101-120000-aaaa")

    await build_run(
        RunOptions(resume=run.state["run_id"], out=tmp_path / "run"),
        console=Console(file=io.StringIO(), width=160, no_color=True),
        settings=settings,
        transport=FakeNCBI().transport(),
    )

    kept, _ = await threads(settings)
    assert "20200101-120000-aaaa" in kept, "the resume pruned on its way in"


async def test_the_store_has_a_ceiling_in_bytes_as_well_as_in_runs(
    settings_factory: Any, tmp_path: Path
) -> None:
    """The count is the rule that normally binds; this is what catches runs several
    times the usual size, so that the ceiling is a number of megabytes rather than a
    number of runs times a guess about how big one is."""
    settings = run_settings(settings_factory, tmp_path)
    ids = [f"2026080{day}-120000-aaaa" for day in range(1, 6)]
    await seed_runs(settings, *ids, size=10_000)

    result = await prune_checkpoints(settings, keep=0, max_bytes=25_000)

    assert result.runs == 3
    kept, kept_writes = await threads(settings)
    assert kept == ids[-1:-3:-1], "dropped by size, newest kept"
    assert kept_writes == kept


async def test_the_newest_run_survives_the_ceiling(settings_factory: Any, tmp_path: Path) -> None:
    """A ceiling that deletes the run somebody is about to resume has stopped being a
    ceiling and become a bug. One run over budget is over budget."""
    settings = run_settings(settings_factory, tmp_path)
    await seed_runs(settings, "20260801-120000-aaaa", size=10_000)

    assert (await prune_checkpoints(settings, keep=0, max_bytes=100)).runs == 0
    kept, _ = await threads(settings)
    assert kept == ["20260801-120000-aaaa"]


async def test_a_finished_build_leaves_the_store_at_its_ceiling(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retention ran only on the way in, so a store rested at its ceiling *plus* the run
    just written — until somebody built again, and if they never did, forever. A cap you
    are over for the whole time you are not using the tool is not much of a cap.

    Run ids are pinned because they are the sort key: two builds inside the same second
    would be ordered by the random suffix, and this asserts which one survived.
    """
    ids = iter(["20260101-000001-aaaa", "20260101-000002-bbbb"])
    monkeypatch.setattr("okf_loremaster.run.new_run_id", lambda: next(ids))
    settings = run_settings(settings_factory, tmp_path).model_copy(
        update={"checkpoint_keep_runs": 1}
    )

    await full_run(settings_factory, tmp_path, monkeypatch, settings=settings)
    await full_run(settings_factory, tmp_path, monkeypatch, settings=settings)

    kept, kept_writes = await threads(settings)
    assert kept == ["20260101-000002-bbbb"], "the first run outlived the build that replaced it"
    assert kept_writes == kept


async def test_a_finished_run_whose_folder_was_deleted_is_dropped(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one honest link between a bundle and what it cost. A run records where it
    wrote and whether it validated, so a finished run whose folder has been deleted is
    hundreds of megabytes of checkpoints for output that no longer exists — and nothing
    will resume it, because there is nothing left to resume to."""
    settings = run_settings(settings_factory, tmp_path)
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    assert (await prune_checkpoints(settings, keep=0)).runs == 0, "its folder is right there"

    shutil.rmtree(tmp_path / "run")

    assert (await prune_checkpoints(settings, keep=0)).runs == 1
    assert await find_run(settings, run.state["run_id"]) is None


async def test_an_unfinished_run_is_not_judged_by_a_folder_it_may_never_have_written(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stopped at the charter may never have created a folder at all, and stopped
    runs are the entire set `--resume` exists for. Judging them the same way would delete
    every one of them."""
    from okf_loremaster.ui.pauses import PauseDecision

    from fake_ncbi import FakeNCBI

    class StopsAtCharter:
        async def charter(self, charter: Any) -> PauseDecision:
            return PauseDecision(proceed=False, reason="not this taxonomy")

        async def retrieve(self, state: Any, *, estimate: Any) -> PauseDecision:
            return PauseDecision(proceed=True)

    given = tmp_path / "given.yaml"
    given.write_text(charter_for().to_yaml(), encoding="utf-8")
    settings = run_settings(settings_factory, tmp_path)
    state, directory = await build_run(
        RunOptions(prompt=PROMPT, charter_path=given, out=tmp_path / "stopped"),
        console=Console(file=io.StringIO(), width=160, no_color=True),
        settings=settings,
        transport=FakeNCBI().transport(),
        pause=StopsAtCharter(),
    )
    assert not state.get("validated")
    shutil.rmtree(directory)

    assert (await prune_checkpoints(settings, keep=0)).runs == 0
    found = await find_run(settings, state["run_id"])
    assert found is not None and found.finished is False


# --- what survives being written down and read back ---------------------------


def reachable_from_state() -> set[type]:
    """Every schema a checkpoint can contain, walked from `RunState` rather than listed.

    Nesting is not shelter: msgpack tags each model with its own type, so a model only
    ever reached through one that is already allowlisted still needs to be there itself.
    That is how five of the seven missing ones hid.
    """
    import inspect
    import typing
    from enum import Enum

    from pydantic import BaseModel

    from okf_loremaster.graph.state import RunState

    seen: set[Any] = set()
    found: set[type] = set()

    def walk(annotation: Any) -> None:
        if annotation in seen:
            return
        seen.add(annotation)
        for arg in typing.get_args(annotation):
            walk(arg)
        if not inspect.isclass(annotation):
            return
        if not annotation.__module__.startswith("okf_loremaster"):
            return
        if issubclass(annotation, BaseModel | Enum):
            found.add(annotation)
        if issubclass(annotation, BaseModel):
            for field in annotation.model_fields.values():
                walk(field.annotation)

    for annotation in typing.get_type_hints(RunState).values():
        walk(annotation)
    return found


def test_every_type_that_can_travel_in_state_is_allowlisted() -> None:
    """The serializer allowlist is written by hand and had drifted seven types behind.

    Two of them showed up as `Blocked deserialization` the moment a real checkpoint was
    read back; the other five had not been reached yet and would have surfaced on some
    later resume instead. LangGraph's msgpack path warns today and has announced it will
    refuse, so a gap here is a resume that stops working at a date nobody chose.
    """
    from okf_loremaster.graph.build import CHECKPOINTED_TYPES

    missing = reachable_from_state() - set(CHECKPOINTED_TYPES)

    assert not missing, "reachable from RunState but not in CHECKPOINTED_TYPES: " + ", ".join(
        sorted(t.__name__ for t in missing)
    )


def test_the_allowlist_names_nothing_that_cannot_travel() -> None:
    """The other direction, and the reason the list stays scannable: a name that state
    can no longer reach is one more line to read past when checking whether the thing
    you are adding is already there."""
    from okf_loremaster.graph.build import CHECKPOINTED_TYPES

    stale = set(CHECKPOINTED_TYPES) - reachable_from_state()

    assert not stale, "in CHECKPOINTED_TYPES but unreachable from RunState: " + ", ".join(
        sorted(t.__name__ for t in stale)
    )
