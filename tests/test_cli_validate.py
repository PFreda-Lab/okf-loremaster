"""The two things step 7 adds to the command surface.

`validate <bundle>` is the same code the graph runs, reached without a run — which is
the only way to check a bundle somebody else built, or one built six months ago. Its
exit code is the whole point: a gate that reported failure and exited zero would be a
gate no script could act on.

The `--review` refusal is the other. `--dry-run` and `--json` each mean nobody is going
to look, and signing anyway would write `by: "human:<id>"` naming a person who never saw
the bundle. That is a false attestation rather than a weak one, so the combination is
refused rather than degraded. An autonomous run is not on that list: nobody steered the
search, but somebody still reads the bundle at the end, which is what the signature is
about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer.main
from typer.testing import CliRunner

from okf_loremaster.cli import app

runner = CliRunner()
@pytest.mark.parametrize("flag", ["--dry-run", "--json"])
def test_review_is_refused_with_any_flag_that_means_nobody_will_look(flag: str) -> None:
    result = runner.invoke(app, ["build", "a prompt", "--review", flag])

    assert result.exit_code == 1
    assert "--review cannot be combined with" in result.output
    assert flag in result.output


def test_review_is_allowed_on_an_autonomous_run() -> None:
    """The default run asks nobody to steer, which says nothing about who signs.

    Refusing this pair would mean a bundle can only be attested by someone who also sat
    through the search — an unrelated requirement, and the common case is the opposite.
    """
    result = runner.invoke(app, ["build", "a prompt", "--review", "--help"])

    assert result.exit_code == 0
    assert "--review cannot be combined with" not in result.output


# --- the question, and where it comes from ------------------------------------


def test_a_build_with_no_question_and_nothing_to_resume_says_which_to_give() -> None:
    """`prompt` became optional so that `--resume` could supply it from the checkpoint.
    That makes an empty `build` a thing typer will now accept, so the sentence it stops
    with has to name both ways of answering the question it is missing."""
    result = runner.invoke(app, ["build"])

    assert result.exit_code == 1
    assert "no prompt" in result.output
    assert "--resume" in result.output


def test_resuming_does_not_require_the_question_to_be_typed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retyping it invites a version that differs from the one the run was built on, and
    the graph does not read it on resume anyway. Pointed at an empty cache so this stops
    at "no such run" rather than at "no prompt" — which is the distinction under test."""
    monkeypatch.setenv("OKF_LOREMASTER_CACHE_DIR", str(tmp_path / "cache"))
    result = runner.invoke(app, ["build", "--resume", "20200101-000000-dead"])

    assert result.exit_code == 1
    assert "no run 20200101-000000-dead" in result.output
    assert "no prompt" not in result.output


def test_a_saved_charter_can_be_handed_back_to_a_fresh_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workflow the charter pause has always assumed and the CLI could not reach.

    Declining at the first pause tells the user to go and edit `charter.yaml`, and every
    run writes one — but `--charter` did not exist, so there was no way to hand the
    edited file back. It also makes a rerun comparable: a charter is drafted by a
    nondeterministic reasoning call, so the same prompt twice is two different runs and
    a fix cannot be checked against the charter that broke.
    """
    charter = tmp_path / "charter.yaml"
    charter.write_text(
        "prompt: a question\ntask: a question\npopulation: adults\noutcome: an outcome\n",
        encoding="utf-8",
    )
    seen: list[Any] = []

    async def stop(options: Any, **kwargs: Any) -> Any:
        seen.append(options)
        raise ValueError("reached the runner")

    monkeypatch.setattr("okf_loremaster.run.build_run", stop)
    result = runner.invoke(app, ["build", "--charter", str(charter)])

    # No prompt on the command line: it comes off the charter, which is the point.
    assert "no prompt" not in result.output
    assert seen and seen[0].charter_path == charter


def test_a_charter_is_refused_with_resume_rather_than_read_and_ignored(
    tmp_path: Path,
) -> None:
    """A resumed run replays the charter node from its checkpoint. Accepting a file here
    and then disregarding it is the failure mode worth refusing outright."""
    charter = tmp_path / "charter.yaml"
    charter.write_text("prompt: a question\ntask: a question\n", encoding="utf-8")

    result = runner.invoke(
        app, ["build", "--charter", str(charter), "--resume", "20200101-000000-dead"]
    )

    assert result.exit_code == 1
    assert "--charter cannot be combined with --resume" in result.output


def test_a_charter_that_is_not_there_is_named_before_anything_starts(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["build", "--charter", str(tmp_path / "nope.yaml")])

    assert result.exit_code != 0
    assert "nope.yaml" in _uncolored(result.output)


def _uncolored(output: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def test_with_no_runs_yet_the_listing_says_so_rather_than_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first thing a new user plausibly types. The checkpoint tables are created on
    first write, so before any run exists this used to raise `no such table: checkpoints`
    out of sqlite."""
    monkeypatch.setenv("OKF_LOREMASTER_CACHE_DIR", str(tmp_path / "cache"))
    result = runner.invoke(app, ["runs"])

    assert result.exit_code == 0
    assert "no runs in" in result.output


@pytest.mark.parametrize("command", ["build"])
def test_cli_defaults_match_the_constants(command: str) -> None:
    """The flags spell out numbers the schema also declares. This is the tie.

    `cli.py` writes them as literals so that `--help` does not have to import pydantic,
    which means nothing but this test stops the two from drifting. A CLI default that
    disagrees with `DEFAULT_TARGET_PAPERS` is not a cosmetic mismatch: the charter node
    overwrites whatever the model drafted with the flag's value, so the constant would
    quietly stop being the default of anything.
    """
    from okf_loremaster.schemas.charter import (
        DEFAULT_MAX_TOPICS,
        DEFAULT_TARGET_PAPERS,
        DEFAULT_TOPIC_PAPER_MAX,
        DEFAULT_TOPIC_PAPER_MIN,
    )

    group = typer.main.get_command(app)
    params = {
        param.name: param.default
        for param in group.commands[command].params  # type: ignore[attr-defined]
    }
    assert params["target_papers"] == DEFAULT_TARGET_PAPERS
    assert params["topic_paper_min"] == DEFAULT_TOPIC_PAPER_MIN
    assert params["topic_paper_max"] == DEFAULT_TOPIC_PAPER_MAX
    assert params["max_topics"] == DEFAULT_MAX_TOPICS
