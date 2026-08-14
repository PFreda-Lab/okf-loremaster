"""The command surface: what `build` refuses, and where its defaults come from.

Everything here decides before the graph starts, which is the only place several of
these can be caught at all.

The `--review` refusal is the human sign-off. `--dry-run` and `--json` each mean nobody
is going to look, and signing anyway would write `by: "human:<id>"` naming a person who
never saw the bundle. That is a false attestation rather than a weak one, so the
combination is refused rather than degraded. An autonomous run is not on that list:
nobody steered the search, but somebody still reads the bundle at the end, which is what
the signature is about.

The rest is the same shape one level down — a question that may come from the command
line or from a checkpoint, a charter that can be handed back to a fresh run but not to a
resumed one, and flag defaults written as literals that have to keep agreeing with the
constants they mirror.

**Validation has no command, by design.** `validate` — with `charter`, `index`, `export`
and `inspect` — was a debugging door cut into a graph node, and all five were closed in
`7b131c4`. The node still runs in every build; `test_okf_validate.py` covers it, including
the hand-edited bundle a standalone command would have been for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer.main
from typer.testing import CliRunner

from okf_loremaster.cli import app

runner = CliRunner()


# --- the human sign-off -------------------------------------------------------


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


def test_a_build_with_no_question_and_nothing_to_resume_says_which_to_give(
    llm_configured: None,
) -> None:
    """`prompt` became optional so that `--resume` could supply it from the checkpoint.
    That makes an empty `build` a thing typer will now accept, so the sentence it stops
    with has to name both ways of answering the question it is missing."""
    result = runner.invoke(app, ["build"])

    assert result.exit_code == 1
    assert "no prompt" in result.output
    assert "--resume" in result.output


def test_resuming_does_not_require_the_question_to_be_typed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_configured: None
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


# --- the template `init` copies ------------------------------------------------


def test_a_checkout_copy_of_the_template_wins_over_the_packaged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a checkout `.env.example` is the file being edited, so it is the one to copy.

    The packaged copy is a snapshot taken at build time. Preferring it would mean an
    edit to the template did not show up in the `.env` that `init` writes next to it.
    """
    from okf_loremaster.cli import _env_template

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("FROM_THE_CHECKOUT=\n", encoding="utf-8")

    origin, text = _env_template()

    assert text == "FROM_THE_CHECKOUT=\n"
    assert origin == ".env.example"


def test_the_packaged_template_is_used_when_there_is_no_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `pip install` has no `.env.example` anywhere, which used to make `init` a no-op.

    It printed "neither .env.example nor .env found" and wrote nothing, leaving anyone who
    installed from PyPI to reconstruct an 8 KB annotated config out of the README.
    """
    import okf_loremaster.cli as cli_module

    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / "env.example").write_text("FROM_THE_WHEEL=\n", encoding="utf-8")
    monkeypatch.setattr(cli_module.resources, "files", lambda _: packaged)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    origin, text = cli_module._env_template()

    assert text == "FROM_THE_WHEEL=\n"
    assert origin == "the packaged template"


def test_init_writes_an_env_with_no_checkout_in_sight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end shape of the same thing: the file lands, from the packaged copy."""
    import okf_loremaster.cli as cli_module

    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / "env.example").write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    monkeypatch.setattr(cli_module.resources, "files", lambda _: packaged)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    result = runner.invoke(app, ["init"])

    assert (empty / ".env").read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=\n"
    assert "wrote" in result.output


def test_init_does_not_overwrite_an_env_that_already_has_secrets_in_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env` is the one file in the tree holding a real key. Clobbering it on a second
    `init` would be silent and unrecoverable, so overwriting takes `--force`."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-real\n", encoding="utf-8")

    runner.invoke(app, ["init"])

    assert (tmp_path / ".env").read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=sk-real\n"

    runner.invoke(app, ["init", "--force"])

    assert (tmp_path / ".env").read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=\n"
