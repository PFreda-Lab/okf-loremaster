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
        DEFAULT_TARGET_PAPERS,
        DEFAULT_TOPIC_MAX,
        DEFAULT_TOPIC_MIN,
    )

    group = typer.main.get_command(app)
    params = {
        param.name: param.default
        for param in group.commands[command].params  # type: ignore[attr-defined]
    }
    assert params["target_papers"] == DEFAULT_TARGET_PAPERS
    assert params["topic_min"] == DEFAULT_TOPIC_MIN
    assert params["topic_max"] == DEFAULT_TOPIC_MAX
