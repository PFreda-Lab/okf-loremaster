"""The two things step 7 adds to the command surface.

`validate <bundle>` is the same code the graph runs, reached without a run — which is
the only way to check a bundle somebody else built, or one built six months ago. Its
exit code is the whole point: a gate that reported failure and exited zero would be a
gate no script could act on.

The `--review` refusal is the other. `--yes`, `--dry-run` and `--json` each mean nobody
is going to look, and signing anyway would write `by: "human:<id>"` naming a person who
never saw the bundle. That is a false attestation rather than a weak one, so the
combination is refused rather than degraded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from okf_loremaster.cli import app

from graph_runs import full_run

runner = CliRunner()


async def test_validate_reports_a_good_bundle_and_exits_zero(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    bundle = Path(run.state["bundle"])

    result = runner.invoke(app, ["validate", str(bundle)])

    assert result.exit_code == 0, result.output
    assert "passes" in result.output


async def test_validate_names_what_is_wrong_and_exits_nonzero(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    bundle = Path(run.state["bundle"])
    (bundle / "index.md").unlink()

    result = runner.invoke(app, ["validate", str(bundle)])

    assert result.exit_code == 1
    assert "error" in result.output
    assert "index.md" in result.output


def test_validate_says_a_bundle_is_missing_rather_than_reporting_on_nothing(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "not-a-bundle")])

    assert result.exit_code == 1
    assert "not-a-bundle" in result.output


@pytest.mark.parametrize("flag", ["--yes", "--dry-run", "--json"])
def test_review_is_refused_with_any_flag_that_means_nobody_will_look(flag: str) -> None:
    result = runner.invoke(app, ["build", "a prompt", "--review", flag])

    assert result.exit_code == 1
    assert "--review cannot be combined with" in result.output
    assert flag in result.output
