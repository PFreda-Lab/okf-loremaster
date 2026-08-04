"""Configuration: role binding, priced/unpriced resolution, and loud failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from okf_loremaster.config import ConfigError, Role, Settings, env_file_candidates


def test_role_binding(settings_factory: Any) -> None:
    settings = settings_factory(model_fast="a", model_balanced="b", model_reasoning="c")
    assert settings.model_for(Role.FAST) == "a"
    assert settings.model_for(Role.BALANCED) == "b"
    assert settings.model_for(Role.REASONING) == "c"


def test_unbound_role_names_the_variable(settings_factory: Any) -> None:
    settings = settings_factory(model_fast="a", model_balanced="b")
    with pytest.raises(ConfigError) as excinfo:
        settings.model_for(Role.REASONING)
    # The whole point of the error is that it tells you what to set.
    assert "OKF_LOREMASTER_MODEL_REASONING" in str(excinfo.value)


def test_missing_variables_are_listed(settings_factory: Any) -> None:
    settings = settings_factory()
    missing = settings.missing_for_llm()
    assert "OKF_LOREMASTER_MODEL_FAST" in missing
    assert "OKF_LOREMASTER_MODEL_BALANCED" in missing
    assert "OKF_LOREMASTER_MODEL_REASONING" in missing
    assert "ANTHROPIC_API_KEY" in missing

    with pytest.raises(ConfigError) as excinfo:
        settings.require_llm()
    assert "OKF_LOREMASTER_MODEL_FAST" in str(excinfo.value)


def test_api_key_accepts_either_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unprefixed")
    assert Settings(_env_file=None).api_key == "unprefixed"

    monkeypatch.setenv("OKF_LOREMASTER_API_KEY", "prefixed")
    assert Settings(_env_file=None).api_key == "prefixed"


def test_half_configured_price_counts_as_unpriced(settings_factory: Any) -> None:
    """An input price with no output price would silently undercount every call."""
    settings = settings_factory(price_reasoning_in=15.0)
    assert settings.price_for(Role.REASONING) == (None, None)
    assert Role.REASONING in settings.unpriced_roles()


def test_fully_configured_price_resolves(settings_factory: Any) -> None:
    settings = settings_factory(price_reasoning_in=15.0, price_reasoning_out=75.0)
    assert settings.price_for(Role.REASONING) == (15.0, 75.0)
    assert Role.REASONING not in settings.unpriced_roles()


def test_concurrency_defaults_descend_with_cost(settings_factory: Any) -> None:
    settings = settings_factory()
    assert settings.concurrency_for(Role.FAST) >= settings.concurrency_for(Role.BALANCED)
    assert settings.concurrency_for(Role.BALANCED) >= settings.concurrency_for(Role.REASONING)


@pytest.mark.parametrize(
    "path",
    [
        "/Users/x/Library/CloudStorage/OneDrive-Example/cache",
        "/Users/x/Dropbox/hf",
        "/Users/x/Google Drive/hf",
    ],
)
def test_hf_home_in_sync_folder_is_flagged(settings_factory: Any, path: str) -> None:
    warning = settings_factory(hf_home=Path(path)).hf_home_warning()
    assert warning is not None
    assert "symlink" in warning


def test_hf_home_outside_sync_folder_is_quiet(settings_factory: Any) -> None:
    assert settings_factory(hf_home=Path("/Users/x/.cache/huggingface")).hf_home_warning() is None
    assert settings_factory().hf_home_warning() is None


def test_relative_output_lands_under_the_output_dir(settings_factory: Any) -> None:
    """The point of the flag: a name, not a path. `-o hiv` writes `bundles/hiv`."""
    settings = settings_factory()
    assert settings.output_dir == Path("bundles")
    assert settings.resolve_output(Path("hiv-suppression")) == Path("bundles/hiv-suppression")
    assert settings.resolve_output(Path("a/b")) == Path("bundles/a/b")


def test_output_dir_is_not_nested_inside_itself(settings_factory: Any) -> None:
    """`-o bundles/hiv` is what a person types before reading the help. Same place."""
    settings = settings_factory()
    assert settings.resolve_output(Path("bundles/hiv")) == Path("bundles/hiv")
    # Only a prefix match counts. A topic that merely starts with the same letters is
    # still a name to be placed inside.
    assert settings.resolve_output(Path("bundlesearch")) == Path("bundles/bundlesearch")


def test_absolute_output_is_used_as_given(settings_factory: Any) -> None:
    """The way out. An absolute path is never rewritten, whatever OUTPUT_DIR says."""
    settings = settings_factory(output_dir=Path("/srv/okf"))
    assert settings.resolve_output(Path("/tmp/elsewhere")) == Path("/tmp/elsewhere")
    # An absolute output dir still collects relative names, and cannot be prefix-matched
    # by one: `-o srv/okf/x` under `/srv/okf` is a different place, not the same one.
    assert settings.resolve_output(Path("hiv")) == Path("/srv/okf/hiv")
    assert settings.resolve_output(Path("srv/okf/hiv")) == Path("/srv/okf/srv/okf/hiv")


def test_env_file_search_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Project .env must outrank the user-level one, and an explicit path must win."""
    candidates = env_file_candidates()
    assert candidates[-1] == Path(".env")
    assert len(candidates) == 2

    monkeypatch.setenv("OKF_LOREMASTER_ENV_FILE", "~/somewhere/.env")
    explicit = env_file_candidates()
    assert len(explicit) == 1
    assert not str(explicit[0]).startswith("~")
