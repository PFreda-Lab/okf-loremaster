"""Configuration: role binding, priced/unpriced resolution, and loud failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from okf_loremaster.config import ConfigError, Role, Settings, env_file_candidates


def test_role_binding(settings_factory: Any) -> None:
    settings = settings_factory(model_fast="a", model_mid="b", model_deep="c")
    assert settings.model_for(Role.FAST) == "a"
    assert settings.model_for(Role.MID) == "b"
    assert settings.model_for(Role.DEEP) == "c"


def test_unbound_role_names_the_variable(settings_factory: Any) -> None:
    settings = settings_factory(model_fast="a", model_mid="b")
    with pytest.raises(ConfigError) as excinfo:
        settings.model_for(Role.DEEP)
    # The whole point of the error is that it tells you what to set.
    assert "OKF_LOREMASTER_MODEL_DEEP" in str(excinfo.value)


def test_missing_variables_are_listed(settings_factory: Any) -> None:
    settings = settings_factory()
    missing = settings.missing_for_llm()
    assert "OKF_LOREMASTER_MODEL_FAST" in missing
    assert "OKF_LOREMASTER_MODEL_MID" in missing
    assert "OKF_LOREMASTER_MODEL_DEEP" in missing
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
    settings = settings_factory(price_deep_in=15.0)
    assert settings.price_for(Role.DEEP) == (None, None)
    assert Role.DEEP in settings.unpriced_roles()


def test_fully_configured_price_resolves(settings_factory: Any) -> None:
    settings = settings_factory(price_deep_in=15.0, price_deep_out=75.0)
    assert settings.price_for(Role.DEEP) == (15.0, 75.0)
    assert Role.DEEP not in settings.unpriced_roles()


def test_concurrency_defaults_descend_with_cost(settings_factory: Any) -> None:
    settings = settings_factory()
    assert settings.concurrency_for(Role.FAST) >= settings.concurrency_for(Role.MID)
    assert settings.concurrency_for(Role.MID) >= settings.concurrency_for(Role.DEEP)


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


def test_env_file_search_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Project .env must outrank the user-level one, and an explicit path must win."""
    candidates = env_file_candidates()
    assert candidates[-1] == Path(".env")
    assert len(candidates) == 2

    monkeypatch.setenv("OKF_LOREMASTER_ENV_FILE", "~/somewhere/.env")
    explicit = env_file_candidates()
    assert len(explicit) == 1
    assert not str(explicit[0]).startswith("~")
