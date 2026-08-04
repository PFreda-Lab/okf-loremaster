"""Configuration. Every environment variable the package reads is declared here.

Nothing else in the package touches `os.environ`. Failures name the exact variable
that is missing or wrong, because a misconfigured run that fails three nodes deep with
an opaque provider error costs far more than a loud one at startup.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from platformdirs import user_cache_path
from pydantic import AliasChoices, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "OKF_LOREMASTER_"


class Role(StrEnum):
    """Model tiers. Nodes ask for a role; config decides which model serves it."""

    FAST = "fast"
    MID = "mid"
    DEEP = "deep"


class ConfigError(RuntimeError):
    """Raised with a message that names the offending environment variable."""


def env_file_candidates() -> tuple[Path, ...]:
    """Env files to load, in increasing order of precedence.

    A project-local `.env` wins over a user-level one, so a per-project override works
    without editing anything global. Set `OKF_LOREMASTER_ENV_FILE` to use exactly one
    file and skip the search — useful for keeping credentials out of a synced folder.
    """
    explicit = os.environ.get(f"{ENV_PREFIX}ENV_FILE")
    if explicit:
        return (Path(explicit).expanduser(),)
    return (
        Path.home() / ".config" / "okf-loremaster" / ".env",
        Path(".env"),
    )


class Settings(BaseSettings):
    """All configuration. Field `foo` reads `OKF_LOREMASTER_FOO` unless aliased."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=env_file_candidates(),
        env_file_encoding="utf-8",
        extra="ignore",
        # Fields here are named model_fast/mid/deep; pydantic reserves the "model_"
        # namespace by default and would warn on every import.
        protected_namespaces=(),
        # Without this, a field carrying a validation_alias (api_key, api_base,
        # hf_home) can only be populated by its alias, so `Settings(api_key=...)`
        # is silently ignored and the ambient environment wins instead. Tests and
        # the selftest both depend on constructing Settings directly.
        validate_by_name=True,
        validate_by_alias=True,
    )

    # --- LLM routing -------------------------------------------------------
    # Passed verbatim to LiteLLM, so any provider it supports works.
    model_fast: str = ""
    model_mid: str = ""
    model_deep: str = ""

    # Credentials are conventionally unprefixed, so accept both spellings.
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices(f"{ENV_PREFIX}API_KEY", "ANTHROPIC_API_KEY"),
    )
    api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}API_BASE", "ANTHROPIC_BASE_URL"),
    )

    concurrency_fast: int = 8
    concurrency_mid: int = 4
    concurrency_deep: int = 2
    max_retries: int = 4
    request_timeout: float = 120.0

    # --- Cost accounting ---------------------------------------------------
    # USD per 1M tokens. Used only when LiteLLM cannot price a model itself, which
    # is the normal case behind a gateway or a custom deployment name.
    price_fast_in: float | None = None
    price_fast_out: float | None = None
    price_mid_in: float | None = None
    price_mid_out: float | None = None
    price_deep_in: float | None = None
    price_deep_out: float | None = None

    max_usd: float | None = None

    # --- NCBI --------------------------------------------------------------
    # An API key raises the E-utilities rate ceiling from 3 to 10 requests/second.
    # `ncbi_email` is not optional in practice: NCBI asks for a contact address on
    # every request and blocks traffic that omits it.
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None
    ncbi_tool: str = "okf-loremaster"

    # --- HTTP --------------------------------------------------------------
    http_timeout: float = 30.0
    http_max_retries: int = 4
    http_cache_enabled: bool = True
    # Bibliographic records are effectively immutable; a month is conservative.
    http_cache_ttl_days: int = 30

    # --- Embeddings --------------------------------------------------------
    # Must be locally runnable: downstream consumers reject remote embedders on
    # attach. Pinned by revision so a rebuild reproduces the same vectors.
    embed_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embed_revision: str | None = None

    # Consumed by the Hugging Face libraries, not by us, but declared here so that
    # `os.environ` stays out of the rest of the package and `init` can check it.
    hf_home: Path | None = Field(default=None, validation_alias=AliasChoices("HF_HOME"))

    # --- Review ------------------------------------------------------------
    # Who a `--review` sign-off is attributed to, recorded as `human:<id>`. Empty falls
    # back to the OS login name; set it when that is a service account or a shared box.
    reviewer_id: str = ""

    # --- Paths -------------------------------------------------------------
    cache_dir: Path = Field(default_factory=lambda: user_cache_path("okf-loremaster"))
    output_dir: Path = Path("bundles")

    # --- Accessors ---------------------------------------------------------

    def model_for(self, role: Role) -> str:
        model = {
            Role.FAST: self.model_fast,
            Role.MID: self.model_mid,
            Role.DEEP: self.model_deep,
        }[role]
        if not model:
            raise ConfigError(
                f"No model bound to the {role.value.upper()} role. "
                f"Set {ENV_PREFIX}MODEL_{role.value.upper()} in your .env."
            )
        return model

    def price_for(self, role: Role) -> tuple[float | None, float | None]:
        """(input, output) USD per 1M tokens, or (None, None) if not fully set."""
        pair = {
            Role.FAST: (self.price_fast_in, self.price_fast_out),
            Role.MID: (self.price_mid_in, self.price_mid_out),
            Role.DEEP: (self.price_deep_in, self.price_deep_out),
        }[role]
        # A half-configured price would silently undercount, so require both.
        if pair[0] is None or pair[1] is None:
            return (None, None)
        return pair

    def concurrency_for(self, role: Role) -> int:
        return {
            Role.FAST: self.concurrency_fast,
            Role.MID: self.concurrency_mid,
            Role.DEEP: self.concurrency_deep,
        }[role]

    # --- Preflight ---------------------------------------------------------

    def missing_for_llm(self) -> list[str]:
        """Variable names required before any model call can be made."""
        missing: list[str] = []
        for role in Role:
            if not {
                Role.FAST: self.model_fast,
                Role.MID: self.model_mid,
                Role.DEEP: self.model_deep,
            }[role]:
                missing.append(f"{ENV_PREFIX}MODEL_{role.value.upper()}")
        if not self.api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing

    def require_llm(self) -> None:
        missing = self.missing_for_llm()
        if missing:
            names = "\n  ".join(missing)
            raise ConfigError(
                f"Cannot make model calls. These are unset:\n  {names}\n"
                "Copy .env.example to .env and fill them in, or run "
                "`okf-loremaster init`."
            )

    def missing_for_ncbi(self) -> list[str]:
        """NCBI asks for a contact address on every request; the key is optional."""
        return [] if self.ncbi_email else [f"{ENV_PREFIX}NCBI_EMAIL"]

    def unpriced_roles(self) -> list[Role]:
        """Roles with no local price override, which may end up unpriced."""
        return [role for role in Role if self.price_for(role) == (None, None)]

    def hf_home_warning(self) -> str | None:
        """Flag a model cache sitting in a sync folder.

        The hub cache links `snapshots/` into `blobs/` with symlinks, which sync
        clients variously break, duplicate, or upload as gigabytes of churn. The
        failure is slow and confusing, so it is worth catching at setup.
        """
        if self.hf_home is None:
            return None
        parts = {part.lower() for part in self.hf_home.parts}
        for marker in ("onedrive", "dropbox", "google drive", "googledrive", "icloud drive"):
            if any(marker in part for part in parts):
                return (
                    f"HF_HOME is inside a synced folder ({self.hf_home}). The Hugging Face "
                    "cache uses symlinks that sync clients corrupt. Move it outside, "
                    "e.g. ~/.cache/huggingface."
                )
        return None


def load_settings() -> Settings:
    """Load settings, converting pydantic's error format into a named-variable one."""
    try:
        return Settings()
    except ValidationError as exc:
        lines: list[str] = []
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "<unknown>"
            lines.append(f"  {ENV_PREFIX}{field.upper()}: {err['msg']}")
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from exc
