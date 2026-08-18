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
    BALANCED = "balanced"
    REASONING = "reasoning"


class Effort(StrEnum):
    """How hard a reasoning model may think before it answers, per tier.

    The names are LiteLLM's `reasoning_effort` vocabulary, which is why they are these
    seven and not a scale of our own: it translates them per provider — to a thinking
    budget for Anthropic, to the native parameter for OpenAI — so one setting means the
    same thing whichever provider a deployment points at. Unset is not `NONE`: unset
    sends nothing at all and takes the provider's default, while `NONE` asks explicitly
    for no reasoning, and on a model that reasons by default those differ.
    """

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    NONE = "none"


# What each level costs in thinking tokens, mirroring LiteLLM's defaults.
#
# Held here because the reply allowance has to be sized around it and LiteLLM will not do
# that: Anthropic requires `max_tokens > thinking.budget_tokens` and counts thinking
# against the same ceiling as the reply, and while LiteLLM caps the budget for a caller
# who passes `thinking` directly, the `reasoning_effort` path sets one and never checks.
# Screening asks for 256 tokens, which is below every budget on this table, so a run with
# effort set on FAST and no headroom added would 400 on every paper it screened.
EFFORT_THINKING_TOKENS = {
    Effort.MINIMAL: 1024,
    Effort.LOW: 1024,
    Effort.MEDIUM: 2048,
    Effort.HIGH: 4096,
    Effort.XHIGH: 8192,
    Effort.MAX: 16384,
    Effort.NONE: 0,
}


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
    model_balanced: str = ""
    model_reasoning: str = ""

    # How hard each tier may think, or unset to take whatever the provider does by
    # default. Per tier rather than one global setting because the tiers are asked for
    # different work: the charter is a judgment call made once, extraction is one call per
    # paper and sets a run's price, and screening is a yes or no on an abstract that a
    # thinking budget would multiply the cost of for no better answer.
    #
    # `max_tokens` is raised by the tier's budget wherever one is set, so a node's
    # measured reply allowance still holds and the thinking happens on top of it.
    effort_fast: Effort | None = None
    effort_balanced: Effort | None = None
    effort_reasoning: Effort | None = None

    # Credentials are conventionally unprefixed, so accept both spellings.
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices(f"{ENV_PREFIX}API_KEY", "ANTHROPIC_API_KEY"),
    )
    api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices(f"{ENV_PREFIX}API_BASE", "ANTHROPIC_BASE_URL"),
    )

    # Screening submits every pooled paper at once and these semaphores are what meter
    # it. The binding limit in practice is tokens per minute, not requests: eight
    # abstracts in flight is a burst no backoff can smooth out, and a run once lost 56
    # of 252 screening calls to it. Four is slower and finishes.
    concurrency_fast: int = 4
    # BALANCED is what sets a run's wall-clock, because extraction lives here: one call
    # per kept paper, so a 150-paper bundle makes 150 of them against every other node's
    # handful. At 2 that serializes into hours, and at 3 a run whose calls take ~30s each
    # spends 25 minutes in one node.
    #
    # Raised past FAST rather than held under it, because the tiers were never competing
    # for one budget. The limit that binds is per model per minute — the 429s name it,
    # `UserByModelByMinute...` — so screening's four-at-a-time on the fast model buys
    # extraction nothing on the balanced one. The old ordering read cost off the tier
    # names and throttled the node that needed the room most.
    concurrency_balanced: int = 6
    # One charter call per run. Concurrency here is a formality.
    concurrency_reasoning: int = 3
    # Attempts, not retries: the warnings count up to `max_retries - 1`. Rate limits
    # clear on a 60-second window, so a run needs enough attempts to outlast one.
    max_retries: int = 6
    # An extraction reads 6,000 tokens of source and writes up to
    # `MAX_EXTRACTION_TOKENS` back. At 120s that call times out on its own success.
    request_timeout: float = 300.0

    # --- Cost accounting ---------------------------------------------------
    # USD per 1M tokens, and the first thing consulted rather than the last. LiteLLM
    # prices from a static table shipped in its wheel, so it cannot price a gateway
    # deployment name at all and keeps quoting release-day figures for the public models
    # it does know. Set these and neither problem reaches the ledger.
    price_fast_in: float | None = None
    price_fast_out: float | None = None
    price_balanced_in: float | None = None
    price_balanced_out: float | None = None
    price_reasoning_in: float | None = None
    price_reasoning_out: float | None = None

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

    # A CA bundle to verify against, instead of the default trust store. Needed on a
    # network whose proxy terminates TLS and presents its own certificate: verification
    # then fails against a certificate that is not in any public store, and the failure
    # looks exactly like the service being down. Unset means the default store. Never
    # set this to disable verification — there is deliberately no option for that, since
    # the whole provenance claim rests on the bytes having come from who they say.
    ca_bundle: Path | None = None

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

    # How many past runs the checkpoint store keeps. A run costs 100 to 350 MB there,
    # because LangGraph serializes the whole state — candidates, abstracts, full texts,
    # extractions — once per node, and nothing expires. Applied when a build starts, so
    # the store holds this many plus the one being written. Set to 0 to keep everything.
    checkpoint_keep_runs: int = 5

    # Ceilings on the three things that accumulate, in MB, so that none of them can grow
    # without limit however long the tool is used. Applied at both ends of a build, so
    # between builds each store is under its ceiling and only a run in flight is over —
    # see `okf_loremaster.retention`. 0 turns one off.
    #
    # Checkpoints are scratch, and the count above is meant to be what binds: five runs
    # measure around 900 MB, so a ceiling of 1024 would quietly cut how many are
    # resumable rather than catching an outlier. This is set clear of that. The other two
    # are not scratch — they are what makes a rerun cheap, and the extraction cache is
    # what makes it free — so they sit where ordinary use will not reach them.
    checkpoint_max_mb: int = 2048
    http_cache_max_mb: int = 1024
    extraction_cache_max_mb: int = 512

    # --- Accessors ---------------------------------------------------------

    def model_for(self, role: Role) -> str:
        model = {
            Role.FAST: self.model_fast,
            Role.BALANCED: self.model_balanced,
            Role.REASONING: self.model_reasoning,
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
            Role.BALANCED: (self.price_balanced_in, self.price_balanced_out),
            Role.REASONING: (self.price_reasoning_in, self.price_reasoning_out),
        }[role]
        # A half-configured price would silently undercount, so require both.
        if pair[0] is None or pair[1] is None:
            return (None, None)
        return pair

    def concurrency_for(self, role: Role) -> int:
        return {
            Role.FAST: self.concurrency_fast,
            Role.BALANCED: self.concurrency_balanced,
            Role.REASONING: self.concurrency_reasoning,
        }[role]

    def effort_for(self, role: Role) -> Effort | None:
        """How hard this tier may think, or `None` to send nothing and take the default."""
        return {
            Role.FAST: self.effort_fast,
            Role.BALANCED: self.effort_balanced,
            Role.REASONING: self.effort_reasoning,
        }[role]

    def thinking_tokens_for(self, role: Role) -> int:
        """Tokens this tier's effort will spend thinking, on top of its reply allowance."""
        effort = self.effort_for(role)
        return 0 if effort is None else EFFORT_THINKING_TOKENS[effort]

    # --- Preflight ---------------------------------------------------------

    def missing_for_llm(self) -> list[str]:
        """Variable names required before any model call can be made."""
        missing: list[str] = []
        for role in Role:
            if not {
                Role.FAST: self.model_fast,
                Role.BALANCED: self.model_balanced,
                Role.REASONING: self.model_reasoning,
            }[role]:
                missing.append(f"{ENV_PREFIX}MODEL_{role.value.upper()}")
        if not self.api_key:
            # Named by our own spelling, not the Anthropic alias. The template writes
            # this one and it is the only spelling that is right for every provider —
            # being told to set ANTHROPIC_API_KEY while configuring Azure or a local
            # server reads as "this tool only speaks to Anthropic", which it does not.
            missing.append(f"{ENV_PREFIX}API_KEY")
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

    def resolve_output(self, out: Path) -> Path:
        """Place a `-o` value inside the configured output directory.

        Everything this tool writes belongs under one folder, so `-o my-bundle` should
        not need `bundles/` typed in front of it every time. A relative name is placed
        under `output_dir`; an absolute path is the deliberate way out and is used as
        given.

        A relative path that already starts with `output_dir` is taken as read rather
        than nested inside itself, so `-o bundles/my-bundle` and `-o my-bundle` name one
        place. That only applies when `output_dir` is itself relative — once it is
        absolute, a relative `-o` cannot be naming it.
        """
        if out.is_absolute():
            return out
        base = self.output_dir
        if not base.is_absolute() and out.parts[: len(base.parts)] == base.parts:
            return out
        return base / out

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
    """Load settings, converting pydantic's error format into a named-variable one.

    The file list is resolved here rather than taken from `model_config`, whose copy is
    frozen when this module is imported. Passing it per call is what makes
    `OKF_LOREMASTER_ENV_FILE` mean the same thing whenever it is set, and what lets a
    test ask for a run that reads no file at all instead of silently picking up whatever
    `.env` happens to sit in the working directory.
    """
    try:
        # `_env_file` is pydantic-settings' documented per-call override. The pydantic
        # mypy plugin builds `__init__` out of the model's fields and does not know the
        # dunder-prefixed settings arguments, so it reads as an unexpected keyword.
        return Settings(_env_file=env_file_candidates())  # type: ignore[call-arg]
    except ValidationError as exc:
        lines: list[str] = []
        for err in exc.errors():
            field = str(err["loc"][0]) if err["loc"] else "<unknown>"
            lines.append(f"  {ENV_PREFIX}{field.upper()}: {err['msg']}")
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from exc
