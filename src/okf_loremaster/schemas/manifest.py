"""The run manifest written into the bundle's root `index.md`.

A bundle has to be able to answer, months later and without this tool installed, what
it was built from and whether it can still be trusted: which prompt, which charter,
which models, how many papers survived each stage, what it cost, and when it goes
stale.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Self

from pydantic import Field, model_validator

from okf_loremaster.schemas.candidates import ExecutedQuery
from okf_loremaster.schemas.common import Model

__all__ = [
    "DEFAULT_FRESHNESS_DAYS",
    "BundleCounts",
    "CostSummary",
    "RunManifest",
    "TopicSummary",
]

# How long a literature bundle stays current enough to rely on without a rebuild.
# Half a year is a judgment about publication rates, not a hard expiry: `stale_after`
# is advisory metadata a downstream reader may act on.
DEFAULT_FRESHNESS_DAYS = 180


class TopicSummary(Model):
    slug: str
    title: str = ""
    papers: int = 0
    # Split out because the two are different grades of evidence and the ratio is the
    # single most useful quality number for a topic.
    full_text: int = 0
    abstract_only: int = 0


class BundleCounts(Model):
    """Corpus size at each stage. The shape of the funnel, in one line."""

    found: int = 0
    unique: int = 0
    screened: int = 0
    included: int = 0
    curated: int = 0
    full_text_fetched: int = 0
    extracted: int = 0
    emitted: int = 0


class CostSummary(Model):
    """Token and USD totals, rendered once and stored as rendered.

    `display` is produced by the router's `format_cost`, the one function allowed to
    turn these numbers into a string. Storing the rendered form rather than
    re-formatting at write time means the manifest cannot disagree with what the run
    printed, and the validator below makes it impossible to persist a total that reads
    as complete when it is not.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    unpriced_calls: int = 0
    display: str = "$0.00"

    @model_validator(mode="after")
    def _display_never_hides_unpriced_calls(self) -> Self:
        """`$0.00` may only ever mean zero calls.

        LiteLLM returns 0.0 for a model it cannot price, so "free" and "no idea" arrive
        as the same float. A manifest that records the first when it meant the second
        is worse than one with no cost line at all — it looks like good news.
        """
        if self.unpriced_calls == 0:
            return self
        if self.unpriced_calls == self.calls and self.display.startswith("$"):
            raise ValueError(
                "every call was unpriced, so the cost display must say so rather than "
                f"showing an amount (got {self.display!r})"
            )
        if 0 < self.unpriced_calls < self.calls and "unpriced" not in self.display:
            raise ValueError(
                f"{self.unpriced_calls} of {self.calls} calls were unpriced, so the "
                f"cost display must say so (got {self.display!r})"
            )
        return self

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class RunManifest(Model):
    run_id: str
    prompt: str = ""
    charter_digest: str = ""
    tool_version: str = ""

    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Role name to model id, so a bundle records what actually produced it rather than
    # what the config happened to say when someone later read it.
    models: dict[str, str] = Field(default_factory=dict)

    counts: BundleCounts = Field(default_factory=BundleCounts)
    topics: list[TopicSummary] = Field(default_factory=list)
    queries: list[ExecutedQuery] = Field(default_factory=list)

    # How each query was retrieved, which is what decides whether repeating one gets the
    # same papers. `retmax` caps every query, so a query with more hits than the cap
    # contributed only its first N — and `sort` is what "first" means. PubMed's relevance
    # ranking is recomputed as the index grows, so a capped query repeated later can
    # return a different N of the same hits while a query under the cap is exact. Recorded
    # per run rather than per query because the search node applies one value to all of
    # them; if that ever stops being true these move onto `ExecutedQuery`.
    retmax: int = 0
    sort: str = ""

    # The `--basis` value the run was built under, as typed. Recorded because the
    # per-paper `text_basis` cannot answer the question a reader actually has: a bundle
    # where every document says `abstract` is a thin corpus if that is all PMC served and
    # a deliberate one if it was asked for, and only this line tells them apart. Empty for
    # a bundle built before the flag existed, which read whatever it could get — the same
    # thing `any` means, and left empty rather than backfilled because a manifest should
    # not claim a decision nobody made.
    text_basis_policy: str = ""

    # Whether `# Abstract` was written into the documents. False under `--no-abstract`.
    # Recorded for the same reason `text_basis_policy` is: the documents cannot say it
    # themselves. That heading is optional either way — roughly one PubMed record in ten
    # carries no abstract at all — so a corpus with none of them is indistinguishable from
    # a corpus PubMed had nothing to serve for, and those are not the same bundle. True for
    # one built before the flag existed, which is what those bundles did.
    abstracts: bool = True
    cost: CostSummary = Field(default_factory=CostSummary)

    # Resolved at index time, not read from config: the descriptor has to name the
    # model that produced the vectors, revision included.
    embed_model: str = ""
    embed_revision: str = ""

    stale_after: date | None = None
    warnings: list[str] = Field(default_factory=list)

    # `human:<id>` when someone signed the bundle off under `--review`, empty otherwise.
    # Empty is the honest answer, and it is what the absence of a `verified` block in
    # every document already says.
    verified_by: str = ""

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def with_staleness(self, *, built_on: date, days: int = DEFAULT_FRESHNESS_DAYS) -> Self:
        updated = self.model_copy(deep=True)
        updated.stale_after = built_on + timedelta(days=days)
        return updated
