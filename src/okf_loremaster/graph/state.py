"""What flows between nodes, and what does not.

Two kinds of thing a node needs, kept strictly apart:

`RunState` is the data. It is checkpointed to SQLite after every node, so everything in
it must survive a serialization round trip and be worth resuming from. That is what
makes `--resume <run-id>` free and what lets a confirmation pause interrupt the graph
without holding a process open.

`Deps` is the machinery — the event bus, the HTTP clients, the router, the run's
settings. None of it is state, none of it is serializable, and none of it belongs in a
checkpoint. Nodes are bound to their deps when the graph is built, so a node signature
stays `(state) -> update` from LangGraph's point of view.

`router` is `None` on a dry run. That is the mechanism behind "zero LLM calls", not a
promise about one: a node that tried to make one would raise rather than quietly spend.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from okf_loremaster.clients import Clients
from okf_loremaster.config import Settings
from okf_loremaster.curation import MAX_ROUNDS
from okf_loremaster.events import EventBus, NodeFinished, NodeStarted, Progress, WarningEvent
from okf_loremaster.extraction_cache import ExtractionCache
from okf_loremaster.llm.router import Router
from okf_loremaster.ranking import DEFAULT_LAMBDA, SelectionComparison
from okf_loremaster.schemas import (
    DEFAULT_MAX_TOPICS,
    DEFAULT_TARGET_PAPERS,
    DEFAULT_TOPIC_PAPER_MAX,
    DEFAULT_TOPIC_PAPER_MIN,
    Candidate,
    Charter,
    ConceptRecord,
    CurationResult,
    ExecutedQuery,
    Extraction,
    PaperText,
    QueryPlan,
    RunManifest,
    ScoredCandidate,
    ScreenVerdict,
    TextBasisPolicy,
    VerificationSummary,
)

if TYPE_CHECKING:  # a protocol implemented in `ui`, which imports this module
    from okf_loremaster.emitters.vectors import Embedder
    from okf_loremaster.review import Reviewer

__all__ = ["Deps", "RunState", "initial_state", "span"]


class RunState(TypedDict, total=False):
    """The checkpointed run. Every key is optional; nodes add theirs as they go."""

    run_id: str
    prompt: str
    dry_run: bool
    # Stamped once, when the state is first built. Checkpointed so the manifest of a
    # resumed run reports when the work started rather than when it was picked back up.
    started_at: datetime
    # The run folder, as a string because this is checkpointed. Recorded so a resume
    # without `-o` writes where the run already was — see `run.run_directory`.
    directory: str
    # `--basis`, as its plain string value. In state and not only on `Deps` because it is
    # a claim the bundle makes about itself: "every paper here is abstract-only" means
    # one thing when it was asked for and another when it is what the literature had, and
    # only the manifest can tell them apart. A resume reads it back rather than taking
    # the flag again, so a bundle cannot report a policy its corpus was not chosen under.
    # A string rather than the enum so nothing has to join the checkpoint allowlist.
    basis: str

    # charter
    charter: Charter | None

    # search — accumulated across rounds, not replaced. The conditional re-query edge
    # comes back here, and a second round is meant to add to the corpus rather than
    # become it.
    plan: QueryPlan | None
    executed: list[ExecutedQuery]
    # Search rounds completed. Bounded by `curation.MAX_ROUNDS`; incremented by the
    # search node, which is the node the bound is about.
    rounds: int
    # Filtered query term to the topic slug it was written for. The ranker's only
    # source of topic affinity before screening assigns one.
    query_topic: dict[str, str]
    candidates: list[Candidate]

    # dedupe
    unique: list[Candidate]
    dropped: dict[str, int]

    # rank
    scored: list[ScoredCandidate]
    pool: list[ScoredCandidate]
    comparison: SelectionComparison | None

    # screen — keyed by PMID on the way in and out, so a second round screens only what
    # it has not already paid for.
    verdicts: list[ScreenVerdict]

    # curate
    curation: CurationResult | None
    # The final placement: topic slug to PMIDs, every charter topic present even when
    # empty. What the emitter walks.
    topics: dict[str, list[str]]

    # fulltext — keyed by PMID. The whole prompt block each extraction was shown, kept
    # verbatim because `verification` checks the model's numbers against exactly this
    # string. Already budgeted by the fulltext node, which is why the checkpoint stays
    # a sane size.
    texts: dict[str, PaperText]

    # extract — keyed by PMID, so a resumed run re-extracts only what it has not paid
    # for. Papers whose extraction failed are simply absent.
    extractions: dict[str, Extraction]

    # reconcile
    records: list[ConceptRecord]
    verification: VerificationSummary | None

    # review — the `verified.by` a sign-off attributed the bundle to. Empty when nobody
    # signed, which is also what the absence of a `verified` block in every file means.
    verified_by: str

    # emit_okf
    bundle: str
    manifest: RunManifest | None

    # validate
    validated: bool
    validation_errors: list[str]

    # index_vectors — the store's path, empty when no index was built. A separate key
    # from `bundle` because the index is derived and optional: a bundle without one is
    # complete, and a resumed run must be able to tell "not asked for" from "failed".
    vector_index: str
    vector_chunks: int

    warnings: list[str]


@dataclass
class Deps:
    """Everything a node needs that is not state."""

    settings: Settings
    bus: EventBus
    clients: Clients
    # None on a dry run. A node must check rather than assume.
    router: Router | None = None
    # Where the bundle is written. Decided before the graph starts, so a resumed run
    # cannot write somewhere else than the run it resumed. None falls back to
    # `settings.output_dir / run_id`.
    bundle_dir: Path | None = None
    # None unless `--review`. The review node treats that as "nobody was asked", which
    # is not the same as a decline.
    reviewer: Reviewer | None = None
    # None when vectors were declined. Injected rather than built by the node because the default
    # one downloads a model on first use, and the test suite never reaches the network.
    embedder: Embedder | None = None
    # Papers already read, from this run or an earlier one. Injected for the same reason
    # as the embedder: built by the node, every test would write into the user's real
    # cache directory and start reading each other's fixtures. None means every paper is
    # read fresh, which is correct and merely more expensive.
    extraction_cache: ExtractionCache | None = None

    pool_size: int = 800
    screen_budget: int = 400
    # Which papers a run is willing to read, and how. `ANY` — the default — takes full
    # text where the open-access subset has it and the abstract everywhere else, which is
    # what makes a corpus of 200 papers affordable. The other two restrict the corpus
    # rather than merely the reading: see `rank` for the availability pass that enforces
    # `FULL_TEXT`, and `fulltext` for the BioC call `ABSTRACT` skips.
    basis: TextBasisPolicy = TextBasisPolicy.ANY
    max_queries: int = 12
    # Search rounds allowed, including the first. Capped at `MAX_ROUNDS` by the CLI;
    # 1 turns the conditional re-query edge off entirely.
    max_rounds: int = MAX_ROUNDS
    # Charter fields the user has the last word on, applied by the charter node
    # whether the charter was drafted or supplied.
    target_papers: int = DEFAULT_TARGET_PAPERS
    topic_paper_min: int = DEFAULT_TOPIC_PAPER_MIN
    topic_paper_max: int = DEFAULT_TOPIC_PAPER_MAX
    max_topics: int = DEFAULT_MAX_TOPICS
    # PubMed ids pulled per query. 200 is enough for a query worth planning and small
    # enough that a badly broadened one cannot flood the pool on its own.
    per_query_retmax: int = 200
    mmr_lambda: float = DEFAULT_LAMBDA
    # Captured once, so two nodes in the same run cannot disagree about the year and a
    # replayed run scores identically to the original.
    now_year: int = field(default_factory=lambda: datetime.now(UTC).year)

    def progress(self, node: str, message: str, *, current: int | None = None,
                 total: int | None = None) -> None:
        self.bus.emit(Progress(node=node, message=message, current=current, total=total))

    def warn(self, node: str, message: str) -> None:
        self.bus.emit(WarningEvent(node=node, message=message))


def screen_budget_spent(state: RunState, deps: Deps) -> bool:
    """Whether the screener can still look at anything it has not already seen.

    The budget is global across rounds rather than per round, so once there are as many
    verdicts as it allows, no later round can screen a single new paper however many a
    re-query finds. A real run walked into exactly that: round two planned two queries,
    retrieved 232 new records, ranked 150 of them, screened **zero**, and re-curated the
    identical kept set — paying for a query-planning call and a curation pass that could
    not, by construction, change the outcome.

    Lives here rather than in the router because two callers need the same answer: the
    edge out of `curate` decides on it, and `curate` itself says so out loud, since a
    topic under its floor that cannot be fixed by searching again is a reason to raise
    `--screen-budget` and the run should not make that inference private.
    """
    return len(state.get("verdicts") or []) >= max(0, deps.screen_budget)


def initial_state(
    run_id: str,
    prompt: str,
    *,
    dry_run: bool = False,
    charter: Charter | None = None,
    directory: str = "",
    basis: TextBasisPolicy = TextBasisPolicy.ANY,
) -> RunState:
    return RunState(
        run_id=run_id,
        prompt=prompt,
        dry_run=dry_run,
        started_at=datetime.now(UTC),
        directory=directory,
        basis=basis.value,
        charter=charter,
        warnings=[],
    )


@contextmanager
def span(deps: Deps, node: str) -> Iterator[dict[str, Any]]:
    """Bracket a node with its start and finish events.

    Yields a mutable dict whose `summary` the node sets; that is what the renderer
    prints beside the node name. A context manager rather than a decorator because the
    summary is only known at the end, and the timing has to include it.
    """
    deps.bus.emit(NodeStarted(node=node))
    started = time.monotonic()
    report: dict[str, Any] = {"summary": ""}
    try:
        yield report
    finally:
        deps.bus.emit(
            NodeFinished(
                node=node,
                summary=str(report.get("summary", "")),
                seconds=time.monotonic() - started,
            )
        )
