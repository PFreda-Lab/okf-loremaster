"""Assembling the graph, and driving it through its two confirmation pauses.

`charter -> search -> dedupe -> rank -> screen -> curate -> fulltext -> extract ->
reconcile -> review -> emit_okf -> validate -> index_vectors`, with a conditional edge
from `curate` back to `search`. Three things here are worth more than the wiring:

**The pauses are interrupts, not prompts inside a node.** The graph is compiled with
`interrupt_after=["charter", "rank"]` and a SQLite checkpointer, so the orchestrator
runs to the interrupt, hands the checkpointed state to a `Pause`, and resumes with
`ainvoke(None, config)`. Nodes stay printless and the run stays resumable — the
checkpoint is written whether or not anyone answers, which is what makes `--resume
<run-id>` fall out for free rather than needing its own machinery.

**The re-query loop is bounded in three independent ways**, because a retry edge that
can ask the same question twice is a bill with no ceiling: `rounds` against
`max_rounds`, and a gap plan that must contain a query no round has already run. Either
one alone would do; both are cheap. `_drive` then resumes until the graph is finished
rather than merely paused, since a second round passes back through `rank` and `rank`
is an interrupt point. The retrieve pause is *not* asked again on that pass —
`screen_budget` is global across rounds, so the number approved once bounds the whole
run.

**The serializer allowlist is not optional.** LangGraph's msgpack path warns on every
unregistered type it deserializes and has announced it will refuse them, so every
schema that travels in state is named explicitly. `from_conn_string` takes no
serializer, so it is assigned to the saver afterward.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from okf_loremaster.config import Settings
from okf_loremaster.events import ErrorEvent, RunFinished, RunStarted
from okf_loremaster.graph.nodes import (
    charter_node,
    curate_node,
    dedupe_node,
    emit_okf_node,
    extract_node,
    fulltext_node,
    index_vectors_node,
    pending_gap_plan,
    rank_node,
    reconcile_node,
    review_node,
    screen_node,
    search_node,
    validate_node,
)
from okf_loremaster.graph.state import Deps, RunState, initial_state, screen_budget_spent
from okf_loremaster.llm.estimate import SpendEstimate, project_spend
from okf_loremaster.ranking import SelectionComparison
from okf_loremaster.schemas import (
    BundleCounts,
    Candidate,
    Charter,
    ConceptRecord,
    Confidence,
    CostSummary,
    CurationDecision,
    CurationResult,
    Direction,
    EvidenceType,
    ExecutedQuery,
    Extraction,
    NullFinding,
    PaperText,
    PlannedQuery,
    PredictorRow,
    QueryPlan,
    RunManifest,
    ScoredCandidate,
    ScreenVerdict,
    SourceRef,
    TextBasis,
    Topic,
    TopicGap,
    TopicSummary,
    Verification,
    VerificationSummary,
)
from okf_loremaster.ui.pauses import AutoApprove, Pause

__all__ = ["NODES", "PAUSE_AFTER", "build_graph", "checkpointer", "run_build"]

# The nodes after which the graph stops for confirmation. Both are the last cheap
# moment before an expensive one.
PAUSE_AFTER = ("charter", "rank")

# The pipeline in order, for a UI that wants to show what has not happened yet. Declared
# rather than read off the compiled graph because `StateGraph.nodes` is a dict whose
# order is insertion order, not topology, and because a renderer should not have to build
# a graph to draw a list. `test_tui.py` asserts the two agree.
NODES = (
    "charter",
    "search",
    "dedupe",
    "rank",
    "screen",
    "curate",
    "fulltext",
    "extract",
    "reconcile",
    "review",
    "emit_okf",
    "validate",
    "index_vectors",
)

NodeFn = Callable[[RunState, Deps], Awaitable[dict[str, Any]]]


class BoundNode(Protocol):
    """A node with its deps already closed over.

    A callback protocol rather than a `Callable[...]`, because LangGraph types a node as
    a protocol whose parameter is *named* `state` and a bare `Callable` carries no
    parameter names — mypy then infers the state type as `Never` and rejects every
    `add_node`.
    """

    async def __call__(self, state: RunState) -> dict[str, Any]: ...

# Every schema that can appear in checkpointed state. Left incomplete, deserializing a
# resumed run warns now and fails later.
#
# The enums belong here as much as the models do, and are easy to forget because leaving
# them out looks harmless: they are `StrEnum`, so a blocked value decodes to its own
# string and pydantic revalidates it on the way back into the model. The data survives —
# but every resume logs a page of `Blocked deserialization` at the user, and the day one
# of these stops being a `StrEnum` the silent fallback becomes silent data loss.
CHECKPOINTED_TYPES = (
    BundleCounts,
    Candidate,
    Charter,
    ConceptRecord,
    Confidence,
    CostSummary,
    CurationDecision,
    CurationResult,
    Direction,
    EvidenceType,
    ExecutedQuery,
    Extraction,
    NullFinding,
    PaperText,
    PlannedQuery,
    PredictorRow,
    QueryPlan,
    RunManifest,
    ScoredCandidate,
    ScreenVerdict,
    SelectionComparison,
    Topic,
    TopicGap,
    TopicSummary,
    SourceRef,
    TextBasis,
    Verification,
    VerificationSummary,
)


def _bind(fn: NodeFn, deps: Deps) -> BoundNode:
    """Close a node over its deps, leaving LangGraph a plain `(state) -> update`."""

    async def node(state: RunState) -> dict[str, Any]:
        return await fn(state, deps)

    node.__name__ = fn.__name__
    return node


def build_graph(deps: Deps) -> StateGraph[RunState, Any, RunState, RunState]:
    graph: StateGraph[RunState, Any, RunState, RunState] = StateGraph(RunState)
    graph.add_node("charter", _bind(charter_node, deps))
    graph.add_node("search", _bind(search_node, deps))
    graph.add_node("dedupe", _bind(dedupe_node, deps))
    graph.add_node("rank", _bind(rank_node, deps))
    graph.add_node("screen", _bind(screen_node, deps))
    graph.add_node("curate", _bind(curate_node, deps))
    graph.add_node("fulltext", _bind(fulltext_node, deps))
    graph.add_node("extract", _bind(extract_node, deps))
    graph.add_node("reconcile", _bind(reconcile_node, deps))
    graph.add_node("review", _bind(review_node, deps))
    graph.add_node("emit_okf", _bind(emit_okf_node, deps))
    graph.add_node("validate", _bind(validate_node, deps))
    graph.add_node("index_vectors", _bind(index_vectors_node, deps))

    graph.add_edge(START, "charter")
    graph.add_edge("charter", "search")
    graph.add_edge("search", "dedupe")
    graph.add_edge("dedupe", "rank")
    graph.add_conditional_edges("rank", _after_rank, {"screen": "screen", END: END})
    graph.add_edge("screen", "curate")
    graph.add_conditional_edges(
        "curate", _after_curate(deps), {"search": "search", "fulltext": "fulltext"}
    )
    graph.add_edge("fulltext", "extract")
    graph.add_edge("extract", "reconcile")
    # `review` is unconditional and a no-op without `--review`. Routing around it would
    # make the wiring depend on a dep, and the node already answers "nobody was asked"
    # by returning an empty update.
    graph.add_edge("reconcile", "review")
    graph.add_edge("review", "emit_okf")
    # Validation reads the bundle back off disk, so it can only run once it exists.
    graph.add_edge("emit_okf", "validate")
    # The index is derived from the same files the validator judged, and is built even
    # when the bundle failed the gate: those errors are things a person fixes in a file,
    # not a reason to withhold a rebuildable artifact. Unconditional and a no-op without
    # the finalize choice, for the same reason `review` is.
    graph.add_edge("validate", "index_vectors")
    graph.add_edge("index_vectors", END)
    return graph


def _after_rank(state: RunState) -> str:
    """A dry run stops at the pool.

    In the graph rather than in the driver, and rather than as a guard inside `screen`:
    "zero LLM calls" is then a property of the wiring that a reader can see, not a
    promise every downstream node has to keep remembering to make.
    """
    return END if state.get("dry_run") else "screen"


def _after_curate(deps: Deps) -> Callable[[RunState], str]:
    """Back to search for another round, or on to reading what was kept.

    Four conditions, all of which must hold for another round: a charter to search with,
    rounds left, screening budget left, and a gap plan carrying at least one query no
    earlier round already ran. Two of them exist because a round that cannot change the
    outcome still costs money — without the gap-plan check a topic that is thin because
    the literature is thin would re-run the same searches and arrive at the same
    shortfall, and without the budget check a round can search and rank and re-curate
    while the screener, which is what turns a candidate into a paper, sits at zero.
    """

    def route(state: RunState) -> str:
        charter = state.get("charter")
        if charter is None or int(state.get("rounds") or 0) >= max(1, deps.max_rounds):
            return "fulltext"
        if screen_budget_spent(state, deps):
            return "fulltext"
        plan = pending_gap_plan(charter, state, deps)
        return "search" if plan is not None and plan.queries else "fulltext"

    return route


def _serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINTED_TYPES))


@asynccontextmanager
async def checkpointer(settings: Settings) -> AsyncIterator[AsyncSqliteSaver]:
    """A SQLite checkpointer under the cache directory.

    Deliberately not beside the bundle: checkpoints are run scratch, they are written
    on every node, and the cache directory is the one place already guaranteed to be
    outside a synced folder.
    """
    path = settings.cache_dir / "checkpoints.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        saver.serde = _serde()
        yield saver


async def run_build(
    *,
    run_id: str,
    prompt: str,
    deps: Deps,
    pause: Pause | None = None,
    charter: Charter | None = None,
    dry_run: bool = False,
    resume: bool = False,
    directory: str = "",
) -> RunState:
    """Drive the graph from start to finish, stopping at each pause.

    Returns whatever state was reached, including when a pause declined: a run stopped
    at the retrieve pause has still produced a real query plan and a real pool, and
    throwing that away because nobody wanted to spend on screening would be perverse.
    """
    gate = pause if pause is not None else AutoApprove()
    deps.bus.emit(RunStarted(run_id=run_id, prompt=prompt, dry_run=dry_run))

    async with checkpointer(deps.settings) as saver:
        compiled = build_graph(deps).compile(
            checkpointer=saver, interrupt_after=list(PAUSE_AFTER)
        )
        config: Any = {"configurable": {"thread_id": run_id}}
        entry: RunState | None = (
            None
            if resume
            else initial_state(
                run_id,
                prompt,
                dry_run=dry_run,
                charter=charter,
                directory=directory,
            )
        )

        try:
            await compiled.ainvoke(entry, config)
            state = await _values(compiled, config)

            drafted = state.get("charter")
            if drafted is not None:
                decision = await gate.charter(drafted)
                if not decision.proceed:
                    return _finish(deps, run_id, state, decision.reason or "stopped at charter")

            await compiled.ainvoke(None, config)
            state = await _values(compiled, config)

            estimate = _estimate(deps, state, charter_supplied=charter is not None)
            decision = await gate.retrieve(state, estimate=estimate)
            if not decision.proceed:
                return _finish(deps, run_id, state, decision.reason or "stopped after ranking")

            state = await _drive(compiled, config, limit=max(1, deps.max_rounds) + 1)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            deps.bus.emit(ErrorEvent(node="graph", message=detail, fatal=True))
            deps.bus.emit(RunFinished(run_id=run_id, ok=False, summary=str(exc)))
            raise

        return _finish(deps, run_id, state, _summary(state), ok=True)


async def _values(compiled: Any, config: Any) -> RunState:
    snapshot = await compiled.aget_state(config)
    values: RunState = snapshot.values
    return values


async def _drive(compiled: Any, config: Any, *, limit: int) -> RunState:
    """Resume until the graph is finished rather than merely paused.

    One `ainvoke(None, ...)` clears one interrupt. The re-query edge sends a run back
    through `rank`, which is an interrupt point, so a single resume would leave it
    parked with a curated topic set nobody ever asked for. `limit` is a backstop against
    a wiring mistake, not the bound that matters — `max_rounds` is, and it is enforced
    in the route function.
    """
    state: RunState = {}
    for _ in range(limit):
        await compiled.ainvoke(None, config)
        snapshot = await compiled.aget_state(config)
        state = snapshot.values
        if not snapshot.next:
            break
    return state


def _summary(state: RunState) -> str:
    pool = len(state.get("pool") or [])
    topics = state.get("topics") or {}
    if not topics:
        return f"{pool} papers pooled"
    kept = sum(len(pmids) for pmids in topics.values())
    filled = sum(1 for pmids in topics.values() if pmids)
    rounds = int(state.get("rounds") or 1)
    detail = f"{kept} papers across {filled} of {len(topics)} topics, from {pool} pooled"
    if rounds > 1:
        detail += f" over {rounds} search rounds"
    records = state.get("records") or []
    if records:
        full = sum(1 for source in (state.get("texts") or {}).values() if source.is_full_text)
        detail += f"; {len(records)} extracted, {full} from full text"
    if state.get("bundle"):
        errors = len(state.get("validation_errors") or [])
        detail += "; bundle valid" if not errors else f"; bundle has {errors} error(s)"
    if state.get("vector_index"):
        detail += f"; {int(state.get('vector_chunks') or 0)} chunks indexed"
    return detail


def _estimate(deps: Deps, state: RunState, *, charter_supplied: bool) -> SpendEstimate | None:
    drafted = state.get("charter")
    if drafted is None:
        return None
    pool = list(state.get("pool") or [])
    return project_spend(
        drafted,
        settings=deps.settings,
        pool=[item.candidate for item in pool],
        screen_budget=deps.screen_budget,
        target_papers=drafted.target_papers,
        charter_was_generated=not charter_supplied,
    )


def _finish(
    deps: Deps, run_id: str, state: RunState, summary: str, *, ok: bool = False
) -> RunState:
    deps.bus.emit(RunFinished(run_id=run_id, ok=ok, summary=summary))
    return state
