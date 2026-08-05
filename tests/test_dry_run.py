"""The whole of `charter -> search -> dedupe -> rank`, offline, with no model.

This is the build step 4 gate as an executable test. `--dry-run` has to plan real
queries, report real hit counts, project spend, print both confirmation surfaces, and
make exactly zero model calls — and the last one is the claim easiest to break by
accident and hardest to notice, since a run that quietly spent would look identical
except on an invoice.

So it is enforced twice. `Router` is replaced with something that raises on
construction, and the event stream is checked afterward for `LLMCall`. The first says
nothing tried; the second says nothing succeeded by another route.

The corpus is `fake_ncbi`: real E-utilities JSON and real PubMed XML over a mock
transport, so every client parser runs, with a shape chosen to make the effect of MMR
and the per-topic quota measurable rather than merely present.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from okf_loremaster.events import Event, EventBus, LLMCall
from okf_loremaster.graph.state import RunState
from okf_loremaster.ranking import topic_affinity
from okf_loremaster.run import RunOptions, build_run
from okf_loremaster.schemas import Charter, Topic
from okf_loremaster.ui.pauses import TOP_TITLES, ConsolePause

from fake_ncbi import TOPICS, FakeNCBI, all_pmids
from graph_runs import full_run

PROMPT = "identify predictors of a measured outcome after a procedure in adults"

# Small enough that the quota has to choose. The corpus is 160 papers across four
# topics; at the default 800 every paper is retained either way and there is nothing
# to compare.
POOL_SIZE = 40


def charter_for(topics: tuple[str, ...] = TOPICS) -> Charter:
    """A charter whose topics map one-to-one onto the fake corpus's topics.

    Supplied rather than drafted, because a dry run with no charter gets the skeleton —
    no taxonomy, so no topic affinity, so nothing for the quota to do. `--charter` is
    how a dry run gets a real plan, and the charter node warns when it has to fall back.
    """
    return Charter(
        prompt=PROMPT,
        task=PROMPT,
        population="adults",
        outcome="measured outcome",
        topic_taxonomy=[
            Topic(slug=topic, title=topic.title(), scope=f"the {topic} facet", seed_terms=[topic])
            for topic in topics
        ],
    )


class Harness:
    """One dry run, plus everything a test needs to look at afterward."""

    def __init__(self, state: RunState, directory: Path, output: str, events: list[Event]) -> None:
        self.state = state
        self.directory = directory
        self.output = output
        self.events = events

    @property
    def comparison(self) -> Any:
        return self.state.get("comparison")

    def charter_yaml(self) -> Charter:
        return Charter.from_yaml((self.directory / "charter.yaml").read_text(encoding="utf-8"))


async def dry_run(
    settings_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    charter: Charter | None = None,
    fake: FakeNCBI | None = None,
    settings_overrides: dict[str, Any] | None = None,
    **overrides: Any,
) -> Harness:
    """Run the graph against the fake corpus and capture what a user would have seen."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run constructed a Router — that is a model call waiting")

    monkeypatch.setattr("okf_loremaster.llm.router.Router", refuse)

    # Every event, alongside whatever the renderer made of them. Subscribing before the
    # run starts is the only way to see RunStarted.
    seen: list[Event] = []
    original = EventBus.emit

    def record(self: EventBus, event: Event) -> None:
        seen.append(event)
        original(self, event)

    monkeypatch.setattr(EventBus, "emit", record)

    charter_path: Path | None = None
    if charter is not None:
        charter_path = tmp_path / "given.yaml"
        charter_path.write_text(charter.to_yaml(), encoding="utf-8")

    buffer = io.StringIO()
    # A fixed width, because the retrieve pause's tables are what several assertions
    # read and a terminal-dependent wrap would make them terminal-dependent too.
    console = Console(file=buffer, width=160, force_terminal=False, no_color=True)

    options = RunOptions(
        prompt=PROMPT,
        charter_path=charter_path,
        out=tmp_path / "run",
        pool_size=POOL_SIZE,
        target_papers=120,
        dry_run=True,
        **overrides,
    )
    settings = settings_factory(
        ncbi_email="test@example.org",
        http_cache_enabled=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        **(settings_overrides or {}),
    )
    state, directory = await build_run(
        options,
        console=console,
        settings=settings,
        transport=(fake or FakeNCBI()).transport(),
    )
    return Harness(state, directory, buffer.getvalue(), seen)


# --- zero LLM calls ---------------------------------------------------------


async def test_a_dry_run_makes_no_model_calls(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())
    assert not [event for event in run.events if isinstance(event, LLMCall)]
    assert run.state["pool"]  # and it still produced a pool


async def test_a_dry_run_needs_no_model_configured(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`require_llm()` is skipped, so an unconfigured machine can still plan a run."""
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())
    assert run.state["charter"] is not None


async def test_without_a_charter_a_dry_run_says_what_it_cannot_do(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skeleton path: real queries from the prompt alone, and a warning saying so."""
    run = await dry_run(settings_factory, tmp_path, monkeypatch)
    charter = run.state["charter"]
    assert charter is not None
    assert charter.topic_taxonomy == []
    assert any("no topic taxonomy" in warning for warning in run.state["warnings"])
    assert run.state["executed"]  # it searched anyway


# --- the queries ------------------------------------------------------------


async def test_the_plan_covers_every_topic_and_reports_hit_counts(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeNCBI()
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for(), fake=fake)

    plan = run.state["plan"]
    assert plan is not None
    assert {q.topic for q in plan.queries} == {"", *TOPICS}
    assert len(fake.esearch_terms) == len(plan.queries)

    executed = run.state["executed"]
    assert [q.count for q in executed] == [len(all_pmids()), 40, 40, 40, 40]
    assert not any(q.suspect for q in executed)


async def test_one_efetch_covers_the_whole_plan(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paper found by four queries is fetched once, carrying all four provenances."""
    fake = FakeNCBI()
    await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for(), fake=fake)

    assert len(fake.efetch_batches) == 1
    assert sorted(fake.efetch_batches[0]) == sorted(all_pmids())
    assert len(fake.icite_batches) == 1


# --- dedupe -----------------------------------------------------------------


async def test_dedupe_counts_every_kind_of_drop(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    assert run.state["dropped"] == {"retracted": 1, "no abstract": 2, "duplicate title": 1}
    assert len(run.state["unique"]) == len(run.state["candidates"]) - 4


# --- MMR and the quota ------------------------------------------------------


async def test_mmr_and_the_quota_change_the_retained_set(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 4 gate's last clause, as a measurement.

    The fake corpus is stacked by citation impact, so pure relevance rank leaves the
    pool lopsided across the four topics. The quota levels it, and the topics it helps
    are exactly the ones pure rank left short of their share.
    """
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())
    comparison = run.comparison
    assert comparison is not None

    quota = POOL_SIZE // len(TOPICS)
    pure = comparison.pure_by_topic
    assert max(pure.values()) >= 3 * min(pure.values())  # lopsided before
    assert set(comparison.diversified_by_topic.values()) == {quota}  # level after
    assert comparison.changed > 0
    assert comparison.topics_helped == sorted(s for s, n in pure.items() if n < quota)


async def test_the_pool_is_capped_and_topic_affinity_is_total(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    pool = run.state["pool"]
    assert len(pool) == POOL_SIZE
    query_topic = run.state["query_topic"]
    # Every paper was found by a topic-targeted query as well as the base one, so none
    # of them falls into the unassigned group.
    assert all(topic_affinity(item.candidate, query_topic) for item in pool)


async def test_the_run_is_reproducible(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same corpus, same charter, same pool — in the same order."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    first = await dry_run(settings_factory, tmp_path / "a", monkeypatch, charter=charter_for())
    second = await dry_run(settings_factory, tmp_path / "b", monkeypatch, charter=charter_for())
    assert [i.pmid for i in first.state["pool"]] == [i.pmid for i in second.state["pool"]]


# --- what the pauses print --------------------------------------------------


async def test_the_charter_pause_prints_the_taxonomy(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The taxonomy is what the pause exists for: it is decided before any paper is
    read, it governs every later node, and it is cheap to fix here and expensive
    afterward."""
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    assert "topic_taxonomy" in run.output
    for topic in TOPICS:
        assert topic in run.output


async def test_the_retrieve_pause_prints_the_totals_and_the_top_titles(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    unique = len(run.state["unique"])
    assert f"{unique:,}" in run.output
    assert f"top {TOP_TITLES} of the pool" in run.output
    # The titles themselves, not just a count of them.
    printed = [item for item in run.state["pool"][:TOP_TITLES] if item.pmid in run.output]
    assert len(printed) == TOP_TITLES


async def test_the_retrieve_pause_prints_the_diversification_comparison(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    assert "pool by topic affinity" in run.output
    assert "MMR + quota" in run.output
    assert "only because of MMR and the per-topic quota" in run.output


async def test_a_dry_run_projects_spend_from_the_real_pool(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    assert "projected spend" in run.output
    # Screening is projected from the abstracts actually retrieved, which is what makes
    # this a measurement rather than a guess.
    assert "measured from the retrieved pool" in run.output
    # A supplied charter costs nothing, and the projection says so instead of billing
    # for a call that will not happen.
    assert "charter supplied" in run.output


async def test_an_unpriced_projection_says_so_rather_than_reporting_nothing_to_pay(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No model bound to a role means no price for it — which is not the same as free."""
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    assert "cost unavailable" in run.output
    assert "unpriced (tokens only)" in run.output
    assert "OKF_LOREMASTER_PRICE_" in run.output


async def test_a_price_override_turns_the_projection_into_a_figure(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch the unpriced note points at, exercised end to end."""
    run = await dry_run(
        settings_factory,
        tmp_path,
        monkeypatch,
        charter=charter_for(),
        settings_overrides={
            "price_fast_in": 0.8,
            "price_fast_out": 4.0,
            "price_balanced_in": 3.0,
            "price_balanced_out": 15.0,
            "price_reasoning_in": 15.0,
            "price_reasoning_out": 75.0,
        },
    )

    assert "cost unavailable" not in run.output
    assert "unpriced (tokens only)" not in run.output
    assert "$" in run.output


# --- the pauses themselves --------------------------------------------------


def test_autonomous_and_dry_run_print_everything_and_ask_nothing() -> None:
    """An autonomous run bypasses the question, not the information."""
    for options in (RunOptions(), RunOptions(dry_run=True)):
        from okf_loremaster.run import _pause

        pause = _pause(options, None)
        assert isinstance(pause, ConsolePause)
        assert pause._interactive is False


def test_interactive_asks_and_a_dry_run_does_not_take_that_away() -> None:
    """`--interactive` is the only thing that asks, and nothing else overrules it.

    A dry run used to suppress the questions, back when asking was the default and a
    dry run was opting out of it. Now that asking has to be requested, the request is
    honored: rehearsing the decisions on a run that costs nothing is the cheapest place
    to see them, and one rule holds on both the console and the TUI.
    """
    from okf_loremaster.run import _pause

    assert _pause(RunOptions(interactive=True), None)._interactive is True
    assert _pause(RunOptions(interactive=True, dry_run=True), None)._interactive is True


async def test_json_output_never_asks_and_never_prints_a_table() -> None:
    from okf_loremaster.run import _pause

    pause = _pause(RunOptions(json_out=True), None)
    assert not isinstance(pause, ConsolePause)
    assert (await pause.charter(charter_for())).proceed


async def test_the_pauses_are_not_asked_on_a_dry_run(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())
    assert run.output.count("continuing without asking") == 2  # charter, then retrieve


# --- a search that finds nothing --------------------------------------------


async def test_a_run_that_retrieves_no_papers_fails_instead_of_emitting_a_bundle(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every node after search is a well-behaved no-op on an empty list, so a run that
    found nothing used to reach the end and print `complete ... bundle valid` over eight
    empty topics. That is the blank-paper defect again one level up: the tool reporting
    success for work it did not do.

    It happened for real. Query planning failed, the deterministic fallback anchored on
    a four-word `population` as an exact phrase, PubMed matched it zero times, and all
    nine queries came back empty.
    """
    with pytest.raises(RuntimeError, match="no papers retrieved"):
        await full_run(settings_factory, tmp_path, monkeypatch, finds_nothing=True)


async def test_a_dry_run_that_finds_nothing_is_still_a_finished_dry_run(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run exists to report what the queries would retrieve, and "nothing" is a
    finding it is supposed to be able to deliver rather than crash on."""
    run = await dry_run(
        settings_factory,
        tmp_path,
        monkeypatch,
        charter=charter_for(),
        fake=FakeNCBI(finds_nothing=True),
    )

    assert "0" in run.output


# --- --vocab ----------------------------------------------------------------
async def test_the_settled_charter_is_written_even_though_one_was_supplied(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run directory records what the run actually used, not what it was handed."""
    run = await dry_run(settings_factory, tmp_path, monkeypatch, charter=charter_for())

    written = run.charter_yaml()
    assert written.target_papers == 120  # from the flag, not the supplied file
    assert written.generated_by == "okf-loremaster/charter/none"


# --- what the projection prices ---------------------------------------------


def _projection(settings_factory: Any, *, target: int, topics: int = 8, floor: int = 8) -> Any:
    """`project_spend` over an empty pool, so only the taxonomy decides the counts.

    No candidates: with none retrieved there is nothing to screen, and extraction falls
    back to what curation is expected to keep — which is the number under test.
    """
    from okf_loremaster.llm.estimate import project_spend

    charter = charter_for(tuple(f"topic-{i}" for i in range(topics))).model_copy(
        update={"target_papers": target, "topic_paper_min": floor}
    )
    return project_spend(
        charter,
        settings=settings_factory(model_balanced="m", model_fast="m", model_reasoning="m"),
        pool=[],
        screen_budget=400,
        target_papers=target,
    )


def test_the_projection_prices_the_papers_curation_will_keep_not_the_target(
    settings_factory: Any,
) -> None:
    """Measured against a real run, and this is the whole gap.

    `--target-papers 10` over 8 topics with `topic_paper_min 8` kept 62 papers and extracted
    61, because trimming stops once no topic is above its floor. The projection priced
    extraction — the dearest node per call — at 10, and came in at $1.01 against $5.04
    actually spent. An estimate a human is shown just before deciding whether to pay it
    is worth more than a fifth of the truth.
    """
    estimate = _projection(settings_factory, target=10)
    extract = next(node for node in estimate.nodes if node.node == "extract")

    assert extract.calls == 64, "8 topics x topic_paper_min 8 is the floor a target cannot undercut"
    assert any("floor of 64" in note for note in estimate.notes), (
        "the table shows a number above the requested target, so it has to say why"
    )


def test_a_target_above_the_floor_is_still_the_target(settings_factory: Any) -> None:
    """The floor is a lower bound, not a replacement. A bundle asked for 200 papers over
    8 topics is priced at 200, and nothing needs explaining in the notes."""
    estimate = _projection(settings_factory, target=200)
    extract = next(node for node in estimate.nodes if node.node == "extract")

    assert extract.calls == 200
    assert not any("floor of" in note for note in estimate.notes)
