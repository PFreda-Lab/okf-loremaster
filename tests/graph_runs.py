"""One finished build against the synthetic corpus, for tests that need a whole run.

Three test modules want the same thing — the graph driven end to end over `fake_ncbi`
with a scripted model — and want to look at different parts of the result. Screening
asks what was retrieved and re-queried; extraction and verification ask what was read
and what survived checking. Sharing the runner keeps them asking about the same run
rather than three that have drifted apart.

The charter and the scripted policy live here for the same reason. They describe the
corpus, not any one test: four topics named after the four topics, a screener that
recognizes a paper from the text it was shown, and a curator that asks for the withheld
slice by name until it has been given one.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from okf_loremaster.clients import build_clients
from okf_loremaster.events import EventBus
from okf_loremaster.graph.state import Deps, RunState
from okf_loremaster.llm.router import Router
from okf_loremaster.run import RunOptions, build_run
from okf_loremaster.schemas import Charter, ConceptRecord, Topic

from fake_llm import ScriptedLLM, curation, verdict
from fake_ncbi import TOPICS, UNLOCK_PHRASE, FakeNCBI, identify, is_rescue, rescue_pmids

PROMPT = "identify predictors of a measured outcome after a procedure in adults"

# Large enough that every candidate is pooled. A pool that had to choose would put the
# ranker between the curator's gap and the papers that fill it.
POOL_SIZE = 200

# The corpus, after dedupe drops one retracted paper, two with no abstract, and one
# duplicate title — all of them in the first topic.
UNIQUE_FIRST_ROUND = 156

TOPIC_MIN, TOPIC_MAX, TARGET = 4, 8, 24

# What each of the three well-served topics keeps on the first round. Equal to the
# ceiling, so the bundle-wide target is what decides the final sizes rather than the
# curator's generosity.
KEEP_PER_TOPIC = TOPIC_MAX

# The two ordinary `delta` papers the screener lets through. Enough that the topic is
# curated at all — a topic offered nothing is never asked, and never reports a gap.
DELTA_INCLUDED = 2


def charter_for(**overrides: Any) -> Charter:
    fields: dict[str, Any] = dict(
        prompt=PROMPT,
        task=PROMPT,
        population="adults",
        outcome="measured outcome",
        topic_taxonomy=[
            Topic(slug=topic, title=topic.title(), scope=f"the {topic} facet", seed_terms=[topic])
            for topic in TOPICS
        ],
        vocabularies=["icd10"],
        target_papers=TARGET,
        topic_min=TOPIC_MIN,
        topic_max=TOPIC_MAX,
    )
    return Charter(**{**fields, **overrides})


def scripted_run(**overrides: Any) -> ScriptedLLM:
    """A screener and a curator with a policy, not a transcript.

    The screener recognizes a paper the way the real one has to: from the text it was
    shown. The curator keeps what it is offered and, until it has seen a rescue paper,
    says in as many words that its topic lacks them — which is the only route by which
    `UNLOCK_PHRASE` can reach a query.
    """
    rescue = set(rescue_pmids())

    def screen(text: str) -> dict[str, Any]:
        topic, n = identify(text)
        if topic != "delta":
            return verdict(include=True, relevance=3, topic=topic, reason="on point")
        if is_rescue(n):
            return verdict(include=True, relevance=3, topic="delta", reason="the rescue cohort")
        if n < DELTA_INCLUDED:
            return verdict(include=True, relevance=2, topic="delta", reason="thin but usable")
        return verdict(include=False, relevance=1, topic="delta", reason="not this review")

    def curate(slug: str, offered: list[str]) -> dict[str, Any]:
        if slug != "delta":
            return curation({p: index < KEEP_PER_TOPIC for index, p in enumerate(offered)})
        found = rescue.intersection(offered)
        return curation(dict.fromkeys(offered, True), missing="" if found else UNLOCK_PHRASE)

    fields: dict[str, Any] = dict(
        screen=screen,
        curate=curate,
        plan=[
            {
                "term": '"measured outcome"[tiab] AND "adults"[tiab]',
                "rationale": "the request as a whole",
                "topic": "",
            },
            *(
                {
                    "term": f'"measured outcome"[tiab] AND "{topic}"[tiab]',
                    "rationale": f"seeds the {topic} topic",
                    "topic": topic,
                }
                for topic in TOPICS
            ),
        ],
    )
    return ScriptedLLM(**{**fields, **overrides})


class Run:
    """One finished build, and everything a test needs to look at afterward."""

    def __init__(self, state: RunState, scripted: ScriptedLLM, fake: FakeNCBI) -> None:
        self.state = state
        self.scripted = scripted
        self.fake = fake

    @property
    def topics(self) -> dict[str, list[str]]:
        return dict(self.state["topics"])

    @property
    def gaps(self) -> list[str]:
        curation_result = self.state["curation"]
        assert curation_result is not None
        return [gap.topic for gap in curation_result.gaps if gap.shortfall > 0]

    @property
    def records(self) -> list[ConceptRecord]:
        return list(self.state["records"])

    def record(self, pmid: str) -> ConceptRecord:
        for record in self.records:
            if record.pmid == pmid:
                return record
        raise AssertionError(f"no record for {pmid}")

    @property
    def warnings(self) -> list[str]:
        return list(self.state["warnings"])

    def calls_for(self, topic: str) -> int:
        return self.scripted.curated.count(topic)


@asynccontextmanager
async def node_deps(
    settings_factory: Any,
    tmp_path: Path,
    *,
    scripted: ScriptedLLM | None = None,
    fake: FakeNCBI | None = None,
    **overrides: Any,
) -> AsyncIterator[Deps]:
    """Deps wired to a scripted model, or to none at all when `scripted` is omitted.

    For the tests that drive one node rather than a whole run. `fake` is passed in when
    the test wants to look at what the transport was asked for afterward.
    """
    settings = run_settings(settings_factory, tmp_path)
    bus = EventBus()
    clients = build_clients(
        settings, bus=bus, transport=(fake or FakeNCBI()).transport()
    )
    router = (
        None if scripted is None else Router(settings, bus, completion_fn=scripted.completion())
    )
    try:
        yield Deps(settings=settings, bus=bus, clients=clients, router=router, **overrides)
    finally:
        await clients.aclose()
        bus.close()


def run_settings(settings_factory: Any, tmp_path: Path) -> Any:
    settings = settings_factory(
        ncbi_email="test@example.org",
        # Only for the rate it selects. These runs make a few hundred requests against
        # an in-process transport, and the courtesy limiter is not what is under test.
        ncbi_api_key="not-a-real-key",
        http_cache_enabled=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        model_fast="fake/fast",
        model_balanced="fake/mid",
        model_reasoning="fake/deep",
        api_key="not-a-real-key",
    )
    return settings


async def full_run(
    settings_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scripted: ScriptedLLM | None = None,
    charter: Charter | None = None,
    **overrides: Any,
) -> Run:
    model = scripted if scripted is not None else scripted_run()
    fake = FakeNCBI()

    def router(settings: Any, bus: Any) -> Router:
        return Router(settings, bus, completion_fn=model.completion())

    monkeypatch.setattr("okf_loremaster.llm.router.Router", router)

    charter_path = tmp_path / "given.yaml"
    charter_path.write_text((charter or charter_for()).to_yaml(), encoding="utf-8")

    options = RunOptions(
        prompt=PROMPT,
        charter_path=charter_path,
        out=tmp_path / "run",
        pool_size=POOL_SIZE,
        target_papers=TARGET,
        topic_min=TOPIC_MIN,
        topic_max=TOPIC_MAX,
        **overrides,
    )
    state, _ = await build_run(
        options,
        console=Console(file=io.StringIO(), width=160, no_color=True),
        settings=run_settings(settings_factory, tmp_path),
        transport=fake.transport(),
    )
    return Run(state, model, fake)
