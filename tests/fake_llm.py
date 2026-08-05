"""A scripted model, so the judgment nodes run without a network or a bill.

`okf_loremaster.llm.fake.FakeCompletion` answers a call; this decides *what* to answer.
It reads the same prompt the real model would: which node is asking is told by the
system message, and what it is being asked about is parsed back out of the user
message. A test therefore writes a policy — "include everything about alpha; the delta
topic is missing its rescue cohort" — rather than a transcript of replies in order,
which would break the moment concurrency reordered two calls.

Parsing the prompt rather than being handed the subject is the point. A change to
`prompts.py` that dropped the topic slug, the offered PMIDs or the paper text out of a
call fails here, loudly, instead of passing against a scripted reply that no longer
corresponds to anything the node sent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from okf_loremaster.llm.fake import FakeCompletion
from okf_loremaster.prompts import (
    CURATE_SYSTEM,
    EXTRACT_SYSTEM,
    QUERY_PLAN_SYSTEM,
    SCREEN_SYSTEM,
    charter_system,
)

# The charter prompt varies with `--max-topics`, so it is recognized by its opening
# line. Sliced from the real thing rather than retyped, so it cannot go stale.
_CHARTER_MARKER = charter_system(1).split("\n", 1)[0]

# The four shapes this reads back out of a prompt, each written by `prompts.py`.
_PAPER = re.compile(r"\n\nPaper:\n\n(.*)$", re.DOTALL)
_SOURCE = re.compile(r"\n\nSource:\n\n(.*)$", re.DOTALL)
_TOPIC = re.compile(r"^Topic: (\S+)$", re.MULTILINE)
_OFFERED = re.compile(r"^ +- (\d+) \[relevance (\d)\]", re.MULTILINE)

# What the default extractor reads back out of a paper. Written against the shape a
# result sentence takes rather than against one fixture's wording, so this is a small
# extractor and not a lookup keyed on a string somebody typed twice. `[^.\n]` keeps the
# quote inside its own sentence and off the section heading above it.
_FINDING = re.compile(
    r"(?P<quote>[^.\n]*?association was (?P<effect>\d+\.\d+) "
    r"\(95% CI (?P<low>\d+\.\d+)-(?P<high>\d+\.\d+); p = (?P<p>[\d.]+)\)\.)"
)
_COHORT = re.compile(r"cohort of (?P<n>\d+) adults")
_CODE = re.compile(r"ICD-10 (?P<code>[A-Z]\d+\.\d+)")

# A paper's title and abstract -> the fields of one `ScreenVerdict`.
ScreenFn = Callable[[str], dict[str, Any]]
# A topic slug and the PMIDs offered to it -> the fields of one `TopicCuration`.
CurateFn = Callable[[str, list[str]], dict[str, Any]]
# The source text a paper was read from -> the fields of one `Extraction`.
ExtractFn = Callable[[str], dict[str, Any]]


def verdict(
    *,
    include: bool,
    relevance: int,
    topic: str = "",
    reason: str = "screened",
    confidence: str = "medium",
) -> dict[str, Any]:
    """One screening reply. No `pmid`: the node injects the one it already holds."""
    return {
        "include": include,
        "relevance": relevance,
        "topic": topic,
        "reason": reason,
        "confidence": confidence,
    }


def curation(decisions: dict[str, bool], *, missing: str = "") -> dict[str, Any]:
    """One curation reply, from `{pmid: keep}`."""
    return {
        "decisions": [
            {"pmid": pmid, "keep": keep, "rationale": "kept" if keep else "dropped"}
            for pmid, keep in decisions.items()
        ],
        "missing": missing,
    }


def row(**fields: Any) -> dict[str, Any]:
    """One `PredictorRow`, with everything a numeric test does not care about filled in."""
    base: dict[str, Any] = {
        "predictor": "exposure",
        "operationalization": "recorded at baseline",
        "timing": "before the outcome window",
        "outcome": "the measured outcome",
        "evidence_type": "observational_association",
        "direction": "increases",
        "confidence": "high",
    }
    return {**base, **fields}


def extraction(
    *,
    predictors: Sequence[dict[str, Any]] = (),
    null_findings: Sequence[dict[str, Any]] = (),
    n: int | None = None,
    vocabulary_hints: list[dict[str, Any]] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """One extraction reply. The prose is boilerplate; the numbers are the subject."""
    body: dict[str, Any] = {
        "description": "A cohort study of one exposure and one measured outcome.",
        "bottom_line": "The exposure was associated with the outcome.",
        "study_design": "cohort study",
        # The normalized category beside the paper's own words, so an end-to-end run
        # scores evidence strength on a real design rather than on `unclear` — which is
        # the one value that exercises none of the scoring.
        "design": "retrospective_cohort",
        "n": n,
        "population": "adults",
        "outcome_definition": "the outcome as this study measured it",
        "predictors": list(predictors),
        "null_findings": list(null_findings),
        "vocabulary_hints": list(vocabulary_hints or []),
        "caveats": "Observational; residual confounding is likely.",
        "tags": ["exposure", "outcome"],
    }
    return {**body, **fields}


def supported(source: str) -> dict[str, Any]:
    """The default extractor: every number it records is one it read in the source.

    It finds them the way a model would, by reading the paper it was handed. That is
    what makes a fabricating extractor a contrast with a working one rather than the
    only case the suite ever exercises — and it means an ordinary run through the graph
    ends with verification finding nothing, which is the result to be suspicious of if
    it ever changes.

    A paper printing no numbers still gets a row. A relationship reported without a
    magnitude is a relationship reported, and an extractor that dropped those would make
    every abstract-only paper look like an empty one.
    """
    finding = _FINDING.search(source)
    cohort = _COHORT.search(source)
    numbers: dict[str, Any] = {}
    if finding is not None:
        numbers = {
            "effect": float(finding["effect"]),
            "effect_measure": "adjusted OR",
            "effect_raw": f"{finding['effect']} (95% CI {finding['low']}-{finding['high']})",
            "ci_low": float(finding["low"]),
            "ci_high": float(finding["high"]),
            "p_value": finding["p"],
            # The corpus prints "adjusted OR", so a reader that noticed would say so.
            # Left off the no-numbers path deliberately: a row with no magnitude has
            # nothing to have been adjusted, and claiming otherwise would mean the suite
            # never exercised an unmeasured adjustment.
            "adjusted": True,
            "quote": finding["quote"],
        }
    # The code is read off the page like everything else. A hint whose concept is named
    # but whose `codes` is empty is the normal case, not a failure, so a paper that
    # prints no code still gets an entry.
    code = _CODE.search(source)
    codes = [{"system": "icd10", "code": code["code"]}] if code is not None else []
    return extraction(
        n=int(cohort["n"]) if cohort is not None else None,
        predictors=[row(**numbers)],
        vocabulary_hints=[{"concept": "the exposure", "codes": codes}],
    )


@dataclass
class ScriptedLLM:
    """Answers the judgment nodes from rules, and records what it was asked.

    The records are what most assertions are actually about — that no paper was screened
    twice across two rounds, that a second round re-curated only the topic that came up
    short — none of which is visible in the run state afterward.
    """

    screen: ScreenFn
    curate: CurateFn
    # The `QueryPlan` a planning call returns, as raw `PlannedQuery` dicts.
    plan: Sequence[dict[str, str]] = ()
    # Defaulted because most tests are not about extraction and would otherwise all have
    # to say so. The default is a faithful reading, so a run that reaches `extract`
    # produces a bundle rather than a topic of dropped papers.
    extract: ExtractFn = supported

    screened: list[str] = field(default_factory=list)
    curated: list[str] = field(default_factory=list)
    extracted: list[str] = field(default_factory=list)
    # Topic slug to the PMIDs offered on each call for it, one list per call.
    offers: dict[str, list[list[str]]] = field(default_factory=dict)
    plans: int = 0

    def completion(self) -> FakeCompletion:
        return FakeCompletion(replies=self._reply)

    def calls_for(self, topic: str) -> int:
        return self.curated.count(topic)

    # --- dispatch ----------------------------------------------------------

    def _reply(self, request: dict[str, Any]) -> str:
        messages = list(request.get("messages") or [])
        system = str(messages[0].get("content", "")) if messages else ""
        user = str(messages[-1].get("content", "")) if messages else ""

        if system == SCREEN_SYSTEM:
            return self._screen(user)
        if system == CURATE_SYSTEM:
            return self._curate(user)
        if system == EXTRACT_SYSTEM:
            return self._extract(messages)
        if system == QUERY_PLAN_SYSTEM:
            self.plans += 1
            return json.dumps({"queries": list(self.plan)})
        # Matched on a stable fragment rather than the whole string: the charter prompt
        # is now assembled per run around `--max-topics`, so no single value of it is
        # the prompt.
        if _CHARTER_MARKER in system:
            raise AssertionError(
                "a charter was drafted; these tests supply one so the taxonomy is fixed"
            )
        raise AssertionError(f"unrecognized system prompt: {system[:80]!r}")

    def _screen(self, user: str) -> str:
        match = _PAPER.search(user)
        if match is None:
            raise AssertionError("screen_user no longer puts the paper under a `Paper:` header")
        paper = match.group(1).strip()
        self.screened.append(paper)
        return json.dumps(self.screen(paper))

    def _extract(self, messages: list[dict[str, Any]]) -> str:
        """The paper comes from the *first* user message, not the last.

        A schema repair appends the failed reply and a hint, so by the second attempt
        the last message no longer carries the source. Reading the first keeps a repair
        round answerable, and keeps `extracted` a list of papers rather than of hints.
        """
        for message in messages:
            if message.get("role") != "user":
                continue
            match = _SOURCE.search(str(message.get("content", "")))
            if match is not None:
                source = match.group(1).strip()
                self.extracted.append(source)
                return json.dumps(self.extract(source))
        raise AssertionError("extract_user no longer puts the paper under a `Source:` header")

    def _curate(self, user: str) -> str:
        topic = _TOPIC.search(user)
        if topic is None:
            raise AssertionError("curate_user no longer names the topic it is asking about")
        offered = [match.group(1) for match in _OFFERED.finditer(user)]
        if not offered:
            raise AssertionError("curate_user offered no papers; the node called for nothing")
        slug = topic.group(1)
        self.curated.append(slug)
        self.offers.setdefault(slug, []).append(offered)
        return json.dumps(self.curate(slug, offered))
