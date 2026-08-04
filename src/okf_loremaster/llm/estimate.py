"""Projecting what a run will cost, without making a single model call.

This is what `--dry-run` reports. It is a projection and it says so: the per-call
overheads for nodes that have not run are allowances, and the token counts are a
character heuristic rather than a tokenizer. What is *not* guessed is the corpus. The
pool has already been retrieved by the time this runs, so the abstract lengths, the
paper counts and the share carrying a PMC id are all measured from the real thing.

Pricing follows the same three stages as the router, for the same reason: a projection
that renders unknown as `$0.00` is worse than one that says it does not know.
  1. LiteLLM's price map, consulted by model name.
  2. `OKF_LOREMASTER_PRICE_<ROLE>_IN` / `_OUT`.
  3. Unpriced — tokens only, rendered through `format_cost`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from okf_loremaster.config import ConfigError, Role, Settings
from okf_loremaster.llm.router import format_cost
from okf_loremaster.prompts import (
    CHARTER_SYSTEM,
    CURATE_SYSTEM,
    EXTRACT_SYSTEM,
    QUERY_PLAN_SYSTEM,
    SCREEN_SYSTEM,
    extract_context,
    screen_context,
)
from okf_loremaster.schemas import MAX_SOURCE_CHARS, Candidate, Charter

__all__ = ["NodeEstimate", "SpendEstimate", "estimate_tokens", "project_spend"]

# Four characters per token is the usual English approximation. Good to roughly ±15%,
# which is well inside the uncertainty already carried by the allowances below.
CHARS_PER_TOKEN = 4.0

# Replies, which cannot be measured before they are written. Screening's is capped by
# `graph.nodes.screen.MAX_VERDICT_TOKENS`, so 80 is a typical answer rather than a
# worst case; curation's is a short rationale per paper.
SCREEN_COMPLETION_ALLOWANCE = 80
# Measured, not guessed: a run of 8 topics was cut off at 80 tokens per paper on every
# one of them and finished on all 8 at 160. The truth is in between.
CURATE_PER_PAPER_COMPLETION = 120
# Per offered paper, beyond its title: the PMID, the relevance marker and the
# screener's one-clause reason travel with it.
CURATE_PER_PAPER_OVERHEAD = 25
# The one part of extraction that cannot be measured before it is written. Capped by
# `graph.nodes.extract.MAX_EXTRACTION_TOKENS`; this is a typical full concept —
# `MAX_PREDICTOR_ROWS` rows with quote locators, null findings, and the prose fields —
# not a worst case. 700 was off by most of a factor of three; 2000 was measured against
# replies that pretty-printed their JSON and copied a whole sentence into every row.
# Compact JSON, capped null findings and vocabulary hints, and locators in place of
# copied sentences take about two thirds back off that figure.
EXTRACT_COMPLETION_ALLOWANCE = 700

# A full text runs about this many times the length of its own abstract. Extraction
# reads whichever it got, so the multiple is what separates a cheap run from an
# expensive one.
FULL_TEXT_MULTIPLE = 12.0

# Share of papers carrying a PMC id whose full text is actually retrievable. A PMC id is
# necessary and not sufficient: part of PMC is outside the open-access subset, and BioC
# answers for those with a 200 and an error body.
#
# Two in five was the original guess and it was too pessimistic — one measured run
# retrieved 121 of 141, or 86%. Three in five splits the difference rather than fitting
# the constant to a single corpus, because that corpus was recent and recency correlates
# with an open-access mandate. This is the largest single unknown in the projection: it
# multiplies the length of the input to the most expensive node in the run.
OPEN_ACCESS_RATE = 0.6


def estimate_tokens(text: str) -> int:
    """Approximate token count for a string."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass(frozen=True, slots=True)
class NodeEstimate:
    """Projected spend for one node."""

    node: str
    role: Role
    calls: int
    prompt_tokens: int
    completion_tokens: int
    # None means the model could not be priced, not that it is free.
    usd: float | None
    basis: str = ""

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def format_usd(self) -> str:
        return format_cost(
            self.usd or 0.0,
            calls=self.calls,
            unpriced=0 if self.usd is not None else self.calls,
        )


@dataclass(frozen=True, slots=True)
class SpendEstimate:
    """The whole projection, node by node."""

    nodes: tuple[NodeEstimate, ...] = ()
    # Assumptions a reader should be able to see rather than infer from the total.
    notes: tuple[str, ...] = ()

    @property
    def calls(self) -> int:
        return sum(node.calls for node in self.nodes)

    @property
    def prompt_tokens(self) -> int:
        return sum(node.prompt_tokens for node in self.nodes)

    @property
    def completion_tokens(self) -> int:
        return sum(node.completion_tokens for node in self.nodes)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def usd(self) -> float:
        return sum(node.usd for node in self.nodes if node.usd is not None)

    @property
    def unpriced_calls(self) -> int:
        return sum(node.calls for node in self.nodes if node.usd is None)

    def format_usd(self) -> str:
        """The single rendering path, shared with the live meter and the manifest."""
        return format_cost(self.usd, calls=self.calls, unpriced=self.unpriced_calls)

    def format_tokens(self) -> str:
        return f"{self.tokens:,} tok ({self.prompt_tokens:,} in / {self.completion_tokens:,} out)"


def _price(
    settings: Settings, role: Role, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """USD for a projected token count, or None if the model cannot be priced."""
    from_map = _price_from_litellm(settings, role, prompt_tokens, completion_tokens)
    if from_map is not None:
        return from_map
    price_in, price_out = settings.price_for(role)
    if price_in is None or price_out is None:
        return None
    return (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out


def _price_from_litellm(
    settings: Settings, role: Role, prompt_tokens: int, completion_tokens: int
) -> float | None:
    try:
        model = settings.model_for(role)
    except ConfigError:
        # No model bound to this role. A dry run is still worth reporting in tokens.
        return None
    try:
        # Imported here, not at module scope: litellm costs seconds to import, and a
        # projection is the only reason a dry run needs it. Pricing by model name reads
        # the local price map — it is not a network call.
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total = float(prompt_cost) + float(completion_cost)
    except Exception:
        return None
    # 0.0 is litellm's answer for a model it does not know, not an assertion that the
    # call is free.
    return total if total > 0.0 else None


def project_spend(
    charter: Charter,
    *,
    settings: Settings,
    pool: Sequence[Candidate],
    screen_budget: int,
    target_papers: int,
    charter_was_generated: bool = True,
) -> SpendEstimate:
    """Project the cost of the nodes a full run would still have to pay for.

    `pool` is the ranked candidate pool, already retrieved. Its abstracts are what make
    this a measurement rather than a guess: screening cost is dominated by abstract
    length, and abstract length varies by a factor of five across the literature.
    """
    nodes: list[NodeEstimate] = []
    notes: list[str] = []

    def add(
        node: str,
        role: Role,
        *,
        calls: int,
        prompt_tokens: int,
        completion_tokens: int,
        basis: str,
    ) -> None:
        if calls <= 0:
            return
        nodes.append(
            NodeEstimate(
                node=node,
                role=role,
                calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usd=_price(settings, role, prompt_tokens, completion_tokens),
                basis=basis,
            )
        )

    # --- charter and query planning, both already measurable -----------------

    if charter_was_generated:
        add(
            "charter",
            Role.REASONING,
            calls=1,
            prompt_tokens=estimate_tokens(CHARTER_SYSTEM) + estimate_tokens(charter.prompt),
            completion_tokens=900,
            basis="one call; the prompt is measured, the reply allowed 900 tokens",
        )
        topic_text = " ".join(
            f"{s.slug} {s.scope} {' '.join(s.seed_terms)}" for s in charter.topic_taxonomy
        )
        add(
            "search",
            Role.BALANCED,
            calls=1,
            prompt_tokens=estimate_tokens(QUERY_PLAN_SYSTEM) + estimate_tokens(topic_text),
            completion_tokens=600,
            basis="one query-planning call over the charter's taxonomy",
        )
    else:
        notes.append("charter supplied, so charter and query planning cost nothing")

    # --- screening, the largest line in most runs ----------------------------

    screened = min(screen_budget, len(pool))
    # The same prefix on every screening call, byte for byte. Measured once here for
    # the same reason the node assembles it once: it is most of a short call's prompt.
    context = estimate_tokens(SCREEN_SYSTEM) + estimate_tokens(
        screen_context(
            task=charter.task or charter.prompt,
            population=charter.population,
            outcome=charter.outcome,
            inclusion=list(charter.inclusion),
            exclusion=list(charter.exclusion),
            topics=[(s.slug, s.scope or s.title) for s in charter.topic_taxonomy],
        )
    )
    if screened:
        measured = sum(estimate_tokens(c.screening_text) for c in pool[:screened])
        average = measured // screened
        add(
            "screen",
            Role.FAST,
            calls=screened,
            prompt_tokens=measured + context * screened,
            completion_tokens=SCREEN_COMPLETION_ALLOWANCE * screened,
            basis=(
                f"{screened} papers, {average} tokens of title and abstract each measured "
                f"from the retrieved pool, on a {context}-token shared prefix"
            ),
        )

    # --- curation, one call per topic ----------------------------------------

    topics = max(1, len(charter.topic_taxonomy))
    # `target_papers` is not the number of papers a run keeps. Trimming stops once no
    # topic is above `topic_min` (`curation.py::_apply_target`), so the taxonomy sets a
    # floor the target cannot pull a bundle under — and the floor is what priced a real
    # run. `--target-papers 10` against 8 topics of 8 kept 62 papers and extracted 61,
    # while this projected 10. Extraction is the dearest node per call, so being six
    # times short there put the whole estimate five times under what the run spent:
    # $1.01 projected, $5.04 paid.
    floor = topics * charter.topic_min
    kept = max(target_papers, floor)
    if floor > target_papers:
        notes.append(
            f"priced on the taxonomy's floor of {kept} papers ({topics} topics x "
            f"topic_min {charter.topic_min}), not on target_papers ({target_papers}) — "
            "when the two disagree the topics win and the bundle comes in over target"
        )
    included = min(screened, kept * 2)
    per_topic = max(1, included // topics)
    title_tokens = (
        sum(estimate_tokens(c.title) for c in pool[:screened]) // screened if screened else 15
    )
    per_paper = title_tokens + CURATE_PER_PAPER_OVERHEAD
    add(
        "curate",
        Role.BALANCED,
        calls=topics,
        prompt_tokens=topics * (estimate_tokens(CURATE_SYSTEM) + per_topic * per_paper),
        completion_tokens=topics * per_topic * CURATE_PER_PAPER_COMPLETION,
        basis=f"{topics} topics, about {per_topic} paper(s) each at {per_paper} tokens",
    )

    # --- extraction, the largest line when full text is available ------------

    retained = min(kept, len(pool)) if pool else kept
    if retained:
        sample = pool[:retained] if pool else []
        abstract_tokens = (
            sum(estimate_tokens(c.abstract) for c in sample) // max(1, len(sample))
            if sample
            else 250
        )
        with_pmcid = sum(1 for c in sample if c.may_have_full_text)
        full_text_share = (with_pmcid / len(sample) * OPEN_ACCESS_RATE) if sample else 0.0
        # The truncation `fulltext` applies is part of the price, so it is part of the
        # projection: without the cap a corpus of long reviews projects several times
        # what the run can actually spend.
        capped = min(
            abstract_tokens * FULL_TEXT_MULTIPLE, MAX_SOURCE_CHARS / CHARS_PER_TOKEN
        )
        per_paper = int(abstract_tokens * (1 - full_text_share) + capped * full_text_share)
        # The same per-topic prefix the node builds, measured on the widest topic so the
        # projection cannot come in under the run.
        prefix = estimate_tokens(EXTRACT_SYSTEM) + max(
            (
                estimate_tokens(
                    extract_context(
                        task=charter.task or charter.prompt,
                        outcome=charter.outcome,
                        topic=s.slug,
                        scope=s.scope or s.title,
                    )
                )
                for s in charter.topic_taxonomy
            ),
            default=0,
        )
        add(
            "extract",
            Role.BALANCED,
            calls=retained,
            prompt_tokens=retained * (prefix + per_paper),
            completion_tokens=retained * EXTRACT_COMPLETION_ALLOWANCE,
            basis=(
                f"{retained} papers on a {prefix}-token shared prefix; {with_pmcid} carry "
                f"a PMC id, of which {OPEN_ACCESS_RATE:.0%} assumed open access at "
                f"{FULL_TEXT_MULTIPLE:.0f}x abstract length, capped at "
                f"{int(MAX_SOURCE_CHARS / CHARS_PER_TOKEN):,} tokens of source"
            ),
        )

    notes.append(
        "a projection, not a quote: token counts are a 4-characters-per-token "
        "approximation, and how much full text a run actually reaches is the single "
        "largest thing it cannot know in advance"
    )
    notes.append(
        "a re-query round, if a topic comes up short, adds at most one curation call "
        "per thin topic — screening is bounded by the budget above whatever happens"
    )
    unpriced = [n.node for n in nodes if n.usd is None]
    if unpriced:
        notes.append(
            "unpriced (tokens only): "
            + ", ".join(unpriced)
            + " — set OKF_LOREMASTER_PRICE_<ROLE>_IN/_OUT for a USD figure"
        )
    return SpendEstimate(nodes=tuple(nodes), notes=tuple(notes))
