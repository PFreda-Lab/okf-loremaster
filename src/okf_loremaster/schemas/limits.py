"""Length budgets, and the pure functions that enforce them.

A concept file exists to be read by an agent that is holding a dozen of them at once.
Left alone, an extraction model writes a page per paper, and a topic of forty becomes
something no reader — human or otherwise — will get through. So every free-text field
has a ceiling.

**Truncate and warn, never reject.** An over-long extraction is a good extraction that
ran on; discarding it would cost a real model call and gain nothing. The node emits a
`WarningEvent` naming the field and `validate` aggregates them.

"Two lines" is operationalized as a character budget. Frontmatter holds `description`
as a single YAML scalar, so how many lines it occupies is a property of whoever is
looking at it, not of the file.
"""

from __future__ import annotations

import re

__all__ = [
    "MAX_BODY_WORDS",
    "MAX_BOTTOM_LINE_SENTENCES",
    "MAX_CAVEAT_SENTENCES",
    "MAX_DESCRIPTION_CHARS",
    "MAX_NULL_FINDINGS",
    "MAX_PREDICTOR_ROWS",
    "MAX_QUOTE_WORDS",
    "MAX_SOURCE_CHARS",
    "MAX_TAGS",
    "MAX_VOCABULARY_HINTS",
    "sentences",
    "truncate_chars",
    "truncate_sentences",
    "word_count",
]

MAX_DESCRIPTION_CHARS = 200
MAX_BOTTOM_LINE_SENTENCES = 2
MAX_PREDICTOR_ROWS = 12
MAX_CAVEAT_SENTENCES = 3
MAX_TAGS = 8

# The two lists that used to have no ceiling at all — not in the prompt, not here, not on
# the model. Left open, they are what a reply spends its last tokens on, and a reply that
# runs out of tokens is a *lost paper* rather than a long one. Both are set where a
# generous paper still fits: a study reporting more than eight nulls is listing its whole
# adjustment set, and one naming more than eight coded variables is printing its data
# dictionary. Enforced the same way as every other budget here — truncate and warn.
MAX_NULL_FINDINGS = 8
MAX_VOCABULARY_HINTS = 8

# How much of a sentence an extraction has to write for `quote`. The model emits the
# opening words; `expand_quote` finds the sentence they open and stores that instead.
#
# This is the single largest saving in the node: a verbatim quote is the longest field in
# a predictor row, there is one per row, and copying it is *transcription of text already
# on disk* — paid for at output rates, in the one node that makes a call per paper. It
# also makes the guarantee
# stronger rather than weaker. A copied quote can drift into paraphrase and nothing
# catches a plausible one; a located quote is sliced out of the source, so it is verbatim
# by construction.
MAX_QUOTE_WORDS = 10

# --- the body guideline, which is derived rather than chosen ------------------
#
# Every list and every prose field above has a hard cap, so once they have all been
# applied the only way a document can still be long is that an individual *cell* ran on
# — a paragraph-length operationalization, or a quote whose located sentence turned out
# to be a whole paragraph. That is what `MAX_BODY_WORDS` is for, and it is the one
# number here that truncates nothing: there is no lever left that would not cut a table
# mid-row. It is a signal, and a signal that fires on everything is not one.
#
# The flat 400 this replaces fired on 11 of 12 papers in a smoke run, because it was set
# when `quote` held the ten words the model wrote rather than the sentence `expand_quote`
# now stores in its place. Nothing was wrong with the documents.
#
# The three figures below are the 90th percentile, measured 2026-08-04 over 1,038
# predictor rows and 401 null findings in two finished bundles. The 90th and not the
# median on purpose: a verbose document sitting at every cap is still a legitimate
# document, and must not warn. At these values the guideline fires on 10% of the
# documents in those bundles, and the ones it fires on are runaways — the worst is 6,363
# words. The flat 400 fired on 81% of them.
P90_PREDICTOR_ROW_WORDS = 110
P90_NULL_FINDING_WORDS = 34
# `bottom_line` at its two sentences and `caveats` at its three, plus `population` and
# `outcome_definition`, which have no cap of their own and render inside the same section.
P90_PROSE_WORDS = 227

MAX_BODY_WORDS = (
    MAX_PREDICTOR_ROWS * P90_PREDICTOR_ROW_WORDS
    + MAX_NULL_FINDINGS * P90_NULL_FINDING_WORDS
    + P90_PROSE_WORDS
)

# The one budget here that bounds an input rather than an output: how much of a paper an
# extraction prompt may carry. At roughly four characters a token that is about 6,000
# tokens of source, which leaves a long review readable without any single paper
# dominating a run's spend.
#
# It lives beside the output budgets so that `graph.nodes.fulltext`, which applies it,
# and `llm.estimate`, which projects what it will cost, read the same number without
# `llm` having to import from `graph`.
MAX_SOURCE_CHARS = 24_000

# A sentence boundary is a terminator, optional closing quote or bracket, then space.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"'\)\]]*\s+")

# Tokens that end in a period without ending a sentence. Deliberately short: this
# guards the common cases and the cost of a miss is a slightly long line, not a wrong
# one, because everything here only ever truncates.
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "ca",
        "cf",
        "e.g",
        "eq",
        "est",
        "fig",
        "figs",
        "i.e",
        "max",
        "min",
        "no",
        "ref",
        "refs",
        "st",
        "vs",
    }
)

_TRAILING_PUNCT = '"\'()[]'


def sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping terminators.

    Not a general-purpose sentence tokenizer, and does not need to be — a boundary that
    is missed leaves two sentences joined, which counts as one and lets slightly more
    text through.
    """
    stripped = " ".join(text.split())
    if not stripped:
        return []

    parts: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(stripped):
        candidate = stripped[start : match.start()]
        if _ends_with_abbreviation(candidate):
            continue
        parts.append(candidate)
        start = match.end()
    tail = stripped[start:]
    if tail:
        parts.append(tail)
    return parts


def _ends_with_abbreviation(fragment: str) -> bool:
    last = fragment.rstrip(_TRAILING_PUNCT).split()
    if not last:
        return False
    token = last[-1].rstrip(".").lower()
    # A single letter before a period is an initial ("Smith J. et al"), not an ending.
    return token in _ABBREVIATIONS or len(token) == 1


def truncate_sentences(text: str, limit: int) -> tuple[str, bool]:
    """First `limit` sentences, and whether anything was dropped."""
    parts = sentences(text)
    if len(parts) <= limit:
        return " ".join(parts), False
    return " ".join(parts[:limit]), True


def truncate_chars(text: str, limit: int) -> tuple[str, bool]:
    """Trim to `limit` characters on a word boundary, and whether it was trimmed."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed, False
    cut = collapsed[: limit - 1]
    # Prefer a word boundary, unless that would throw away most of the budget.
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.") + "…", True


def word_count(text: str) -> int:
    return len(text.split())
