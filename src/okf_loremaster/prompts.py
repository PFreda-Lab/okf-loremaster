"""Prompt text for the judgment nodes, as module constants.

String constants rather than files: prompts are code — they change with the schema they
have to satisfy, and they belong in the same review as it. Keeping them here also means
no package-data entry to forget and nothing to resolve at runtime.

Every prompt here is written so that the *subject* arrives from the user, never from
this file. No example names a condition, a drug, a lab, a specialty, or a coding system;
where an example would otherwise have to, it uses an angle-bracket placeholder. That is
not a stylistic choice. An example naming a real condition biases every charter drafted
afterward toward it, which is exactly the failure `tests/test_domain_agnostic.py` exists
to prevent.
"""

from __future__ import annotations

from okf_loremaster.schemas.limits import (
    MAX_DESCRIPTION_CHARS,
    MAX_NULL_FINDINGS,
    MAX_PREDICTOR_ROWS,
    MAX_QUOTE_WORDS,
    MAX_TAGS,
    MAX_VOCABULARY_HINTS,
)

__all__ = [
    "CURATE_SYSTEM",
    "EXTRACT_SYSTEM",
    "QUERY_PLAN_SYSTEM",
    "SCREEN_SYSTEM",
    "charter_system",
    "charter_user",
    "curate_user",
    "extract_context",
    "extract_user",
    "query_plan_user",
    "screen_context",
    "screen_user",
]


_CHARTER_SYSTEM = """\
You are drafting the terms of reference for a literature review that will be read by \
another agent, not by a person browsing for interest.

Return a single JSON object and nothing else.

Decide these, in this order:

1. `task` — the user's request restated as one sentence naming what the review must \
support.
2. `population` — who or what is being studied. Empty string if the request does not \
imply one.
3. `outcome` — the thing being predicted, measured, or explained. Empty string if the \
request is not about one.

`population` and `outcome` are searched verbatim as quoted phrases, so each must be a \
short noun phrase of about four words that a paper would actually print. A longer \
restatement of the request matches nothing at all.
4. `inclusion` / `exclusion` — short criteria a screener can apply to a title and \
abstract alone. Four or fewer each. Say nothing you would not be able to check from an \
abstract.
5. `topic_taxonomy` — {topics}. Between them they must cover the request without \
overlapping. Each needs:
   - `slug`: lowercase, hyphenated, 2-64 characters, unique. It becomes a directory \
name.
   - `title`: how a reader would name the topic.
   - `scope`: one line saying what belongs here and what does not.
   - `seed_terms`: 3-6 concepts a literature search for this topic would use. Concepts, \
not query syntax — no field tags, no boolean operators, no quotation marks.

6. `sample_size_typical` / `sample_size_large` — the analytic sample size of an ordinary \
study in this literature, and of a large one. Whole numbers, and `sample_size_large` \
must be the bigger of the two. These set the scale that evidence strength scores sample \
size against, and there is no answer that holds across literatures: a few hundred \
participants is a large cohort for a rare condition and a pilot for a national registry. \
Judge it from the population above. If you genuinely cannot, omit both — sample size \
then scores as unmeasured, which is honest. Never guess one without the other.

Before you divide anything, list out the distinct ways the outcome could actually vary — \
the separate mechanisms, exposures, and circumstances that researchers have proposed as \
explaining it. Build the taxonomy from that list. If you pick the headings first and \
fill them in afterward, whole classes of predictor never get considered at all.

Look beyond the one field the question obviously belongs to. That field has published \
the most on this outcome, so it will fill the taxonomy by default unless you go looking \
elsewhere. What gets left out is what other fields study: causes further upstream, the \
conditions under which people are treated and measured, and factors acting on the whole \
person rather than on the mechanism. If every topic you wrote could have come from the \
same department, you stopped too early.

Widen the explanation, not the question. `population` and `outcome` stay exactly as the \
request set them. You are asking what could account for that outcome in that population \
— not about a different outcome, a related condition, or a broader population.

On the taxonomy: divide by the kind of evidence a reader would go looking for, not by \
publication type or study design. A topic that would hold two papers is not a topic; a \
topic that would hold half the corpus is two topics.

Do not invent a scope the request did not ask for, and do not narrow one it did.\
"""

# Below this the taxonomy stops dividing the literature and starts enumerating it, so
# the floor only follows `max_topics` down, never up.
_CHARTER_TOPIC_FLOOR = 4


def charter_system(max_topics: int) -> str:
    """The charter prompt, with the taxonomy's size set by `--max-topics`.

    A function rather than a constant because a ceiling the model never sees is not a
    setting: `Charter.problems` would warn about a taxonomy that the prompt had asked
    for. One number, said once, in the only place that can act on it.
    """
    floor = min(_CHARTER_TOPIC_FLOOR, max_topics)
    if floor < max_topics:
        topics = f"between {floor} and {max_topics} topics"
    else:
        topics = "exactly 1 topic" if max_topics == 1 else f"exactly {max_topics} topics"
    return _CHARTER_SYSTEM.format(topics=topics)


def charter_user(prompt: str) -> str:
    """The charter request: the user's words, unedited."""
    return "\n".join(["Request:", "", prompt.strip()])


QUERY_PLAN_SYSTEM = """\
You are planning PubMed searches for a literature review. Return a single JSON object \
and nothing else.

Produce one query per topic in the charter's taxonomy, plus one or two broad queries \
covering the request as a whole. Between 6 and 12 in total.

Each query needs:
  - `term`: the PubMed query string.
  - `rationale`: one line saying what it is for. This is shown to a human before \
anything is spent.
  - `topic`: the slug of the topic it is meant to fill, or an empty string for a broad \
query.

Query syntax rules, all of which matter:

  - Tag every clause. An untagged word is expanded by automatic term mapping into \
something you did not write.
  - Use `[tiab]` for concepts. Do not use MeSH tags: MeSH is assigned months after \
publication, so a MeSH query systematically misses recent work.
  - Quote every phrase: `"<two or more words>"[tiab]`.
  - Combine synonyms with OR inside parentheses, and distinct concepts with AND: \
`("<term a>"[tiab] OR "<term b>"[tiab]) AND "<term c>"[tiab]`.
  - Keep phrases to four words or fewer. PubMed matches a phrase verbatim, so a long \
one matches nothing.
  - Use only field tags you are certain of. PubMed does not reject an unknown tag — it \
silently rewrites the clause into a free-text search, returns tens of thousands of \
irrelevant hits, and reports no error at all.
  - Do not add language or date filters. Those are applied afterward from the charter, \
identically across every query.

Aim each query at a few hundred to a few thousand hits. Tens of thousands means the \
query is too broad to rank usefully; under about twenty means it is too narrow to be \
worth a slot.\
"""


def query_plan_user(
    *,
    task: str,
    population: str,
    outcome: str,
    topics: list[tuple[str, str, list[str]]],
    max_queries: int,
) -> str:
    """The query-planning request, assembled from the charter.

    Takes the charter's parts rather than the charter itself so that the prompt cannot
    quietly grow a dependency on fields it does not use — what the planner sees is
    exactly what is listed here.
    """
    lines = [f"Task: {task}"]
    if population:
        lines.append(f"Population: {population}")
    if outcome:
        lines.append(f"Outcome: {outcome}")
    lines += ["", "Topics to cover:"]
    for slug, scope, seeds in topics:
        seed_text = "; ".join(seeds) if seeds else "no seed terms given"
        lines.append(f"  - {slug}: {scope or 'no scope given'} [seeds: {seed_text}]")
    lines += ["", f"Return at most {max_queries} queries."]
    return "\n".join(lines)


SCREEN_SYSTEM = """\
You are screening one paper for a literature review, from its title and abstract alone. \
Return a single JSON object and nothing else.

Fields:
  - `include`: true if the paper reports something this review could use as evidence.
  - `relevance`: 0 unrelated, 1 tangential, 2 relevant, 3 directly on point.
  - `topic`: the slug of the topic it belongs on, copied exactly from the list below. \
Exactly one slug, never a list — a paper that spans several belongs on the one it says \
most about. Empty string if none of them fits.
  - `reason`: one clause, fifteen words or fewer.
  - `confidence`: "high", "medium" or "low" — how sure you are of `include`, not how \
strong the paper is.

`include` and `relevance` are separate answers and both are used. A paper excluded at \
relevance 2 or 3 is the first one reconsidered if a topic later comes up short, so an \
honest relevance on an excluded paper is worth as much as an inclusion. Do not collapse \
them into each other: relevance 3 with `include` false is the normal answer for a paper \
squarely on topic that fails an exclusion criterion.

Judge against the review's question and the criteria given, and nothing else. You are \
reading an abstract, so decide only what an abstract supports — one that is too vague to \
tell is `confidence: "low"`, not an exclusion. Exclude only on evidence in front of you.\
"""


def screen_context(
    *,
    task: str,
    population: str,
    outcome: str,
    inclusion: list[str],
    exclusion: list[str],
    topics: list[tuple[str, str]],
) -> str:
    """The review's terms, prefixed to every screening call.

    Assembled once per run and reused verbatim, so every call in the largest node of the
    pipeline shares a byte-identical prefix. That is the shape a provider's prompt cache
    can charge once for; rebuilding the text per paper would look the same and cost the
    full rate on every one of a few hundred calls.
    """
    lines = [f"Review question: {task}"]
    if population:
        lines.append(f"Population of interest: {population}")
    if outcome:
        lines.append(f"Outcome of interest: {outcome}")
    for label, items in (("Include", inclusion), ("Exclude", exclusion)):
        for item in items:
            lines.append(f"{label}: {item}")
    lines += ["", "Topics:"]
    if topics:
        lines += [f"  - {slug}: {scope}" for slug, scope in topics]
    else:
        lines.append("  (none — leave `topic` empty)")
    return "\n".join(lines)


def screen_user(*, context: str, paper: str) -> str:
    """One screening request: the shared context, then the paper.

    The paper goes last so the varying part of the prompt is the suffix, which is what
    lets the constant prefix above be cached.
    """
    return f"{context}\n\nPaper:\n\n{paper.strip()}"


CURATE_SYSTEM = """\
You are curating one topic of a literature review. Unlike the screener, you see every \
paper proposed for this topic at once. Return a single JSON object and nothing else:

  {"decisions": [{"pmid": "<id>", "keep": true, "rationale": "<one clause>"}],
   "missing": "<what the topic still lacks>"}

Give one decision per paper offered, with its `pmid` copied exactly as given.

Keep a paper when it belongs on this topic specifically, not merely somewhere in the \
review. Because you see the whole topic at once, you can catch what the screener could \
not: prefer the paper reporting a usable finding over the one that only mentions the \
topic, prefer the primary report over a narrative review of it, and where several papers \
report the same result from the same cohort, keep the fullest one.

The topic's floor and ceiling are given below. Neither is a quota to fill. Keeping a \
paper that does not belong is worse than a thin topic: a shortfall is refilled by a \
further, broader search, while a padded topic is never questioned again. Ordering and \
trimming to the ceiling happen afterward in code, so do not rank, and do not stop at the \
ceiling — judge each paper on its own.

`missing` is what this topic still lacks, written as the topics a further literature \
search should look for. It is turned into another query, so name concepts, not \
complaints: "<concept a>, <concept b>", never "not enough good papers". Empty string \
when the topic is in good shape.\
"""


def curate_user(
    *,
    task: str,
    topic: str,
    scope: str,
    seed_terms: list[str],
    floor: int,
    ceiling: int,
    papers: list[tuple[str, str, int, str]],
) -> str:
    """One topic's curation request.

    `papers` is `(pmid, title, relevance, screener's reason)`. The screener's own note
    travels with each paper because it is already paid for and it says what the title
    does not — why a paper the curator is now second-guessing was proposed at all.
    """
    lines = [
        f"Review question: {task}",
        "",
        f"Topic: {topic}",
        f"Scope: {scope or 'no scope given'}",
    ]
    if seed_terms:
        lines.append(f"Seed concepts: {', '.join(seed_terms)}")
    lines += [
        f"Holds between {floor} and {ceiling} papers.",
        "",
        f"Papers proposed for this topic ({len(papers)}):",
    ]
    for pmid, title, relevance, reason in papers:
        note = f" — screener: {reason}" if reason else ""
        lines.append(f"  - {pmid} [relevance {relevance}] {title}{note}")
    return "\n".join(lines)


EXTRACT_SYSTEM = f"""\
You are reading one paper and recording what it found, so that another agent can use it \
as evidence without opening the paper. Return a single JSON object and nothing else.

Write the JSON compactly: no indentation, no line breaks between fields, no spaces after \
`:` or `,`. It is read by a parser, never by a person. Pretty-printing the same content \
costs about a fifth of the reply's room and buys nothing.

Fields:
  - `description`: at most {MAX_DESCRIPTION_CHARS} characters — about two lines — saying \
what this paper reports. An agent reads this before deciding whether to open the file, \
so lead with the finding, not with the topic. Anything past that is cut mid-sentence.
  - `bottom_line`: the finding itself, in at most two sentences.
  - `study_design`: as the paper describes it.
  - `design`: the same thing as one of "systematic_review", "randomized_trial", \
"prospective_cohort", "retrospective_cohort", "case_control", "cross_sectional", \
"case_series", "modeling", "narrative_review", "unclear". Go by what was actually done, \
not by what the paper calls itself. "unclear" is a real answer and the right one when \
the methods do not say; it is scored as unknown rather than as poor, so guessing is \
strictly worse than admitting it.
  - `n`: the analytic sample size, as a whole number, or null if the paper does not \
state one. Never estimate it.
  - `population`: who or what was studied, in one short phrase.
  - `outcome_definition`: how the outcome was defined or measured in this study \
specifically.
  - `adjusted_for`: the covariates the paper says its models adjusted for, in its own \
words. Empty list if it does not say. Do not list the predictors themselves — only what \
was held constant while they were estimated.
  - `predictors`: one row per relationship this paper reports, and at most \
{MAX_PREDICTOR_ROWS} of them. A paper reporting more than that is reporting a model's \
whole coefficient table; give the rows that answer the question above and stop. Rows \
past the limit are discarded after the fact, so writing them costs the reply the room \
it needs to finish — and a reply that stops mid-row is not a shorter extraction, it is \
no extraction at all. See below.
  - `null_findings`: one row per relationship the paper looked for and did not find, \
and at most {MAX_NULL_FINDINGS} of them. This is as valuable as the section above it and \
is almost always shorter than it should be, because a result that did not hold is easy \
to read past. If the paper truly reports none, return one row with `predictor` set to \
"none reported".
  - `vocabulary_hints`: one entry per variable this paper names, so a reader can find \
the same variable in their own data, and at most {MAX_VOCABULARY_HINTS} of them. Each \
entry has a `concept` — the paper's own words for it — and `codes`, a list of the codes \
this paper gives for that same concept, each with the `system` it comes from and the \
`code` itself. Most papers name variables and never code them: leave `codes` empty \
rather than looking a code up or guessing one. Record a code only where the paper \
prints it.
  - `caveats`: at most three sentences on what would make this evidence weaker than it \
looks.
  - `tags`: at most {MAX_TAGS} short topic terms.

Each row in `predictors`:
  - `predictor`: the thing being related to the outcome.
  - `operationalization`: how it was actually measured or defined here — the difference \
between a concept and something a person could compute.
  - `timing`: when it is observed relative to the outcome. Say plainly when the paper \
does not make this clear; a predictor measured after the outcome is a different claim \
from one measured before it.
  - `outcome`: the outcome this row is about.
  - `evidence_type`: "observational_association" when the paper observed a relationship, \
"randomized_intervention" when it assigned the exposure, "outcome_definition" when the \
row is about how the outcome itself is measured rather than about a predictor of it.
  - `effect`, `ci_low`, `ci_high`: the numbers, as numbers. Null when the paper gives \
none.
  - `effect_measure`: what the number is, named as the paper names it.
  - `effect_raw`: the effect exactly as the paper printed it, copied character for \
character.
  - `p_value`: as printed, as a string. "<0.001" and "NS" are answers; do not convert \
them to a number.
  - `direction`: "increases", "decreases", "none" or "unclear".
  - `confidence`: how sure you are that you read this row correctly — not how good the \
study is. A crisply reported number from a small unadjusted survey is high confidence. \
How much weight the study carries is scored separately and is not yours to judge.
  - `adjusted`: true when *this* number came from a model holding other variables \
constant, false when it is a crude or unadjusted estimate, null when the paper does not \
make it clear. A paper printing both an unadjusted and an adjusted column is reporting \
two different claims; record the one this row's number came from. Null is scored as \
unknown, so it costs a paper nothing to be honest here.
  - `quote`: the **first {MAX_QUOTE_WORDS} words or fewer** of the sentence the numbers \
came from, copied exactly as the text above prints them. Not the whole sentence. Enough \
words to pick that sentence out from every other sentence in the paper, and no more — \
the rest of it is retrieved from the source afterward and added for you.

Every number you record is checked against the text afterward, automatically. One that \
is not there is deleted and its row is marked less reliable, so a number you can see is \
worth more than one you can reconstruct. The `quote` is what makes that check strict: it \
is grown back into the full sentence and the numbers are checked against that one \
sentence rather than against the whole paper. Opening words that match nothing lose the \
row that narrower check, so copy them exactly — including any digits or symbols they \
contain, and starting at the beginning of the sentence rather than in the middle of it.

A relationship reported without a magnitude is still a relationship reported: leave the \
numeric fields null and record the row anyway. Record only what this paper says. Do not \
add what you know from elsewhere, and do not round a partial result up into a \
complete-looking one.\
"""


def extract_context(
    *,
    task: str,
    outcome: str,
    topic: str,
    scope: str,
) -> str:
    """The review's terms, prefixed to every extraction call on one topic.

    Assembled per topic rather than per paper, for the same caching reason as
    `screen_context`: papers on a topic are extracted back to back, so a byte-identical
    prefix across them is the shape a provider's prompt cache can charge once for.

    Nothing here names a coding system. An extraction records whatever vocabulary its
    paper used, so there is no list to agree on and none to get wrong in advance.
    """
    lines = [f"Review question: {task}"]
    if outcome:
        lines.append(f"Outcome of interest: {outcome}")
    lines += [
        "",
        f"Topic: {topic}",
        f"Scope: {scope or 'no scope given'}",
        "",
    ]
    return "\n".join(lines)


def extract_user(*, context: str, paper: str) -> str:
    """One extraction request: the shared context, then the source text.

    The source goes last so the varying part is the suffix. `paper` is passed through
    unchanged and stored as `PaperText.text`, because verification checks the model's
    numbers against exactly the string it was given here.
    """
    return f"{context}\n\nSource:\n\n{paper.strip()}"
