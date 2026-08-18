"""Checking an extraction's literals against the text it was taken from. No model call.

Deterministic post-processing, not a prompt instruction. A model told to copy exactly
does so almost always, and the almost is what makes this necessary: an invented effect
size reads exactly like a real one, and it lands in a bundle another agent will treat as
evidence. So everything the extraction claims to have copied — numbers, quoted sentences
and vocabulary codes — is looked for in the source text afterward, by code, and what is
not there is removed.

Removed, not rejected. `PredictorRow.downgraded()` keeps the predictor, its
operationalization and its timing — all of which the paper did report — and drops only
the magnitude, lowering the row's confidence. Discarding the row would throw away good
evidence to punish one bad field; discarding the paper would let one unsupported number
cost a run everything else that paper said. The run continues either way. Quotes and
codes follow the same rule: the claim survives without the part that failed.

**A code is checked the same way a number is, and for a sharper reason.** Asked for the
ICD-10 code of a condition a paper is about, a model will supply a plausible one from its
own memory whether or not the paper printed it — and a fabricated code is worse than a
fabricated effect size, because it is short, well-formed, and indistinguishable from a
real one to anything downstream. `codes: []` is documented as the normal case precisely so
that dropping an unsupported code costs nothing.

**The scope of the check is the text the extractor actually read.** That is why
`fulltext` applies its length budget before `extract` sees anything, and why
`PaperText.text` is the whole prompt block rather than a pointer to it. Checking against
text the model was never shown would report correct extractions as fabricated, which is
the one failure that would make this worse than having no check at all.

Two limits, stated rather than hidden:

- A full text contains hundreds of numbers, so an invented one can coincide with a page
  number or a sample size. That is why a row's *own* magnitude, whenever it carries a
  verbatim `quote` the source contains, is checked against **the quote alone**: a scope of
  one sentence makes a coincidence unlikely rather than merely uncommon. An interaction
  coefficient is the documented exception and keeps the whole document — see
  `_verify_interactions`, which says why narrowing it would break the check rather than
  tighten it.
- A claim is matched at **its own** precision, so a source reading `1.84` supports a
  claim of `1.8`. That is a rounding, not a fabrication, and flagging it would fill the
  log with noise until nobody read it. The asymmetry is deliberate: a claim may be less
  precise than the source and never more. Matching at the coarser of the two instead
  would let any bare integer in the text — a year, a count, a table number — support a
  claimed effect of `4.44`, which is most of the numbers in a paper and no check at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from okf_loremaster.schemas import (
    CODE,
    EFFECT,
    INTERACTION,
    INTERVAL,
    QUOTE,
    CodedAs,
    Extraction,
    Interaction,
    NullFinding,
    PredictorRow,
    VocabularyHint,
)
from okf_loremaster.schemas.limits import MAX_LOCATED_QUOTE_WORDS, sentences

__all__ = [
    "ExtractionCheck",
    "Quantity",
    "RowCheck",
    "Source",
    "is_supported",
    "normalize",
    "quantities_in",
    "verify_extraction",
]

# The characters the literature prints where a keyboard would print ASCII. The middle
# dot is a decimal point in several journals' house styles; the true minus and the en
# dash arrive wherever text was converted from typeset copy. Written as named escapes
# rather than as glyphs: a hyphen, a minus and an en dash are indistinguishable in a
# diff, and this is the one module where the difference is the subject.
MINUS = "\N{MINUS SIGN}"
EN_DASH = "\N{EN DASH}"
MIDDLE_DOT = "\N{MIDDLE DOT}"

# `-` leads the class so it stays a literal rather than opening a range.
_DASHES = "-" + MINUS + EN_DASH
_NUMBER = re.compile(rf"[{_DASHES}]?\d[\d,]*(?:[.{MIDDLE_DOT}]\d+)?")
_DECIMALS = re.compile(rf"[.{MIDDLE_DOT}](\d+)$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Quantity:
    """A number, with how precisely it was written.

    The precision travels with the value because it decides what counts as agreement:
    a source that prints three decimals cannot contradict a claim that prints one.
    """

    value: float
    decimals: int

    @classmethod
    def parse(cls, literal: str) -> Quantity | None:
        normalized = literal.replace(",", "").replace(MIDDLE_DOT, ".")
        for dash in (MINUS, EN_DASH):
            normalized = normalized.replace(dash, "-")
        try:
            value = float(normalized)
        except ValueError:
            return None
        found = _DECIMALS.search(literal)
        return cls(value=value, decimals=len(found.group(1)) if found else 0)

    @classmethod
    def of(cls, value: float) -> Quantity:
        """A claimed number, at the precision it carries.

        Rendered fixed-point rather than through `repr`, which switches to exponent
        notation for small values and would report a precision of zero for them —
        turning the check on a tiny effect into no check at all.
        """
        text = f"{float(value):.10f}".rstrip("0")
        _, _, fraction = text.partition(".")
        return cls(value=float(value), decimals=len(fraction))


def quantities_in(text: str) -> tuple[Quantity, ...]:
    """Every number in a passage, in order."""
    found: list[Quantity] = []
    for match in _NUMBER.finditer(text):
        literal = match.group(0)
        if literal[0] in _DASHES and not _is_sign(text, match.start()):
            literal = literal[1:]
        quantity = Quantity.parse(literal)
        if quantity is not None:
            found.append(quantity)
    return tuple(found)


def _is_sign(text: str, start: int) -> bool:
    """Whether the dash at `start` negates the number after it.

    Three cases, and getting any of them wrong corrupts every number after it:

    - Attached to a digit or a letter, it is not a sign. `1.21-2.74` is an interval and
      `follow-up` is a word, so a reported interval does not silently acquire a negative
      lower bound nobody wrote.
    - Spaced away from a digit, it is still not a sign: `1.21 - 2.74` is the same
      interval typeset differently.
    - Anywhere else — after a word, a bracket, an equals sign, or the start of the text
      — it is a sign, which is what keeps the minus on `beta -0.44`.
    """
    index = start - 1
    if index < 0:
        return True
    if text[index].isdigit() or text[index].isalpha():
        return False
    spaced = index
    while spaced >= 0 and text[spaced] in " \t":
        spaced -= 1
    return not (spaced < index and spaced >= 0 and text[spaced].isdigit())


def is_supported(claim: Quantity, found: Sequence[Quantity]) -> bool:
    """Whether any number in `found` supports `claim` at the claim's own precision."""
    return any(_supports(claim, other) for other in found)


def _supports(claim: Quantity, found: Quantity) -> bool:
    """Whether `found`, rounded to how precisely `claim` was written, is `claim`.

    Asymmetric on purpose. Rounding the source down to the claim's precision forgives a
    paper printing `1.84` where the extraction says `1.8`; rounding the claim down to
    the source's would forgive an extraction saying `4.44` because some table somewhere
    printed a `4`.
    """
    return round(found.value, claim.decimals) == round(claim.value, claim.decimals)


def normalize(text: str) -> str:
    """Fold text for quote matching: case, punctuation and whitespace all collapse.

    Aggressive on purpose. A quote is a provenance claim, not the numeric check, and
    the cost of a false negative — a genuine quote dropped from the bundle — is higher
    than the cost of matching a sentence whose citation markers were reflowed.
    """
    return _NON_ALNUM.sub(" ", text.casefold()).strip()


class Source:
    """One paper's text, prepared once for every check made against it."""

    __slots__ = ("_bounded", "_normalized", "_quantities", "_sentences", "text")

    def __init__(self, text: str) -> None:
        self.text = text
        self._normalized = normalize(text)
        # `normalize` leaves space-separated alnum tokens, so padding the whole document
        # once turns token-boundary matching into a plain substring search.
        self._bounded = f" {self._normalized} "
        self._quantities = quantities_in(text)
        # Folded alongside the original, so `sentence_for` can match on the folded form
        # and return the published one. Built here rather than lazily because every
        # predictor row in the extraction asks for it.
        #
        # Line by line, because `sentences` collapses whitespace and a full text is not
        # prose end to end — it is section headings and table captions with paragraphs
        # between them. Split as one string, `## RESULTS` has no terminator, so it joins
        # the sentence after it and rides along on the front of every quote taken from
        # that section. A line break is a boundary here even when punctuation says
        # nothing.
        self._sentences = tuple(
            (part, normalize(part))
            for line in text.splitlines()
            for part in sentences(line)
        )

    @property
    def quantities(self) -> tuple[Quantity, ...]:
        return self._quantities

    def sentence_for(self, locator: str) -> str:
        """The source sentence a locator opens, or `""` if it opens none.

        An extraction writes the opening words of the sentence its numbers came from
        rather than the whole sentence, and this is what turns those words back into the
        published text. The stored quote is therefore sliced out of the source, which is
        a stronger guarantee than asking for a copy and checking it afterward: a copied
        sentence can drift into paraphrase and still pass, while a located one cannot
        differ from the source at all.

        Empty on a miss rather than a guess. The caller keeps whatever the model wrote
        and the ordinary quote check judges it, so a locator that finds nothing degrades
        to exactly the behavior that existed before locators did.

        Capped at `MAX_LOCATED_QUOTE_WORDS`, because what a locator lands in is not
        always a sentence — see `_windowed`.
        """
        needle = normalize(locator)
        if not needle:
            return ""
        for original, folded in self._sentences:
            if needle in folded:
                return _windowed(original, needle)
        return ""

    def holds_quote(self, quote: str) -> bool:
        needle = normalize(quote)
        return bool(needle) and needle in self._normalized

    def holds_code(self, code: str) -> bool:
        """Whether the source prints `code`, matched on token boundaries.

        Bounded, unlike `holds_quote`, because a code is short and often all digits.
        `250` is a real ICD-9 code, and an unbounded search finds it inside `1250`, inside
        `2500`, and inside most page ranges in the paper — which would accept every short
        code ever invented. A quote is long enough that the same risk does not arise.

        The normalized form is what is searched, so `E11.9`, `E11·9` and `E11 9` are one
        code. That is the same folding applied to quotes, and it matters more here: a
        code's punctuation is exactly what a typesetter reflows.
        """
        needle = normalize(code)
        return bool(needle) and f" {needle} " in self._bounded

    def holds(self, value: float) -> bool:
        return is_supported(Quantity.of(value), self._quantities)

    def scope(self, quote: str) -> tuple[tuple[Quantity, ...], bool]:
        """The numbers a row is checked against, and whether its quote was found.

        A row that quoted its source verbatim is checked against that one sentence. A
        row with no quote, or with one the text does not contain, falls back to the
        whole document — a weaker check, and the only alternative to none.
        """
        if quote.strip() and self.holds_quote(quote):
            return quantities_in(quote), True
        return self._quantities, False


@dataclass(frozen=True, slots=True)
class RowCheck:
    """What checking one predictor row found."""

    index: int
    predictor: str
    # The magnitude as the model claimed it, for a warning a person can act on.
    claimed: str = ""
    quote_missing: bool = False
    effect_missing: bool = False
    interval_missing: bool = False
    # `(feature, claimed)` per interaction whose number the source does not print. A tuple
    # rather than a flag because one row can state several, each naming a different
    # variable, and a warning that cannot say which one is a warning nobody can act on.
    interactions_missing: tuple[tuple[str, str], ...] = ()

    @property
    def clean(self) -> bool:
        return not (
            self.quote_missing
            or self.effect_missing
            or self.interval_missing
            or self.interactions_missing
        )

    @property
    def kind(self) -> str:
        """Which check took something from this row, in the order `note` reports them.

        Covers the three that `note` speaks for. Interactions are reported per casualty by
        `ExtractionCheck.notes`, because one row can lose several and this returns one.
        """
        if self.effect_missing:
            return EFFECT
        if self.interval_missing:
            return INTERVAL
        return QUOTE

    def note(self) -> str:
        if self.effect_missing:
            return f"{self.predictor!r}: {self.claimed} is not in the source text"
        if self.interval_missing:
            return (
                f"{self.predictor!r}: the interval around {self.claimed} "
                "is not in the source text"
            )
        return f"{self.predictor!r}: the quoted sentence is not in the source text"


@dataclass(frozen=True, slots=True)
class ExtractionCheck:
    """An extraction with its unsupported numbers stripped, and what was stripped."""

    extraction: Extraction
    rows: tuple[RowCheck, ...] = ()
    sample_size_missing: bool = False
    quotes_dropped: int = 0
    codes_dropped: int = 0
    # `(concept, system, code)` for each code the source did not print, so a warning can
    # name what was invented rather than only count it.
    codes_missing: tuple[tuple[str, str, str], ...] = ()

    @property
    def effects_dropped(self) -> int:
        return sum(1 for row in self.rows if row.effect_missing)

    @property
    def intervals_dropped(self) -> int:
        return sum(1 for row in self.rows if row.interval_missing)

    @property
    def interactions_dropped(self) -> int:
        return sum(len(row.interactions_missing) for row in self.rows)

    @property
    def clean(self) -> bool:
        return (
            not self.quotes_dropped
            and not self.codes_dropped
            and not self.sample_size_missing
            and all(row.clean for row in self.rows)
        )

    def notes(self) -> tuple[tuple[str, str], ...]:
        """`(kind, note)` per casualty, so a warning can name only its own.

        Tagged rather than plain, because these are read two ways. The run summary prints
        them all under a header that already carries the counts, where the mix is fine.
        A *warning* names one check, and for as long as this was an untagged list it
        attached the whole mix to the effect warning — which then said it had removed one
        effect size and went on to name five dropped intervals.
        """
        rows = tuple(
            (row.kind, row.note())
            for row in self.rows
            if row.quote_missing or row.effect_missing or row.interval_missing
        )
        interactions = tuple(
            (
                INTERACTION,
                f"{row.predictor!r}: {claimed} against {feature!r} "
                "is not in the source text",
            )
            for row in self.rows
            for feature, claimed in row.interactions_missing
        )
        codes = tuple(
            (CODE, f"{concept!r}: {system} {code} is not in the source text")
            for concept, system, code in self.codes_missing
        )
        return rows + interactions + codes


def verify_extraction(extraction: Extraction, source_text: str) -> ExtractionCheck:
    """Strip every number the source text does not contain, and report what went."""
    source = Source(source_text)
    extraction = _expand_quotes(extraction, source)

    rows: list[RowCheck] = []
    verified: list[PredictorRow] = []
    for index, row in enumerate(extraction.predictors):
        checked, outcome = _verify_row(index, row, source)
        verified.append(checked)
        rows.append(outcome)

    findings, quotes_dropped = _verify_findings(extraction.null_findings, source)
    quotes_dropped += sum(1 for outcome in rows if outcome.quote_missing)
    hints, codes_missing = _verify_vocabulary(extraction.vocabulary_hints, source)

    sample_size_missing = extraction.n is not None and not source.holds(float(extraction.n))
    updated = extraction.model_copy(
        update={
            "predictors": verified,
            "null_findings": findings,
            "vocabulary_hints": hints,
            "n": None if sample_size_missing else extraction.n,
        }
    )
    return ExtractionCheck(
        extraction=updated,
        rows=tuple(rows),
        sample_size_missing=sample_size_missing,
        quotes_dropped=quotes_dropped,
        codes_dropped=len(codes_missing),
        codes_missing=codes_missing,
    )


def _windowed(sentence: str, needle: str) -> str:
    """`sentence`, or the `MAX_LOCATED_QUOTE_WORDS` words that open at `needle`.

    A full text is prose, headings and *tables*, and BioC delivers a table as one
    unbroken line with no terminator anywhere in it — so `sentences` hands back the whole
    table as a single "sentence". Uncapped, a locator landing inside one carried the
    entire table into the bundle for every row that had quoted it: a real run produced a
    10,281-word document that was the same 1,166-word table eight times over.

    The window opens at the locator rather than centering on it, because a locator is by
    construction the *opening* words of the finding — the numbers it was written to point
    at come after it. What is returned is still a contiguous slice of the source, so it
    is still verbatim, and `holds_quote` still finds it.
    """
    words = sentence.split()
    if len(words) <= MAX_LOCATED_QUOTE_WORDS:
        return sentence
    start = _opens_at(words, needle.split())
    return " ".join(words[start : start + MAX_LOCATED_QUOTE_WORDS])


def _opens_at(words: Sequence[str], needle: Sequence[str]) -> int:
    """Where in `words` the folded `needle` begins, or 0 if it cannot be placed.

    Zero on a miss rather than a guess. `sentence_for` matched the needle as a substring
    of the folded span, which can land mid-token where this token-wise walk will not; the
    window then opens at the start of the span, which is the text the caller would have
    received anyway, only bounded.
    """
    tokens: list[str] = []
    owner: list[int] = []
    for index, word in enumerate(words):
        for token in normalize(word).split():
            tokens.append(token)
            owner.append(index)

    span = len(needle)
    for start in range(len(tokens) - span + 1):
        if tokens[start : start + span] == list(needle):
            return owner[start]
    return 0


def _expand_quotes(extraction: Extraction, source: Source) -> Extraction:
    """Grow each `quote` from the words the model wrote into the sentence they open.

    Runs before anything is checked, so every check downstream sees a full sentence and
    none of them had to learn that quotes arrive short. That is the whole reason this
    lives here rather than in the extract node: the numeric check narrows its scope to
    the quoted sentence, and a scope of ten words would be a narrower check than the one
    documented at the top of this module.

    Idempotent, and safe on extractions written before locators existed. A quote that is
    already a whole sentence locates that same sentence and expands to itself; a quote
    the source does not contain expands to nothing and is left exactly as written, for
    `_verify_row` and `_verify_findings` to drop as they always have.
    """

    def grown(quote: str) -> str:
        return source.sentence_for(quote) or quote

    return extraction.model_copy(
        update={
            "predictors": [
                row.model_copy(update={"quote": grown(row.quote)})
                for row in extraction.predictors
            ],
            "null_findings": [
                finding.model_copy(update={"quote": grown(finding.quote)})
                for finding in extraction.null_findings
            ],
        }
    )


def _verify_row(index: int, row: PredictorRow, source: Source) -> tuple[PredictorRow, RowCheck]:
    scope, quote_found = source.scope(row.quote)
    quote_missing = bool(row.quote.strip()) and not quote_found

    def supported(value: float | None) -> bool:
        return value is None or is_supported(Quantity.of(value), scope)

    effect_ok = supported(row.effect) and _matches_raw(row)
    interval_ok = supported(row.ci_low) and supported(row.ci_high)
    interactions, interactions_missing = _verify_interactions(row, source)

    updated = row.model_copy(update={"quote": ""}) if quote_missing else row
    if not effect_ok:
        updated = updated.downgraded()
    elif not interval_ok:
        updated = updated.without_interval()
    if interactions_missing:
        updated = updated.model_copy(update={"interacts_with": interactions})

    return updated, RowCheck(
        index=index,
        predictor=row.predictor,
        claimed=row.effect_raw or (f"{row.effect:g}" if row.effect is not None else "the effect"),
        quote_missing=quote_missing,
        effect_missing=not effect_ok,
        interval_missing=effect_ok and not interval_ok,
        interactions_missing=interactions_missing,
    )


def _verify_interactions(
    row: PredictorRow, source: Source
) -> tuple[list[Interaction], tuple[tuple[str, str], ...]]:
    """Strip interaction coefficients the source does not print. The claims themselves stay.

    **Scoped to the whole document, never to the row's quote.** A row's quote is the
    sentence stating what the predictor did to the *outcome*; a correlation between two
    predictors lives in a correlation matrix, a collinearity paragraph or a table caption
    somewhere else entirely. Narrowing to the quote would fail nearly every real
    coefficient, and a check that is wrong most of the time is worse than no check.

    The looser scope is affordable because less rides on it. A coincidence here leaves an
    interaction the extraction had already claimed, banded perhaps one step off; the same
    coincidence on an effect size would be the row's headline. What this is really for is
    the number a model *invented* to decorate a relationship it read in prose, and an
    invented `0.62` is no likelier to appear somewhere in the paper than an invented
    effect size is.

    `value` goes and `measure_raw` stays, which is the same trade `PredictorRow.downgraded`
    makes: the emitter reads the surviving raw string to tell "the paper reported no
    coefficient" from "the coefficient it reported is not in the text we read". Magnitude
    is a property of the two, so it falls back to `stated` on its own.

    Runs before `mirror_interactions`, so a mirror is built from a value that has already
    survived this. Mirroring first would double every fabricated number and then check
    both halves against the same absent source.
    """
    kept: list[Interaction] = []
    missing: list[tuple[str, str]] = []
    for interaction in row.interacts_with:
        if interaction.value is None or source.holds(interaction.value):
            kept.append(interaction)
            continue
        missing.append(
            (
                interaction.feature,
                interaction.measure_raw or f"{interaction.value:g}",
            )
        )
        kept.append(interaction.model_copy(update={"value": None}))
    return kept, tuple(missing)


def _matches_raw(row: PredictorRow) -> bool:
    """Whether the parsed effect is one of the numbers in the verbatim string.

    An internal check, needing no source: a row claiming `effect: 3.91` beside
    `effect_raw: "1.82 (95% CI 1.21-2.74)"` contradicts itself, and one of the two is
    wrong regardless of what the paper said. It also catches a silent unit conversion,
    which `effect_raw` exists specifically to make impossible.
    """
    if row.effect is None or not row.effect_raw.strip():
        return True
    return is_supported(Quantity.of(row.effect), quantities_in(row.effect_raw))


def _verify_findings(
    findings: Sequence[NullFinding], source: Source
) -> tuple[list[NullFinding], int]:
    """Drop quotes the source does not contain. The findings themselves stay.

    A null finding carries no magnitude to remove — its whole claim is that there was
    none — so an unsupported quote is the only thing here that can be checked, and the
    finding survives without it.
    """
    kept: list[NullFinding] = []
    dropped = 0
    for finding in findings:
        if finding.quote.strip() and not source.holds_quote(finding.quote):
            dropped += 1
            kept.append(finding.model_copy(update={"quote": ""}))
        else:
            kept.append(finding)
    return kept, dropped


def _verify_vocabulary(
    hints: Sequence[VocabularyHint], source: Source
) -> tuple[list[VocabularyHint], tuple[tuple[str, str, str], ...]]:
    """Drop codes the source does not print. The concepts themselves stay.

    The same shape as `_verify_findings`, and for the same reason: a concept is the
    paper's own words for a variable, and a paraphrase of them is still evidence, while a
    code is a literal string that either appears on the page or came from somewhere else.

    A hint left with no codes is the normal case rather than a loss, which is what makes
    dropping the safe move here. The concept is never dropped with the code: the paper did
    name that variable, and losing it would cost a reader the one part they were going to
    read anyway.
    """
    kept: list[VocabularyHint] = []
    missing: list[tuple[str, str, str]] = []
    for hint in hints:
        supported: list[CodedAs] = []
        for entry in hint.codes:
            if source.holds_code(entry.code):
                supported.append(entry)
            else:
                missing.append((hint.concept, entry.system, entry.code))
        kept.append(
            hint
            if len(supported) == len(hint.codes)
            else hint.model_copy(update={"codes": supported})
        )
    return kept, tuple(missing)
