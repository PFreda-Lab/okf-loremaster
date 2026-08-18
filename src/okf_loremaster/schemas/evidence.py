"""The text an extraction read, and what checking it against that text found.

`PaperText.text` is deliberately the **whole prompt block** shown to the extractor —
header, section labels and all — rather than a pointer to where the text came from or a
reconstruction of it. Numeric verification checks the model's numbers against this
string, and that check is only sound if the string is byte-for-byte what the model saw.
A pointer would let the two drift the moment section selection or truncation changed,
and the failure would look like fabricated numbers rather than like a bug here.

The cost of that decision is that `PaperText` is the largest thing in the checkpointed
state, re-serialized on every super-step. That is why `fulltext` applies its length
budget before storing rather than `extract` applying one on the way out.
"""

from __future__ import annotations

from pydantic import Field

from okf_loremaster.schemas.common import Model, TextBasis

__all__ = [
    "CODE",
    "EFFECT",
    "INTERACTION",
    "INTERVAL",
    "QUOTE",
    "PaperText",
    "VerificationExample",
    "VerificationSummary",
]

# Which check took something, for an example that has to find its way back to the one
# warning that counted it. Strings rather than an enum because they are only ever
# compared to each other and written to a checkpoint.
EFFECT = "effect"
INTERVAL = "interval"
QUOTE = "quote"
CODE = "code"
INTERACTION = "interaction"


class PaperText(Model):
    """Exactly what one paper's extraction was shown."""

    pmid: str
    basis: TextBasis = TextBasis.ABSTRACT
    # Verbatim from BioC's `infons.license`, never inferred. Empty for an abstract-only
    # record: no license was ever served to us, and guessing one is how a bundle becomes
    # undistributable without anyone noticing.
    license: str = ""
    pmcid: str = ""
    text: str = ""
    # Section types kept, in the order they appear in the prompt. Empty for an abstract.
    sections: list[str] = Field(default_factory=list)
    # What the retrieved full text ran to before the budget was applied, so a run can
    # report how much of the literature it actually read.
    source_chars: int = 0
    truncated: bool = False

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def is_full_text(self) -> bool:
        return self.basis is TextBasis.FULL_TEXT


class VerificationExample(Model):
    """One named casualty of numeric verification, and which check took it.

    The kind travels with the note because the two are read apart. The run summary prints
    every example under one header that already carries the counts, where the mix reads
    correctly. A *warning* names one check — and for as long as this was an untagged list
    of strings, `reconcile` attached the whole mix to the effect warning, which then said
    it had removed one effect size and went on to name five dropped intervals.
    """

    kind: str
    note: str


class VerificationSummary(Model):
    """What deterministic checking did to a run's extractions."""

    papers: int = 0
    rows: int = 0
    effects_dropped: int = 0
    intervals_dropped: int = 0
    quotes_dropped: int = 0
    codes_dropped: int = 0
    sample_sizes_dropped: int = 0
    # Interaction coefficients the source text does not contain. Counted apart from
    # `effects_dropped` because the two fail for different reasons and are read by
    # different people: an effect size is the row's headline and its loss is a hole in the
    # finding, while an interaction coefficient is a secondary number the model was more
    # likely to have paraphrased than invented. Merging them would make a run look worse at
    # its main job than it was.
    interactions_dropped: int = 0
    # A handful of the offending rows, named. Not all of them: this gets printed, and a
    # run where every row failed should say so in one line rather than in two hundred.
    examples: list[VerificationExample] = Field(default_factory=list)

    def examples_for(self, kind: str) -> list[str]:
        """The notes from one check, for the warning that reports that check."""
        return [example.note for example in self.examples if example.kind == kind]

    @property
    def clean(self) -> bool:
        return not (
            self.effects_dropped
            or self.intervals_dropped
            or self.quotes_dropped
            or self.codes_dropped
            or self.sample_sizes_dropped
            or self.interactions_dropped
        )

    def line(self) -> str:
        """One line for the run summary."""
        if not self.rows:
            return "nothing to check"
        if self.clean:
            return f"{self.rows} row(s) checked, all supported by the source text"
        parts = []
        if self.effects_dropped:
            parts.append(f"{self.effects_dropped} effect(s) dropped")
        if self.intervals_dropped:
            parts.append(f"{self.intervals_dropped} interval(s) dropped")
        if self.quotes_dropped:
            parts.append(f"{self.quotes_dropped} quote(s) dropped")
        if self.codes_dropped:
            parts.append(f"{self.codes_dropped} code(s) dropped")
        if self.sample_sizes_dropped:
            parts.append(f"{self.sample_sizes_dropped} sample size(s) dropped")
        if self.interactions_dropped:
            parts.append(f"{self.interactions_dropped} interaction value(s) dropped")
        return f"{self.rows} row(s) checked; " + ", ".join(parts)
