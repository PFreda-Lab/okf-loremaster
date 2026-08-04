"""What evidence strength scoring produced, for one paper and for each of its rows.

Deliberately not part of `Extraction`. The extract node hands `Extraction` to the model
as its response format, so a field added there is a field the model is asked to fill —
and a model-supplied strength score is exactly what this is not. Every number here is
computed by `strength.py` from fields that were extracted, verified, and only then
scored, which is what makes it reproducible and what lets the weights change without
re-reading a single paper.

`parts` is not decoration. A score whose components are not visible is one nobody can
argue with when a paper that obviously belongs scores badly — the same reason
`ranking.relevance` returns its contributions alongside its total.
"""

from __future__ import annotations

from pydantic import Field

from okf_loremaster.schemas.common import Model, StrengthGrade

__all__ = ["PaperStrength", "RowStrength"]


class RowStrength(Model):
    """One predictor row's strength. Positionally parallel to `Extraction.predictors`.

    By position rather than by key because the rows have no stable identifier — the
    predictor name is free text and repeats within a paper. Both lists are written in
    one pass in `reconcile`, after length budgets have already dropped whatever they
    were going to drop, so the pairing cannot drift.
    """

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    grade: StrengthGrade = StrengthGrade.UNGRADED
    parts: dict[str, float] = Field(default_factory=dict)
    # Components with nothing behind them, named. A row scored on two of three signals
    # is a weaker claim than one scored on all three, and the score alone cannot say so.
    unmeasured: list[str] = Field(default_factory=list)


class PaperStrength(Model):
    """One paper's strength, and every row's."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    grade: StrengthGrade = StrengthGrade.UNGRADED
    parts: dict[str, float] = Field(default_factory=dict)
    unmeasured: list[str] = Field(default_factory=list)
    rows: list[RowStrength] = Field(default_factory=list)

    @property
    def graded(self) -> bool:
        return self.grade is not StrengthGrade.UNGRADED
