"""What a finished run keeps on disk.

A run always produces the OKF corpus — the vector store is built by walking it, so it
cannot be the only thing that ever existed. This is about what survives the run, asked
once at the end rather than guessed from a flag nobody remembers to pass.

Stdlib only, and deliberately its own module: `cli.py` needs the choice to declare the
option and to word the question, and importing `run` for it would pull pydantic and
litellm into `--help`.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Finalize"]


class Finalize(Enum):
    """Which resources a finished run leaves behind.

    The values are what a user types after `--finalize`. The numbers in `prompt_index`
    are what they press when asked, and the two must stay in step — the question and the
    flag are the same decision reached two ways.
    """

    OKF = "okf"
    VECTORS = "vectors"
    BOTH = "both"

    @property
    def builds_vectors(self) -> bool:
        """Whether the embedding pass runs at all.

        `OKF` skips it, which is the only one of the three that saves real time: the
        model download on first use is the slowest thing this tool does.
        """
        return self is not Finalize.OKF

    @property
    def keeps_okf(self) -> bool:
        """Whether the corpus survives.

        False only for `VECTORS`, and that is a deletion of the thing the run actually
        paid for — the caller confirms separately before acting on it.
        """
        return self is not Finalize.VECTORS

    @property
    def label(self) -> str:
        """How the choice reads back to a person."""
        return {
            Finalize.OKF: "OKF corpus only",
            Finalize.VECTORS: "vector store only",
            Finalize.BOTH: "OKF corpus and vector store",
        }[self]


# The order the end-of-run question offers them in, so `1`, `2`, `3` mean the same thing
# every time. `BOTH` is last and is the default, because it is what a run that nobody
# answers should keep.
PROMPT_ORDER = (Finalize.OKF, Finalize.VECTORS, Finalize.BOTH)
