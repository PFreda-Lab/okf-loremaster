"""What a run is willing to read, as an instruction rather than an observation.

The default takes full text where the open-access subset has it and the abstract
everywhere else, which is what makes a corpus of 200 papers affordable. The other two
values narrow the *corpus*, not merely the reading: a paper that cannot satisfy the
policy is dropped in `rank`, before screening pays for it.

Stdlib only, and deliberately its own module for the same reason as `finalize.py`:
`cli.py` needs the choice to declare the option, and importing `schemas` for it would
pull pydantic into `--help`. `schemas.common` re-exports it so the rest of the package
imports it from beside `TextBasis`, which is the type it is most easily confused with.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["TextBasisPolicy"]


class TextBasisPolicy(StrEnum):
    """What a run was *told* to read, which is not what it read.

    `schemas.TextBasis` is an observation about one paper; this is an instruction about a
    whole run, and the two answer different questions. "Every paper in this bundle is
    abstract-only" is ambiguous on its own — it is equally "we asked for abstracts" and
    "nothing here was open access" — and only the second is a fact about the literature.
    So the policy is recorded in the manifest beside the per-paper basis, and the two are
    read together.

    The values are spelled for a command line, which is why `FULL_TEXT` here is
    `full-text` where `TextBasis.FULL_TEXT` is `full_text`. Nothing anywhere compares one
    against the other as a string, and nothing should: they are near-misses that would
    each be a plausible typo for the other, and the only reason the difference is
    survivable is that the enums never meet.
    """

    ANY = "any"
    ABSTRACT = "abstract"
    FULL_TEXT = "full-text"

    @property
    def label(self) -> str:
        """How the policy reads back to a person, in the manifest and the run log.

        Every one of them says *why* a bundle looks the way it does, because the two
        restricted policies produce a corpus that is smaller than the searches justify
        and a reader who does not know a filter ran will read that as a thin literature.
        """
        return {
            TextBasisPolicy.ANY: "full text where available, abstract otherwise",
            TextBasisPolicy.ABSTRACT: "abstracts only, by policy",
            TextBasisPolicy.FULL_TEXT: "open-access full text only, by policy",
        }[self]
