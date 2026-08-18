"""Writing the bundle. Code, not an agent: every decision here is already made.

The whole file is a rendering of things the pipeline decided earlier, which is why it
contains no judgment and no model call. Four choices in it are load-bearing:

**An effect size is printed only when verification kept it.** `PredictorRow.downgraded`
strips `effect` but keeps `effect_raw`, because the raw string is the evidence the check
was run against. Printing `effect_raw` unconditionally would put the exact number the
source text does not contain straight back into the bundle, undoing the check in the one
place a reader would ever see it. So a row whose magnitude was removed reads
`unverified`, and a row that never carried one reads `—`. The two are distinguished by
whether the raw string holds a number at all: prose like "not significant" is the
extractor's own words and no check ever touched it, while a stripped magnitude always
leaves digits behind.

**Nothing is deleted.** Writing over an existing bundle warns and overwrites the files it
owns; a stale topic directory from an earlier taxonomy is reported, not removed. A tool
that tidies up a directory it was pointed at is a tool that eventually tidies up the
wrong one.

**An empty topic still gets a directory and an index** saying so. An absent topic and a
topic that retained nothing are different claims about the literature, and collapsing
them loses the more interesting one.

**Verbatim quotes go below the table, numbered against its `#` column.** A quote is a
whole sentence from a paper and a table cell is not where a sentence is readable; keying
them to the row number keeps each one next to its numbers without making the table
unreadable in a terminal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from okf_loremaster import __version__
from okf_loremaster.interactions import fold_variable, interaction_rows, variable_rows
from okf_loremaster.okf.frontmatter import render, stamp
from okf_loremaster.okf.layout import (
    ABSTRACT_SECTION,
    BODY_SECTIONS,
    BOTTOM_LINE_SECTION,
    CATALOG_FILENAME,
    CAVEATS_SECTION,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    DOCUMENT_TYPE,
    INDEX_FILENAME,
    INTERACTION_COLUMNS,
    INTERACTION_SEPARATOR,
    INTERACTIONS_SECTION,
    LOG_FILENAME,
    NONE_CELL,
    NULL_FINDINGS_SECTION,
    PREDICTOR_COLUMNS,
    PREDICTOR_INDEX_TYPE,
    PREDICTORS_FILENAME,
    PREDICTORS_SECTION,
    QUOTE_LEAD,
    ROOT_INDEX_TYPE,
    SEARCH_FILENAME,
    SEARCH_STRATEGY_TYPE,
    SITE_COLUMNS,
    TOPIC_INDEX_TYPE,
    UNVERIFIED_CELL,
    VOCABULARY_SECTION,
)
from okf_loremaster.okf.markdown import facts, inline, table_row, table_rule
from okf_loremaster.recurrence import MIN_PAPERS, index_predictors
from okf_loremaster.schemas import (
    Charter,
    ConceptRecord,
    ExecutedQuery,
    Extraction,
    Interaction,
    NullFinding,
    OutcomeGroup,
    PaperStrength,
    PredictorGroup,
    PredictorRow,
    RowStrength,
    RunManifest,
    SourceRef,
    StrengthGrade,
    TextBasis,
    TextBasisPolicy,
)
from okf_loremaster.verification import quantities_in

__all__ = [
    "CONTESTED",
    "TOPIC_COLUMNS",
    "TOPIC_PREDICTORS",
    "BundleWrite",
    "body_for",
    "catalog_row",
    "descriptor",
    "document_for",
    "effect_cell",
    "frontmatter_for",
    "interaction_cell",
    "log_markdown",
    "predictor_index",
    "root_index",
    "search_markdown",
    "strength_cell",
    "topic_index",
    "write_bundle",
]

# Authors named in full before the line becomes "et al.". A paper with two hundred
# collaborators would otherwise put two hundred names on one frontmatter line.
MAX_AUTHORS = 6

LOG_TYPE = "Build Log"

_NULL_COLUMNS = ("#", "Predictor", "Outcome", "Detail")
# The browse table. `strength` sits beside `design` and `n` because it is the one column
# that reads the other two together — a reader choosing which of forty papers to open is
# asking exactly the question it answers.
TOPIC_COLUMNS = ("pmid", "title", "design", "n", "strength", "key predictors")

# How many predictor names the topic index shows per paper. A browse table, not a
# summary: enough to tell two papers apart, short enough that the column stays a column.
TOPIC_PREDICTORS = 3

# What `predictors.md` prints where papers disagree about the sign of a relationship.
# A marker rather than an adjudication: the file cannot say which paper is right, and the
# useful thing it can say is that the reader has to open both.
CONTESTED = "⚠ contested"


@dataclass(frozen=True, slots=True)
class BundleWrite:
    """What `write_bundle` did, for the node that called it to report."""

    path: Path
    documents: int = 0
    topics: int = 0
    files: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


def write_bundle(
    path: Path,
    *,
    records: Sequence[ConceptRecord],
    charter: Charter,
    manifest: RunManifest,
    log: str = "",
) -> BundleWrite:
    """Write a whole bundle: documents, topic indexes, root index, catalog, descriptor."""
    warnings: list[str] = []
    written: list[Path] = []

    existing = path / INDEX_FILENAME
    if existing.exists():
        warnings.append(
            f"{path} already holds a bundle; the files this run owns are overwritten and "
            f"anything else is left alone"
        )
    path.mkdir(parents=True, exist_ok=True)

    grouped = _by_topic(records, charter)
    for slug, topic_records in grouped.items():
        directory = path / slug
        directory.mkdir(parents=True, exist_ok=True)
        for record in topic_records:
            # Whether abstracts are written is read off the manifest rather than passed in
            # beside it. The manifest is what the bundle will claim about itself, so taking
            # both from one value is what makes it impossible for `abstracts: false` to sit
            # in the descriptor of a corpus whose documents all carry an `# Abstract`.
            written.append(
                _write(
                    directory / record.filename,
                    document_for(record, abstracts=manifest.abstracts),
                )
            )
        written.append(
            _write(directory / INDEX_FILENAME, topic_index(slug, topic_records, charter=charter))
        )

    for stale in _stale_directories(path, grouped):
        warnings.append(
            f"{stale.name}/ is not a topic in this charter and was left in place — "
            f"delete it by hand if it is from an earlier taxonomy"
        )

    written.append(
        _write(path / INDEX_FILENAME, root_index(grouped, charter=charter, manifest=manifest))
    )
    written.append(
        _write(
            path / CATALOG_FILENAME,
            "".join(
                json.dumps(catalog_row(record), ensure_ascii=False, sort_keys=True) + "\n"
                for topic_records in grouped.values()
                for record in topic_records
            ),
        )
    )
    written.append(_write(path / PREDICTORS_FILENAME, predictor_index(records, charter=charter)))
    written.append(
        _write(path / DESCRIPTOR_FILENAME, descriptor(grouped, charter=charter, manifest=manifest))
    )
    written.append(_write(path / LOG_FILENAME, log or log_markdown(charter, manifest)))
    written.append(_write(path / SEARCH_FILENAME, search_markdown(charter, manifest)))
    # Written here as well as by the run directory, so the descriptor's charter pointer
    # resolves for a bundle that was moved or copied on its own.
    written.append(_write(path / CHARTER_FILENAME, charter.to_yaml()))

    return BundleWrite(
        path=path,
        documents=sum(len(items) for items in grouped.values()),
        topics=len(grouped),
        files=tuple(written),
        warnings=tuple(warnings),
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _by_topic(records: Sequence[ConceptRecord], charter: Charter) -> dict[str, list[ConceptRecord]]:
    """Charter order first, then any topic a record claims that the charter does not.

    An empty topic keeps its entry. A record on an unknown topic keeps its file rather
    than being dropped: the file is real evidence and the taxonomy is the thing that
    drifted.
    """
    grouped: dict[str, list[ConceptRecord]] = {topic.slug: [] for topic in charter.topic_taxonomy}
    for record in records:
        grouped.setdefault(record.domain, []).append(record)
    return grouped


def _stale_directories(path: Path, grouped: Mapping[str, list[ConceptRecord]]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_dir() and not item.name.startswith((".", "_")) and item.name not in grouped
    )


# --- one document -----------------------------------------------------------


def document_for(record: ConceptRecord, *, abstracts: bool = True) -> str:
    return render(frontmatter_for(record)) + "\n" + body_for(record, abstracts=abstracts)


def frontmatter_for(record: ConceptRecord) -> dict[str, Any]:
    """The frontmatter block for one paper.

    `verified` is present only when a human signed off. Its absence is the spec's
    `unverified` tier, which is the honest tier for a machine extraction — a
    self-attestation written by the process that did the extracting discriminates
    nothing.
    """
    extraction = record.extraction
    fields: dict[str, Any] = {
        "type": DOCUMENT_TYPE,
        "title": record.title,
        "description": extraction.description,
        "resource": record.resource_url,
        "domain": record.domain,
        "id": record.pmid,
        "pmid": record.pmid,
        "journal": record.journal or record.journal_abbrev,
        "authors": _author_line(record.authors),
        "published": str(record.year) if record.year else "",
        "tags": list(extraction.tags),
        "study_design": extraction.study_design,
        "n": extraction.n,
        # Two flat keys rather than one nested block, though the block would render as
        # legal flow style. Flow style is reserved for the three structures OKF v0.2
        # actually nests — `generated`, `verified`, `sources` — and a downstream line
        # parser reads anything else nested as one opaque string it has to re-parse.
        # Grade to filter on, score to sort by; `parts` stays out, because frontmatter is
        # for choosing a document and the audit belongs in `# Bottom line`.
        **_strength_fields(record.strength),
        "text_basis": record.text_basis.value,
        "license": record.license,
        # A value, not a flag: `false` is written, because "we know this may not be
        # redistributed" is different from "nobody asked".
        "export_safe": record.export_safe,
        "generated": _generated(record),
        "sources": [_source(ref) for ref in record.sources],
    }
    if record.verified:
        fields["verified"] = [{"by": entry.by, "at": stamp(entry.at)} for entry in record.verified]
    return fields


def _strength_fields(strength: PaperStrength | None) -> dict[str, Any]:
    """`strength: moderate` and `strength_score: 0.58`, or neither.

    Nothing at all for an ungraded paper. A key reading `ungraded` is still a key every
    downstream filter has to special-case, and its absence says the same thing without
    asking anyone to.
    """
    if strength is None or not strength.graded:
        return {}
    return {"strength": strength.grade.value, "strength_score": round(strength.score, 2)}


def _generated(record: ConceptRecord) -> dict[str, str]:
    generated: dict[str, str] = {}
    if record.generated_by:
        generated["by"] = record.generated_by
    if record.generated_at is not None:
        generated["at"] = stamp(record.generated_at)
    return generated


def _source(ref: SourceRef) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": ref.id}
    if ref.resource:
        entry["resource"] = ref.resource
    if ref.last_modified:
        entry["last_modified"] = ref.last_modified
    if ref.usage_count is not None:
        entry["usage_count"] = ref.usage_count
    return entry


def _author_line(authors: Sequence[str]) -> str:
    if not authors:
        return ""
    named = list(authors[:MAX_AUTHORS])
    line = ", ".join(named)
    return f"{line}, et al." if len(authors) > MAX_AUTHORS else line


def body_for(record: ConceptRecord, *, abstracts: bool = True) -> str:
    """The document body, in `BODY_SECTIONS` order, every section written non-empty.

    Five sections are unconditional and each has a stated fallback — "None reported",
    "None stated" — because their absence is itself a finding a reader needs. `# Abstract`
    and `# Interactions` are the two that may be missing entirely: a heading over nothing
    is worse than no heading, and both are missing on ordinary papers rather than
    exceptional ones. Roughly one PubMed record in ten carries no abstract, and most
    papers state no interaction at all — the predictor table's `Interacts with` column is
    already where "none" gets said, blank, on every row.

    `abstracts=False` is `--no-abstract`: the section is dropped from every document
    whether or not the paper had one. It is a drop at write time and not an unread
    abstract — screening, curation and extraction all still ran on the same text, so the
    corpus is the corpus either way and only the documents are shorter. What the run did
    is on the manifest, because these files cannot say it: a missing heading looks the
    same as a paper PubMed served no abstract for.

    Order is fixed and skipping never disturbs it, which is what the validator checks: a
    document with five sections and one with seven agree on the order of the five they
    share.
    """
    extraction = record.extraction
    written = {
        BOTTOM_LINE_SECTION: _bottom_line(record),
        ABSTRACT_SECTION: _abstract(record) if abstracts else "",
        PREDICTORS_SECTION: _predictors(record),
        INTERACTIONS_SECTION: _interactions(extraction),
        NULL_FINDINGS_SECTION: _null_findings(extraction),
        VOCABULARY_SECTION: _vocabulary(extraction),
        CAVEATS_SECTION: extraction.caveats.strip() or "None stated.",
    }
    return (
        "\n\n".join(
            f"# {heading}\n\n{written[heading]}"
            for heading in BODY_SECTIONS
            if written[heading].strip()
        )
        + "\n"
    )


def _abstract(record: ConceptRecord) -> str:
    """`# Abstract` — the publisher's own words, copied and never summarized.

    Verbatim is what makes it worth having: an agent that has read the bottom line and
    wants the authors' own framing before the tables gets the framing rather than our
    compression of it, and nothing here can drift from the source because nothing here
    was rewritten. It is also the only block in a bundle we did not write, which is why
    `is_export_safe` already reads false for every abstract-only record — PubMed serves an
    abstract with no license attached, so redistributing a corpus of them is a decision
    for whoever redistributes it and not one this tool can make on their behalf.
    `--no-abstract` leaves it out, which takes about a fifth of the concept documents with
    it — 20% of their bytes measured across a 199-paper bundle, and up to 31% on a small
    one whose tables are short. It does not make a bundle redistributable: every predictor
    row's quote is sliced out of the paper too, and those stay.

    Structured abstracts arrive from E-utilities as `Label: body` paragraphs and are kept
    that way, because the Methods/Results distinction is exactly what a reader checking an
    extracted row against the source is looking for.

    A paragraph opening with `# ` is escaped. `reader.body_sections` splits on that
    sequence at the start of a line, so untrusted text carrying one would silently become
    a heading, and the document would then fail its own validator on a section nobody
    wrote. Rare enough to never see, cheap enough to make impossible.
    """
    paragraphs = [inline(part) for part in record.abstract.strip().split("\n\n")]
    return "\n\n".join(
        f"\\{part}" if part.startswith("# ") else part for part in paragraphs if part
    )


def _interactions(extraction: Extraction) -> str:
    """`# Interactions` — one line per claim, keyed to the predictor table's `#`.

    One line per interaction rather than one per row: a predictor standing in three
    relationships is making three claims, and merging them into a cell makes them one.

    Empty, and so omitted whole, when no row states an interaction. That is most papers.
    """
    rows = extraction.predictors
    by_name = variable_rows(rows)
    lines: list[str] = []
    for number, row in enumerate(rows, start=1):
        for interaction in interaction_rows(row):
            # 0 when the named variable is not a row here, which is what
            # `interaction_cell` reads as "no row to point back at".
            origin = by_name.get(fold_variable(interaction.feature), -1) + 1
            lines.append(
                table_row(
                    (
                        str(number),
                        row.predictor,
                        interaction.feature,
                        interaction.kind.label,
                        interaction.magnitude.value,
                        interaction_cell(interaction, stated_on=origin),
                    )
                )
            )
    if not lines:
        return ""
    return "\n".join([table_row(INTERACTION_COLUMNS), table_rule(len(INTERACTION_COLUMNS)), *lines])


def _bottom_line(record: ConceptRecord) -> str:
    extraction = record.extraction
    lines = [extraction.bottom_line.strip() or "Not stated."]
    pairs: list[tuple[str, str]] = []
    if extraction.study_design:
        pairs.append(("Design", extraction.study_design))
    if extraction.n is not None:
        pairs.append(("N", f"{extraction.n:,}"))
    if extraction.population:
        pairs.append(("Population", extraction.population))
    if extraction.outcome_definition:
        pairs.append(("Outcome", extraction.outcome_definition))
    pairs.append(
        (
            "Read from",
            "the full text" if record.text_basis is TextBasis.FULL_TEXT else "the abstract only",
        )
    )
    pairs.append(("Evidence strength", _strength_fact(record.strength)))
    lines.append("")
    lines.extend(facts(pairs))
    return "\n".join(lines)


def _strength_fact(strength: PaperStrength | None) -> str:
    """The paper's strength as a sentence, with what went unmeasured named.

    Spelled out rather than left as a bare `moderate 0.58`, because the number alone
    invites a reader to treat a score resting on one signal as if it rested on four.
    Naming the gaps is also what makes the score arguable: a paper marked down for saying
    nothing about adjustment can be checked against the paper.
    """
    if strength is None or not strength.graded:
        return ""
    line = f"{strength.grade.value} ({strength.score:.2f})"
    if strength.unmeasured:
        # Not "the paper did not say": `size` also goes unmeasured when the charter
        # carries no scale to measure it against, and blaming the paper for that would be
        # a claim about the paper that the score cannot support.
        line += f" — nothing to score on {_and_list(strength.unmeasured)}"
    return line


def _and_list(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" or {items[-1]}"


def _predictors(record: ConceptRecord) -> str:
    extraction = record.extraction
    rows = extraction.predictors
    if not rows:
        return "No predictor rows were extracted from this paper."
    # Positionally parallel, written together in `reconcile`. Padded rather than zipped
    # strictly: a record read back from an older bundle has no strength at all, and that
    # is a column of dashes, not a crash.
    scores = list(record.strength.rows) if record.strength is not None else []
    body = [
        table_row(PREDICTOR_COLUMNS),
        table_rule(len(PREDICTOR_COLUMNS)),
    ]
    for number, row in enumerate(rows, start=1):
        strength = scores[number - 1] if number <= len(scores) else None
        body.append(
            table_row(
                (
                    str(number),
                    row.predictor,
                    row.operationalization,
                    row.timing,
                    row.outcome,
                    row.evidence_type.label,
                    effect_cell(row),
                    row.p_value,
                    row.direction.value,
                    row.confidence.value,
                    strength_cell(strength),
                    _interacts_cell(row),
                )
            )
        )
    quotes = [(number, row.quote) for number, row in enumerate(rows, start=1) if row.quote.strip()]
    if quotes:
        body.append("")
        body.append(QUOTE_LEAD)
        body.append("")
        body.extend(f"{number}. {inline(quote)}" for number, quote in quotes)
    return "\n".join(body)


def effect_cell(row: PredictorRow) -> str:
    """What goes in the Effect column.

    Three outcomes, and the distinction between the last two is the whole point of the
    verification pass: a magnitude it kept, a magnitude it removed, and a row that never
    reported one.
    """
    if row.effect is not None:
        if row.effect_raw.strip():
            return row.effect_raw
        measure = f" {row.effect_measure}" if row.effect_measure else ""
        return f"{row.effect:g}{measure}"
    if not row.effect_raw.strip():
        return NONE_CELL
    # No numbers left to have been removed, so this is the extractor's own words about
    # a paper that reported no magnitude — not something a check took away.
    return row.effect_raw if not quantities_in(row.effect_raw) else UNVERIFIED_CELL


def _interacts_cell(row: PredictorRow) -> str:
    """The predictor table's `Interacts with` cell: names, and nothing else.

    A pointer at `# Interactions`, not a summary of it. The type and the coefficient are
    what a reader needs in order to act on an interaction, they do not fit in a cell
    beside ten other columns, and a table that answers the question is a table nobody
    scrolls past to the section that answers it properly.

    Blank — `NONE_CELL`, via `cell` — on most rows, which is the expected case rather than
    a gap. Interactions are rare, and a column of dashes says so honestly.
    """
    return INTERACTION_SEPARATOR.join(
        interaction.feature for interaction in interaction_rows(row) if interaction.feature
    )


def interaction_cell(interaction: Interaction, stated_on: int = 0) -> str:
    """What goes in the `# Interactions` table's Evidence column.

    The same three outcomes as `effect_cell`, distinguished the same way and for the same
    reason: a coefficient verification kept, a coefficient it removed, and a relationship
    the paper described without measuring. Collapsing the last two would hide exactly what
    the verification pass exists to expose.

    A mirrored line says so and points at the row that stated it, because the paper wrote
    the claim once and from one side. `stated_on` is that row's `#`, or 0 when the mirror
    came from a variable that is not itself a row here — which `mirror_interactions` makes
    impossible, and which is handled anyway rather than printed as `row 0`.
    """
    if interaction.value is not None:
        measure = f"{interaction.measure}=" if interaction.measure else ""
        base = interaction.measure_raw or f"{measure}{interaction.value:g}"
    elif not interaction.measure_raw.strip():
        base = NONE_CELL
    else:
        # No numbers left to have been removed, so this is the extractor's own words about
        # a relationship the paper never quantified — not something a check took away.
        base = (
            interaction.measure_raw
            if not quantities_in(interaction.measure_raw)
            else UNVERIFIED_CELL
        )
    if not interaction.mirrored:
        return base
    origin = f"mirrored from row {stated_on}" if stated_on > 0 else "mirrored"
    return origin if base == NONE_CELL else f"{base} ({origin})"


def strength_cell(strength: PaperStrength | RowStrength | None) -> str:
    """What goes in a Strength column: the grade, then the score that produced it.

    Both, because they answer different questions. The grade is what a person skims and
    is banded on purpose — the inputs do not support ranking 0.61 above 0.60. The number
    is what a downstream agent filters on, and printing it keeps the grade honest by
    showing where in its band a row actually sits.

    An ungraded row prints the empty cell. Nothing was measured, and `limited 0.50` would
    read as a finding about the study rather than as an absence of one.

    Takes either level: a paper and a row score the same way and read the same way, and
    the topic index shows the paper's while the predictor table shows each row's.
    """
    if strength is None or strength.grade is StrengthGrade.UNGRADED:
        return NONE_CELL
    return f"{strength.grade.value} {strength.score:.2f}"


def _null_findings(extraction: Extraction) -> str:
    reported = [finding for finding in extraction.null_findings if not finding.is_sentinel]
    if not reported:
        return "None reported — the paper states no null or non-significant finding."
    body = [table_row(_NULL_COLUMNS), table_rule(len(_NULL_COLUMNS))]
    for number, finding in enumerate(reported, start=1):
        body.append(table_row((str(number), finding.predictor, finding.outcome, finding.detail)))
    quotes = _null_quotes(reported)
    if quotes:
        body.append("")
        body.append(QUOTE_LEAD)
        body.append("")
        body.extend(f"{number}. {inline(quote)}" for number, quote in quotes)
    return "\n".join(body)


def _null_quotes(findings: Sequence[NullFinding]) -> list[tuple[int, str]]:
    return [
        (number, finding.quote)
        for number, finding in enumerate(findings, start=1)
        if finding.quote.strip()
    ]


def _vocabulary(extraction: Extraction) -> str:
    """`# Vocabulary hints` — each variable the paper named, then how it coded it.

    The concept leads and the codes follow on the same line, because the two belong to
    one variable: a reader matching on English finds it at the front, and one that can
    match on a code finds the code attached to the name rather than in a separate list
    it would have to re-associate by guessing.

    A concept with no codes is written plainly, with nothing after it. Most papers name
    their variables without ever coding them, and a trailing empty marker would make
    the common case look like a failure.
    """
    hints = [hint for hint in extraction.vocabulary_hints if hint.concept]
    if not hints:
        return "None recorded."
    lines = []
    for hint in hints:
        codes = ", ".join(f"{code.system} `{code.code}`" for code in hint.codes)
        rendered = f"- **{inline(hint.concept)}**"
        lines.append(f"{rendered} — {codes}" if codes else rendered)
    return "\n".join(lines)


# --- indexes ----------------------------------------------------------------


def topic_index(slug: str, records: Sequence[ConceptRecord], *, charter: Charter) -> str:
    topic = charter.topic(slug)
    title = topic.title if topic is not None else slug
    scope = topic.scope if topic is not None else ""
    fields: dict[str, Any] = {
        "type": TOPIC_INDEX_TYPE,
        "title": title,
        "description": scope or f"{len(records)} paper(s) on {title}.",
        "domain": slug,
        "domain_title": title,
    }

    body = [f"# {title}", ""]
    if scope:
        body += [scope, ""]
    if not records:
        body += [
            "No papers were retained on this topic. The searches ran and the screener "
            "saw candidates; none survived curation.",
            "",
            f"[← bundle index](../{INDEX_FILENAME})",
            "",
        ]
        return render(fields) + "\n" + "\n".join(body)

    full = sum(1 for record in records if record.text_basis is TextBasis.FULL_TEXT)
    body += [
        f"{len(records)} paper(s) — {full} read from full text, "
        f"{len(records) - full} from the abstract only.",
        "",
        table_row(TOPIC_COLUMNS),
        table_rule(len(TOPIC_COLUMNS)),
    ]
    for record in records:
        body.append(
            table_row(
                (
                    record.pmid,
                    # Escaping is left to `table_row`; escaping twice would turn a
                    # pipe in a title into a backslash and a column break.
                    f"[{record.title}](./{record.filename})",
                    record.extraction.study_design,
                    f"{record.extraction.n:,}" if record.extraction.n is not None else NONE_CELL,
                    strength_cell(record.strength),
                    _key_predictors(record),
                )
            )
        )
    body += ["", f"[← bundle index](../{INDEX_FILENAME})", ""]
    return render(fields) + "\n" + "\n".join(body)


def _key_predictors(record: ConceptRecord) -> str:
    names = [row.predictor for row in record.extraction.predictors[:TOPIC_PREDICTORS]]
    extra = len(record.extraction.predictors) - len(names)
    if not names:
        return NONE_CELL
    return ", ".join(names) + (f", +{extra} more" if extra > 0 else "")


def root_index(
    grouped: Mapping[str, list[ConceptRecord]], *, charter: Charter, manifest: RunManifest
) -> str:
    total = sum(len(items) for items in grouped.values())
    title = charter.task or charter.prompt
    fields: dict[str, Any] = {
        "type": ROOT_INDEX_TYPE,
        "title": title,
        "description": (
            f"{total} paper(s) across {len(grouped)} topic/topics, built from PubMed "
            f"and PMC by {manifest.tool_version or 'okf-loremaster'}."
        ),
    }

    body = [f"# {title}", "", inline(charter.prompt), "", "## Topics", ""]
    body.append(table_row(("topic", "title", "papers", "full text", "abstract only", "scope")))
    body.append(table_rule(6))
    for slug, records in grouped.items():
        topic = charter.topic(slug)
        full = sum(1 for record in records if record.text_basis is TextBasis.FULL_TEXT)
        body.append(
            table_row(
                (
                    f"[{slug}]({slug}/{INDEX_FILENAME})",
                    topic.title if topic is not None else "",
                    str(len(records)),
                    str(full),
                    str(len(records) - full),
                    topic.scope if topic is not None else "",
                )
            )
        )

    body += ["", "## Charter", ""]
    body += facts(
        [
            ("Population", charter.population),
            ("Outcome", charter.outcome),
            ("Languages", ", ".join(charter.languages) or "any"),
            (
                "Target",
                f"{charter.target_papers} papers, {charter.topic_paper_min}"
                f"-{charter.topic_paper_max} per topic",
            ),
            ("From", str(charter.min_year) if charter.min_year else ""),
            ("Full charter", f"[{CHARTER_FILENAME}]({CHARTER_FILENAME})"),
        ]
    )

    body += ["", "## Run", ""]
    body += facts(
        [
            ("Run id", manifest.run_id),
            ("Built", stamp(manifest.finished_at) if manifest.finished_at else ""),
            ("Duration", _duration(manifest)),
            ("Tool", manifest.tool_version),
            ("Charter digest", manifest.charter_digest),
            ("Models", ", ".join(f"{role}: {name}" for role, name in manifest.models.items())),
            # Beside the corpus funnel it explains. A run under a restricted basis drops
            # papers before screening, so the funnel's own numbers look like a thin
            # literature unless this line says a filter ran.
            ("Read from", _basis_policy(manifest)),
            ("Abstracts", _abstract_policy(manifest)),
            ("Stale after", manifest.stale_after.isoformat() if manifest.stale_after else ""),
            ("Signed off by", manifest.verified_by),
        ]
    )

    body += ["", "## Corpus", ""]
    body.append(table_row(("stage", "papers")))
    body.append(table_rule(2))
    for label, value in _funnel(manifest):
        body.append(table_row((label, f"{value:,}")))

    body += ["", "## Cost", ""]
    body += facts(
        [
            ("Model calls", f"{manifest.cost.calls:,}"),
            (
                "Tokens",
                f"{manifest.cost.tokens:,} ({manifest.cost.prompt_tokens:,} in / "
                f"{manifest.cost.completion_tokens:,} out)",
            ),
            ("Spend", manifest.cost.display),
        ]
    )

    if manifest.warnings:
        body += ["", "## Warnings", ""]
        body.extend(f"- {inline(warning)}" for warning in manifest.warnings)

    body += [
        "",
        "## Files",
        "",
        f"- [{PREDICTORS_FILENAME}]({PREDICTORS_FILENAME}) — what recurs across the "
        f"topics, and which row of which paper to read it in",
        f"- [{SEARCH_FILENAME}]({SEARCH_FILENAME}) — every query behind this bundle, why "
        f"it was asked, and what PubMed made of it",
        f"- [{LOG_FILENAME}]({LOG_FILENAME}) — how this bundle was built",
        f"- `{CATALOG_FILENAME}` — one JSON row per document",
        f"- [{DESCRIPTOR_FILENAME}]({DESCRIPTOR_FILENAME}) — what a consumer reads on attach",
        "",
    ]
    return render(fields) + "\n" + "\n".join(body)


def predictor_index(records: Sequence[ConceptRecord], *, charter: Charter) -> str:
    """`predictors.md` — what recurs across the corpus, and where to read each occurrence.

    The one file in the bundle that cuts across topics. It carries no `domain` key and
    cannot: a document's `domain` must equal the folder it sits in, and this sits at the
    root beside the folders rather than in one of them. That is what keeps it from being
    read as a paper filed nowhere.

    Every row is an address — a document, and the `#` of one row inside it — and nothing
    is written here that is not also written in a paper's own file. An index that can be
    read *instead of* the corpus is one that will be, and the quotes, operationalizations
    and provenance that justify the bundle would stop being opened.
    """
    index = index_predictors(records, effect_of=effect_cell)
    title = f"Predictors — {charter.task or charter.prompt}"
    fields: dict[str, Any] = {
        "type": PREDICTOR_INDEX_TYPE,
        "title": title,
        "description": (
            f"{index.predictors} predictor(s) reported by {MIN_PAPERS} or more of the "
            f"{index.papers} paper(s) in this bundle, each one pointing back at the rows "
            f"it was read from."
        ),
    }

    body = [
        f"# {title}",
        "",
        f"{index.rows:,} predictor row(s) across {index.papers} paper(s). "
        f"{index.predictors} of them recur across {MIN_PAPERS} or more papers and are "
        f"listed below. Another {index.once} appear in one paper each and are not listed — "
        f"that paper's own document already describes them in full, and its topic index "
        f"points at it.",
        "",
        "Every row here is a pointer, not a finding. `paper` is the file to open and "
        "`row` is the `#` to find in its `# Predictors reported` table, where the "
        "operationalization, the verbatim quote and the provenance live. Counts say how "
        "many documents to read, never how established a relationship is — this is a "
        "curated corpus, so how often something appears is a fact about the curation.",
        "",
        f"Grouped by predictor **and** outcome together. One paper can report one "
        f"exposure against several outcomes in several directions, and only the pairing "
        f"tells those apart from a disagreement. `{CONTESTED}` marks an outcome where the "
        f"papers do disagree about the sign.",
        "",
    ]

    if not index.groups:
        body += [
            f"No predictor is reported by {MIN_PAPERS} or more papers in this bundle. "
            f"That is a claim about the corpus rather than a missing section: with "
            f"{index.papers} paper(s) across {len(charter.topic_taxonomy) or 1} topic(s), "
            f"nothing recurred.",
            "",
            f"[← bundle index]({INDEX_FILENAME})",
            "",
        ]
        return render(fields) + "\n" + "\n".join(body)

    for group in index.groups:
        body += _predictor_section(group)

    body += [f"[← bundle index]({INDEX_FILENAME})", ""]
    return render(fields) + "\n" + "\n".join(body)


def _predictor_section(group: PredictorGroup) -> list[str]:
    lines = [f"## {inline(group.predictor)}", ""]
    summary = [
        f"{group.papers} paper(s)",
        f"{group.rows} row(s)",
        f"{len(group.topics)} topic(s): {', '.join(group.topics)}" if group.topics else "no topic",
    ]
    lines.append(" · ".join(summary))
    # The audit trail for the clustering, printed only when it actually merged something.
    # A merge nobody can see is a merge nobody can dispute, and this is a lexical match
    # that will occasionally be wrong in a way only a reader who knows the field can spot.
    if len(group.surface_forms) > 1:
        lines += ["", "Counted as one: " + " · ".join(f"*{form}*" for form in group.surface_forms)]
    lines.append("")
    for outcome in group.outcomes:
        lines += _outcome_section(outcome)
    return lines


def _outcome_section(outcome: OutcomeGroup) -> list[str]:
    heading = outcome.outcome or "outcome not recorded"
    lines = [f"### → {inline(heading)}", ""]
    # `increases (2)` rather than `2 increases`, so the enum value is printed exactly as
    # the `Direction` column of the document prints it and a reader can match the two by
    # eye — and so a count of one does not have to read as "1 increases".
    counts = " · ".join(f"{direction.value} ({count})" for direction, count in outcome.directions)
    line = f"{outcome.papers} paper(s) — {counts}"
    if outcome.contested:
        line += f"  {CONTESTED}"
    lines += [line, "", table_row(SITE_COLUMNS), table_rule(len(SITE_COLUMNS))]
    for site in outcome.sites:
        lines.append(
            table_row(
                (
                    # Linked on the filename rather than the bare PMID: it names the
                    # first author as well, which is how a reader recognizes a paper they
                    # have already opened, and the PMID is still its first token.
                    f"[{site.file.rsplit('/', 1)[-1].removesuffix('.md')}]({site.file})",
                    str(site.row),
                    site.domain,
                    # The paper's own words for the predictor, plus how it measured it.
                    # The heading is a cluster label and this is what was actually
                    # written, which is the difference between a group and a claim.
                    _as_measured(site.predictor, site.operationalization),
                    site.direction.value,
                    site.effect,
                    strength_cell(site.strength),
                )
            )
        )
    lines.append("")
    return lines


def _as_measured(predictor: str, operationalization: str) -> str:
    if not operationalization.strip():
        return predictor
    return f"{predictor} — {operationalization}"


def log_markdown(charter: Charter, manifest: RunManifest, *, verification: str = "") -> str:
    """`log.md` — how the bundle was built, in the order it happened."""
    fields: dict[str, Any] = {
        "type": LOG_TYPE,
        "title": f"Build log — {manifest.run_id}",
        "description": f"What ran, what it found, and what it cost for run {manifest.run_id}.",
    }

    body = [f"# Build log — {manifest.run_id}", "", "## Request", ""]
    body += facts(
        [
            ("Prompt", manifest.prompt or charter.prompt),
            ("Task", charter.task),
            ("Started", stamp(manifest.started_at) if manifest.started_at else ""),
            ("Finished", stamp(manifest.finished_at) if manifest.finished_at else ""),
            ("Duration", _duration(manifest)),
            # Under Request rather than under Stages, because it is something that was
            # asked for and not something that happened. What happened to each paper as a
            # result is the full-text-versus-abstract split in the stage counts.
            ("Read from", _basis_policy(manifest)),
            ("Abstracts", _abstract_policy(manifest)),
        ]
    )

    body += ["", "## Queries", ""]
    if manifest.queries:
        body.append(table_row(("#", "term", "hits", "retrieved", "note")))
        body.append(table_rule(5))
        for number, query in enumerate(manifest.queries, start=1):
            note = query.note
            if query.suspect:
                note = ("suspect — " + note) if note else "suspect: PubMed rewrote the term"
            body.append(
                table_row((str(number), query.term, f"{query.count:,}", str(query.retrieved), note))
            )
    else:
        body.append("No queries were executed.")

    body += ["", "## Stages", ""]
    body.append(table_row(("stage", "papers")))
    body.append(table_rule(2))
    for label, value in _funnel(manifest):
        body.append(table_row((label, f"{value:,}")))

    if verification:
        body += ["", "## Numeric verification", "", verification]

    body += ["", "## Cost", ""]
    body += facts(
        [
            ("Calls", f"{manifest.cost.calls:,}"),
            ("Tokens", f"{manifest.cost.tokens:,}"),
            ("Spend", manifest.cost.display),
        ]
    )

    body += ["", "## Warnings", ""]
    if manifest.warnings:
        body.extend(f"- {inline(warning)}" for warning in manifest.warnings)
    else:
        body.append("None.")
    body.append("")
    return render(fields) + "\n" + "\n".join(body)


def search_markdown(charter: Charter, manifest: RunManifest) -> str:
    """`search.md` — the search written out to be repeated by hand.

    `log.md` already lists the same queries, two lines each, as part of a run's
    forensics. This file exists because a methods section and a build log are read by
    different people looking for different things: one wants to know what happened in
    this run, the other wants to run the search again and is entitled to know what
    would come back differently and why. So everything here is spelled out — what each
    field is, which of them PubMed chose rather than us, and which of the three things
    that decide a replay are outside anyone's control.

    Nothing is computed that the run did not already record. `translation`, `rationale`,
    `topic` and `search_round` all ride on `ExecutedQuery`, and the retrieval parameters
    on the manifest, precisely so that this can be written after the plan is gone.
    """
    run = manifest.run_id or "this run"
    fields: dict[str, Any] = {
        "type": SEARCH_STRATEGY_TYPE,
        "title": f"Search strategy — {run}",
        "description": (
            "Every PubMed query this bundle was built from, why it was asked, and what "
            "PubMed actually ran — enough to repeat the search by hand."
        ),
    }

    body = [f"# Search strategy — {run}", ""]
    body += _search_preamble(charter, manifest)
    body += _search_how_to_read(manifest)
    body += _search_replay(manifest)
    body += _search_queries(charter, manifest)
    return render(fields) + "\n" + "\n".join(body)


def _search_preamble(charter: Charter, manifest: RunManifest) -> list[str]:
    """What ran, when, and against what — one paragraph before any detail."""
    queries = manifest.queries
    when = manifest.started_at.date().isoformat() if manifest.started_at else "an unrecorded date"
    counts = manifest.counts
    lines = [
        f"{_count(len(queries), 'query', 'queries')} ran against PubMed through NCBI's "
        f"E-utilities on {when}. Between them they matched "
        f"{sum(q.count for q in queries):,} papers, retrieved {counts.found:,}, and kept "
        f"{counts.unique:,} — the rest were the same paper found twice, retracted, or "
        "carried no abstract to screen. "
        f"Screening, curation and extraction narrowed those to the {counts.emitted:,} "
        "papers in this bundle; that part of the story is in "
        f"[{LOG_FILENAME}]({LOG_FILENAME}).",
        "",
        "No web page was scraped. Every search went to `esearch.fcgi` and every record "
        "came back from `efetch.fcgi`, so any term below can be repeated in a browser by "
        "pasting it into the PubMed search box.",
        "",
    ]

    applied = _shared_filters(charter)
    if applied:
        lines += [
            "Every query carries the same filters, appended after the model wrote the "
            f"term rather than folded into it: {applied}. Identical across the plan on "
            "purpose — it keeps them out of the way when you are reading what a query "
            "actually asks.",
            "",
        ]
    return lines


def _shared_filters(charter: Charter) -> str:
    """The language and date filters `queries.with_filters` appends, in English."""
    parts = []
    if charter.languages:
        codes = ", ".join(f"`{code}[la]`" for code in charter.languages)
        parts.append(f"{codes} for language")
    if charter.min_year:
        parts.append(
            f"`{charter.min_year}:3000[dp]` for anything published since {charter.min_year}"
        )
    return " and ".join(parts)


def _search_how_to_read(manifest: RunManifest) -> list[str]:
    """A key to the four fields under each query, and what each one is evidence of."""
    ordering = f"`sort={manifest.sort}`" if manifest.sort else "PubMed's default order"
    return [
        "## How to read a query",
        "",
        "- **Why** — what the query was reaching for. Written by the model that planned "
        "it, before it knew what it would find.",
        "- **Sent** — the term exactly as it left this tool. Copy it whole.",
        "- **PubMed ran** — what PubMed made of that term. It is checked for every "
        "query, because PubMed does not reject a field tag it does not recognize: it "
        'silently rewrites `x[nosuchfield]` into `"x"[All Fields]`, returns far more '
        "papers than intended, and reports no error at all. When the expansion only "
        "writes out tags the term already carried, this line says so in one sentence. "
        "When PubMed reached for a field or a MeSH heading the term did not ask for, "
        "the expansion is printed in full and the query is marked **suspect**.",
        "- **Result** — how many papers matched, and how many of them were actually "
        f"taken. Retrieval was capped at {manifest.retmax or 'no'} per query, ordered by "
        f"{ordering}, so a query that matched more than the cap contributed only the "
        "top slice of what it found.",
        "",
    ]


def _search_replay(manifest: RunManifest) -> list[str]:
    """What repeating this search will and will not give you.

    Stated rather than implied. A search strategy printed without its caveats reads as
    a promise of the same corpus, and two of the three things that decide that are
    nobody's to control.
    """
    cap = manifest.retmax or 0
    over = [q for q in manifest.queries if cap and q.count > cap]
    exact = len(manifest.queries) - len(over)
    lines = [
        "## Will this reproduce?",
        "",
        "The terms will. The corpus may not, and it is worth knowing why before you "
        "compare two runs.",
        "",
        "1. **The terms are exact.** They are stored as sent, and PubMed's query syntax "
        "is stable. Pasting one into PubMed today asks the same question it asked here.",
    ]
    if cap:
        whole = _count(exact, "query", "queries")
        capped = _count(len(over), "query", "queries")
        lines.append(
            f"2. **The cap is not a tie-breaker, it is a filter.** {whole} matched {cap} "
            f"papers or fewer and so were taken whole — those are exact. {capped} matched "
            f"more, and only the first {cap} were kept."
        )
        lines.append(
            "3. **PubMed's relevance ranking is not frozen.** It is recomputed as the "
            "index grows, so for the capped queries above, repeating the search later "
            "can return a different set of the same size. The uncapped ones are "
            "unaffected. This is the reason a bundle records the date it was searched "
            "rather than only the terms."
        )
    else:
        lines.append(
            "2. **PubMed keeps growing.** A query repeated later matches everything it "
            "matched before plus whatever has been indexed since, which is why a bundle "
            "records the date it was searched rather than only its terms."
        )
    lines.append("")
    return lines


def _search_queries(charter: Charter, manifest: RunManifest) -> list[str]:
    """One section per query, in the order they ran."""
    lines = ["## The queries", ""]
    if not manifest.queries:
        return [*lines, "No queries were executed.", ""]

    rounds = {query.search_round for query in manifest.queries if query.search_round}
    for number, query in enumerate(manifest.queries, start=1):
        heading = f"### {number}. {_query_heading(charter, query)}"
        if len(rounds) > 1 and query.search_round:
            heading += f" (round {query.search_round})"
        lines += [heading, ""]
        if query.rationale:
            lines += [f"**Why** — {inline(query.rationale)}", ""]
        lines += ["**Sent**", "", "```text", query.term, "```", ""]
        if _expansion_is_mechanical(query):
            lines += [
                "**PubMed ran** — the same term, with each field tag written out in full. "
                "Nothing was substituted, expanded or reinterpreted.",
                "",
            ]
        elif query.translation:
            lines += ["**PubMed ran**", "", "```text", query.translation, "```", ""]
        lines += [f"**Result** — {_search_result(query, manifest.retmax)}", ""]
        lines += _search_flags(query)
    return lines


_TAG = re.compile(r"\[([^\[\]]+)\]")

# PubMed's short field tags and the long names its expansion prints them as. Comparing the
# two forms is the only way to tell "the same query, spelled out" from "a different query".
_TAG_ALIASES = {
    "ab": "abstract",
    "au": "author",
    "dp": "date - publication",
    "la": "language",
    "mh": "mesh terms",
    "pt": "publication type",
    "sb": "filter",
    "ta": "journal",
    "ti": "title",
    "tiab": "title/abstract",
}

# The filters this tool appends after the planner writes a term. PubMed renders them
# inconsistently — `[Language]` in one response, `[Filter]` in the next — so they say
# nothing about whether the query itself was read as written.
_APPENDED_TAGS = frozenset({"date - publication", "filter", "language"})


def _tags(query_text: str) -> set[str]:
    """The field tags a query names, in PubMed's long form and without our own filters."""
    found = (raw.strip().lower() for raw in _TAG.findall(query_text))
    return {_TAG_ALIASES.get(tag, tag) for tag in found} - _APPENDED_TAGS


def _expansion_is_mechanical(query: ExecutedQuery) -> bool:
    """True when PubMed only spelled out tags the term already carried.

    A term written entirely in explicit field tags comes back as itself with `[tiab]`
    written `[Title/Abstract]`, and printing both is a few hundred characters of noise per
    query. What earns the space is the opposite case: an unrecognized tag becomes
    `[All Fields]` and an untagged word picks up `[MeSH Terms]`, neither of which the term
    asked for. So the test is whether PubMed introduced a tag, not whether the two strings
    differ — they always differ.
    """
    if not query.translation or query.suspect:
        return False
    return _tags(query.translation) <= _tags(query.term)


def _query_heading(charter: Charter, query: ExecutedQuery) -> str:
    """The topic a query was meant to fill, in the charter's words."""
    if not query.topic:
        return "Across the whole task"
    return _topic_title(charter, query.topic)


def _search_result(query: ExecutedQuery, retmax: int) -> str:
    """How many matched and how many were taken, said in English rather than two cells."""
    if query.count == 0:
        return (
            "no papers matched. Not an error — PubMed reports an empty search as a "
            "successful one — but a term that matches nothing usually has a phrase in it "
            "longer than any paper would print."
        )
    if query.retrieved >= query.count:
        return f"{query.count:,} papers matched, and all {query.retrieved:,} were retrieved."
    held = query.count - query.retrieved
    return (
        f"{query.count:,} papers matched. The first {query.retrieved:,} were retrieved"
        + (f" (the cap is {retmax:,})" if retmax else "")
        + f"; the other {held:,} were never seen by this run."
    )


def _search_flags(query: ExecutedQuery) -> list[str]:
    """Anything PubMed did to the query that a reader would otherwise have to notice."""
    lines: list[str] = []
    if query.suspect:
        lines += [
            f"> **Suspect.** {inline(query.note) or 'the expansion looks nothing like the term'}"
            " — read the expansion above before trusting this query's share of the corpus.",
            "",
        ]
    if query.fields_not_found:
        tags = ", ".join(f"`{tag}`" for tag in query.fields_not_found)
        # PubMed's own account rather than ours, which is why it is printed even when the
        # suspect line above already says the same thing in different words. The two come
        # from different places — one is a verdict we reached by comparing the term with
        # its expansion, the other is a list the service returned — and a reader deciding
        # whether to trust this query is better served by both than by the tidier one.
        lines += [
            f"> PubMed's own response reports {tags} as a field it does not have. It "
            "searched the clause as plain text instead, which is broader than what was "
            "asked for.",
            "",
        ]
    return lines


def _count(number: int, singular: str, plural: str) -> str:
    """`1 query` / `10 queries`, so a sentence reads right at either end."""
    return f"{number:,} {singular if number == 1 else plural}"


def _topic_title(charter: Charter, slug: str) -> str:
    """The human name of a topic, falling back to its slug.

    A topic can be in the bundle without being in the charter — an extraction may place a
    paper on a domain nobody planned — so the lookup has to survive a miss.
    """
    topic = charter.topic(slug)
    return topic.title if topic is not None else slug


def descriptor(
    grouped: Mapping[str, list[ConceptRecord]], *, charter: Charter, manifest: RunManifest
) -> str:
    """`resource_descriptor.yaml` — what a consumer reads before opening anything."""
    payload: dict[str, Any] = {
        "kind": "okf",
        "id": manifest.run_id,
        "name": charter.task or charter.prompt,
        "description": (
            f"{sum(len(items) for items in grouped.values())} curated papers from PubMed "
            f"and PMC, grouped into {len(grouped)} domains."
        ),
        "index": INDEX_FILENAME,
        "catalog": CATALOG_FILENAME,
        "log": LOG_FILENAME,
        "charter": CHARTER_FILENAME,
        # Declared beside the others so a consumer finds it without walking the root. OKF
        # v0.2 has a reader ignore keys it does not know, so this costs nothing to a
        # consumer that has never heard of it and saves a directory listing to one that has.
        "predictors": PREDICTORS_FILENAME,
        # Declared for the same reason, and worth its own key rather than a mention in
        # the log: a consumer asking "where did this corpus come from" is asking a
        # question about method, and the answer should not require reading a build log.
        "search": SEARCH_FILENAME,
        "domains": {slug: _topic_title(charter, slug) for slug in grouped},
        "documents": sum(len(items) for items in grouped.values()),
        "tool": "okf-loremaster",
        "tool_version": manifest.tool_version or __version__,
        "charter_digest": manifest.charter_digest,
    }
    if manifest.finished_at is not None:
        payload["built_on"] = manifest.finished_at.date().isoformat()
    if manifest.stale_after is not None:
        payload["stale_after"] = manifest.stale_after.isoformat()
    if manifest.verified_by:
        payload["verified_by"] = manifest.verified_by
    # Only when a policy narrowed the corpus. A consumer attaching this bundle is deciding
    # how much to trust it, and "every document here is abstract-only because we asked for
    # that" is a different answer to that question than "because that is what was open
    # access" — which is all the per-document `text_basis` can say.
    if _basis_policy(manifest):
        payload["text_basis_policy"] = manifest.text_basis_policy
    # Only when they were dropped, and for the same reason. A consumer that indexes the
    # prose of these documents is entitled to know that the authors' own framing is not
    # in them, and cannot find that out by looking: `# Abstract` is an optional heading,
    # so its absence everywhere reads as an unlucky corpus rather than a decision.
    if not manifest.abstracts:
        payload["abstracts"] = False
    return str(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))


def catalog_row(record: ConceptRecord) -> dict[str, Any]:
    """One `_catalog.jsonl` line."""
    row: dict[str, Any] = {
        "pmid": record.pmid,
        "title": record.title,
        "domain": record.domain,
        "file": f"{record.domain}/{record.filename}",
        "description": record.extraction.description,
        "design": record.extraction.study_design,
        "n": record.extraction.n,
        "tags": list(record.extraction.tags),
    }
    # Two flat keys rather than a nested one, because the catalog is what gets loaded into
    # a dataframe or filtered with `jq`, and both are easier on `strength` than on
    # `strength.grade`. Absent for an ungraded paper, matching the frontmatter.
    if record.strength is not None and record.strength.graded:
        row["strength"] = record.strength.grade.value
        row["strength_score"] = round(record.strength.score, 2)
    return row


# --- rendering helpers ------------------------------------------------------


def _basis_policy(manifest: RunManifest) -> str:
    """What the run was told to read, in English, or nothing at all.

    Empty for a bundle built before `--basis` existed and for one that took what it
    could get, and empty is right in both cases: `any` is the absence of a restriction,
    and a line saying so on every default bundle would train readers to skip the one
    place the two deliberate policies announce themselves.
    """
    try:
        policy = TextBasisPolicy(manifest.text_basis_policy)
    except ValueError:
        return ""
    return "" if policy is TextBasisPolicy.ANY else policy.label


def _abstract_policy(manifest: RunManifest) -> str:
    """What `--no-abstract` did, or nothing at all.

    Empty on a default run, for the reason `_basis_policy` is empty on one: a line saying
    "yes, the normal thing happened" on every bundle ever built teaches readers to skip
    the place the one deliberate choice announces itself.
    """
    if manifest.abstracts:
        return ""
    return (
        "omitted from every document by --no-abstract. Every paper was still screened "
        "and read from the text this run had; only the section is gone."
    )


def _funnel(manifest: RunManifest) -> list[tuple[str, int]]:
    counts = manifest.counts
    return [
        ("found", counts.found),
        ("unique", counts.unique),
        ("screened", counts.screened),
        ("included", counts.included),
        ("curated", counts.curated),
        ("full text fetched", counts.full_text_fetched),
        ("extracted", counts.extracted),
        ("emitted", counts.emitted),
    ]


def _duration(manifest: RunManifest) -> str:
    seconds = manifest.duration_seconds
    if seconds is None:
        return ""
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder}s" if minutes else f"{remainder}s"
