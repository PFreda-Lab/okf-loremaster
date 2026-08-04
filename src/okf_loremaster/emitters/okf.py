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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from okf_loremaster import __version__
from okf_loremaster.okf.frontmatter import render, stamp
from okf_loremaster.okf.layout import (
    BODY_SECTIONS,
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    DOCUMENT_TYPE,
    INDEX_FILENAME,
    LOG_FILENAME,
    NONE_CELL,
    PREDICTOR_COLUMNS,
    QUOTE_LEAD,
    ROOT_INDEX_TYPE,
    TOPIC_INDEX_TYPE,
    UNVERIFIED_CELL,
)
from okf_loremaster.okf.markdown import facts, inline, table_row, table_rule
from okf_loremaster.schemas import (
    Charter,
    ConceptRecord,
    Extraction,
    NullFinding,
    PredictorRow,
    RunManifest,
    SourceRef,
    TextBasis,
)
from okf_loremaster.verification import quantities_in

__all__ = [
    "TOPIC_COLUMNS",
    "TOPIC_PREDICTORS",
    "BundleWrite",
    "body_for",
    "catalog_row",
    "descriptor",
    "document_for",
    "effect_cell",
    "frontmatter_for",
    "log_markdown",
    "root_index",
    "topic_index",
    "write_bundle",
]

# Authors named in full before the line becomes "et al.". A paper with two hundred
# collaborators would otherwise put two hundred names on one frontmatter line.
MAX_AUTHORS = 6

LOG_TYPE = "Build Log"

_NULL_COLUMNS = ("#", "Predictor", "Outcome", "Detail")
TOPIC_COLUMNS = ("pmid", "title", "design", "n", "key predictors")

# How many predictor names the topic index shows per paper. A browse table, not a
# summary: enough to tell two papers apart, short enough that the column stays a column.
TOPIC_PREDICTORS = 3


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
            written.append(_write(directory / record.filename, document_for(record)))
        written.append(
            _write(directory / INDEX_FILENAME, topic_index(slug, topic_records, charter=charter))
        )

    for stale in _stale_directories(path, grouped):
        warnings.append(
            f"{stale.name}/ is not a topic in this charter and was left in place — "
            f"delete it by hand if it is from an earlier taxonomy"
        )

    written.append(_write(path / INDEX_FILENAME, root_index(grouped, charter=charter,
                                                            manifest=manifest)))
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
    written.append(_write(path / DESCRIPTOR_FILENAME, descriptor(grouped, charter=charter,
                                                                 manifest=manifest)))
    written.append(_write(path / LOG_FILENAME, log or log_markdown(charter, manifest)))
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


def _by_topic(
    records: Sequence[ConceptRecord], charter: Charter
) -> dict[str, list[ConceptRecord]]:
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
        if item.is_dir()
        and not item.name.startswith((".", "_"))
        and item.name not in grouped
    )


# --- one document -----------------------------------------------------------


def document_for(record: ConceptRecord) -> str:
    return render(frontmatter_for(record)) + "\n" + body_for(record)


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
        "text_basis": record.text_basis.value,
        "license": record.license,
        # A value, not a flag: `false` is written, because "we know this may not be
        # redistributed" is different from "nobody asked".
        "export_safe": record.export_safe,
        "generated": _generated(record),
        "sources": [_source(ref) for ref in record.sources],
    }
    if record.verified:
        fields["verified"] = [
            {"by": entry.by, "at": stamp(entry.at)} for entry in record.verified
        ]
    return fields


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


def body_for(record: ConceptRecord) -> str:
    """The five sections, in order, every one of them non-empty."""
    extraction = record.extraction
    sections = (
        _bottom_line(record),
        _predictors(extraction),
        _null_findings(extraction),
        _vocabulary(extraction),
        extraction.caveats.strip() or "None stated.",
    )
    return "\n\n".join(
        f"# {heading}\n\n{text}" for heading, text in zip(BODY_SECTIONS, sections, strict=True)
    ) + "\n"


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
    lines.append("")
    lines.extend(facts(pairs))
    return "\n".join(lines)


def _predictors(extraction: Extraction) -> str:
    rows = extraction.predictors
    if not rows:
        return "No predictor rows were extracted from this paper."
    body = [
        table_row(PREDICTOR_COLUMNS),
        table_rule(len(PREDICTOR_COLUMNS)),
    ]
    for number, row in enumerate(rows, start=1):
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


def _null_findings(extraction: Extraction) -> str:
    reported = [finding for finding in extraction.null_findings if not finding.is_sentinel]
    if not reported:
        return "None reported — the paper states no null or non-significant finding."
    body = [table_row(_NULL_COLUMNS), table_rule(len(_NULL_COLUMNS))]
    for number, finding in enumerate(reported, start=1):
        body.append(
            table_row((str(number), finding.predictor, finding.outcome, finding.detail))
        )
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


def topic_index(
    slug: str, records: Sequence[ConceptRecord], *, charter: Charter
) -> str:
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
            ("Target", f"{charter.target_papers} papers, {charter.topic_min}"
                       f"-{charter.topic_max} per topic"),
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
        f"- [{LOG_FILENAME}]({LOG_FILENAME}) — how this bundle was built",
        f"- `{CATALOG_FILENAME}` — one JSON row per document",
        f"- [{DESCRIPTOR_FILENAME}]({DESCRIPTOR_FILENAME}) — what a consumer reads on attach",
        "",
    ]
    return render(fields) + "\n" + "\n".join(body)


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
                table_row(
                    (str(number), query.term, f"{query.count:,}", str(query.retrieved), note)
                )
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
    return row


# --- rendering helpers ------------------------------------------------------


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
