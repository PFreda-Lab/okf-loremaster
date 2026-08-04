"""What is actually in a bundle, read off the bundle.

`validate` answers "is this well formed"; this answers "what did the run get you", and
they are different questions asked of the same directory. Nothing here re-derives
anything from a run: a bundle handed over on a USB stick summarizes the same as the one
that was just built, which is the only version of this command worth having.

Two sources, deliberately:

**The catalog is the spine.** Topic sizes, designs, sample sizes and tags come from
`_catalog.jsonl`, because that is the file a downstream consumer reads and summarizing
something other than what it sees would be summarizing the wrong thing. When the catalog
disagrees with the disk, that is reported rather than reconciled.

**The documents carry what the catalog does not.** `text_basis`, `export_safe`, and the
state of every effect size are only in the files. The last of those is the point: the
count of magnitudes that numeric verification kept versus removed is the one number that
says how much of this corpus can be quoted with a number attached, and it exists nowhere
but in the `Effect` column of each table.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from okf_loremaster.okf.layout import (
    BODY_SECTIONS,
    CATALOG_FILENAME,
    DESCRIPTOR_FILENAME,
    PREDICTOR_COLUMNS,
    UNVERIFIED_CELL,
    vector_store_path,
)
from okf_loremaster.okf.reader import OkfBundle, fact_list, markdown_table, read_bundle

__all__ = ["BundleOverview", "TopicOverview", "read_overview"]

_PREDICTORS_SECTION = BODY_SECTIONS[1]
_NULLS_SECTION = BODY_SECTIONS[2]
_VOCABULARY_SECTION = BODY_SECTIONS[3]
_EFFECT = PREDICTOR_COLUMNS[6]

# How many designs and vocabulary keys a summary shows before it stops being a summary.
TOP_DESIGNS = 8
TOP_VOCABULARIES = 12


@dataclass(frozen=True, slots=True)
class TopicOverview:
    slug: str
    title: str
    documents: int = 0
    full_text: int = 0
    exportable: int = 0
    predictors: int = 0

    @property
    def abstract_only(self) -> int:
        return self.documents - self.full_text


@dataclass(frozen=True, slots=True)
class BundleOverview:
    """Everything `okf-loremaster inspect` prints, and nothing it computes twice."""

    path: Path
    name: str = ""
    resource_id: str = ""
    documents: int = 0
    topics: tuple[TopicOverview, ...] = ()

    # From the documents.
    full_text: int = 0
    exportable: int = 0
    untagged: int = 0
    predictors: int = 0
    with_effect: int = 0
    unverified: int = 0
    reporting_nulls: int = 0

    # From the catalog.
    catalog_rows: int = 0
    designs: tuple[tuple[str, int], ...] = ()
    sample_sizes: tuple[int, ...] = ()
    vocabularies: tuple[tuple[str, int], ...] = ()

    index_facts: dict[str, str] = field(default_factory=dict)
    descriptor: dict[str, Any] = field(default_factory=dict)
    vectors: dict[str, Any] = field(default_factory=dict)
    problems: tuple[tuple[Path, str], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def abstract_only(self) -> int:
        return self.documents - self.full_text

    @property
    def has_vectors(self) -> bool:
        return bool(self.vectors)

    @property
    def median_n(self) -> int | None:
        """The middle reported sample size. A mean would be a single trial's outlier."""
        values = sorted(self.sample_sizes)
        if not values:
            return None
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) // 2


def read_overview(path: Path) -> BundleOverview:
    """Read a bundle and summarize it. Raises only for a bundle that is not there."""
    bundle = read_bundle(path)
    notes: list[str] = []

    topics: list[TopicOverview] = []
    full_text = exportable = untagged = 0
    predictors = with_effect = unverified = reporting_nulls = 0
    vocabulary: Counter[str] = Counter()

    for topic in bundle.topics:
        topic_full = topic_export = topic_rows = 0
        for document in topic.documents:
            rows = markdown_table(document.section(_PREDICTORS_SECTION) or "")
            topic_rows += len(rows)
            for row in rows:
                effect = row.get(_EFFECT, "").strip()
                if effect == UNVERIFIED_CELL:
                    unverified += 1
                elif effect:
                    with_effect += 1
            if markdown_table(document.section(_NULLS_SECTION) or ""):
                reporting_nulls += 1
            # Keys, not the mapping. `Counter.update` on a dict adds its *values*, which
            # for a fact list are code strings — a count that reads "A00A00A00".
            vocabulary.update(fact_list(document.section(_VOCABULARY_SECTION) or "").keys())
            if document.full_text:
                topic_full += 1
            if document.export_safe:
                topic_export += 1
            if not document.tags:
                untagged += 1
        predictors += topic_rows
        full_text += topic_full
        exportable += topic_export
        topics.append(
            TopicOverview(
                slug=topic.slug,
                title=topic.title or topic.slug,
                documents=len(topic.documents),
                full_text=topic_full,
                exportable=topic_export,
                predictors=topic_rows,
            )
        )

    designs, sample_sizes = _from_catalog(bundle, notes)
    return BundleOverview(
        path=bundle.path,
        name=_name(bundle),
        resource_id=str(bundle.descriptor.get("id") or ""),
        documents=bundle.document_count,
        topics=tuple(topics),
        full_text=full_text,
        exportable=exportable,
        untagged=untagged,
        predictors=predictors,
        with_effect=with_effect,
        unverified=unverified,
        reporting_nulls=reporting_nulls,
        catalog_rows=len(bundle.catalog),
        designs=designs,
        sample_sizes=sample_sizes,
        vocabularies=tuple(vocabulary.most_common(TOP_VOCABULARIES)),
        index_facts=fact_list(bundle.index.body) if bundle.index is not None else {},
        descriptor=dict(bundle.descriptor),
        vectors=_vectors(bundle),
        problems=bundle.problems,
        notes=tuple(notes),
    )


def _from_catalog(
    bundle: OkfBundle, notes: list[str]
) -> tuple[tuple[tuple[str, int], ...], tuple[int, ...]]:
    """Designs and sample sizes as a consumer reads them, with the disk cross-checked."""
    if not bundle.has_catalog:
        notes.append(f"no {CATALOG_FILENAME} — designs and sample sizes read from the documents")
        designs = Counter(
            str(document.fields.get("study_design") or "").strip()
            for document in bundle.documents()
        )
        sizes = [
            document.n for document in bundle.documents() if document.n is not None
        ]
    else:
        listed = {str(row.get("file") or "") for row in bundle.catalog}
        present = {f"{document.topic}/{document.filename}" for document in bundle.documents()}
        if listed != present:
            notes.append(
                f"{CATALOG_FILENAME} and the topics disagree about "
                f"{len(listed ^ present)} file(s) — run `okf-loremaster validate` for which"
            )
        designs = Counter(str(row.get("design") or "").strip() for row in bundle.catalog)
        sizes = [
            int(row["n"])
            for row in bundle.catalog
            if isinstance(row.get("n"), (int, float)) and not isinstance(row.get("n"), bool)
        ]
    del designs[""]
    return tuple(designs.most_common(TOP_DESIGNS)), tuple(sizes)


def _name(bundle: OkfBundle) -> str:
    named = str(bundle.descriptor.get("name") or "")
    if named:
        return named
    return (bundle.index.title if bundle.index is not None else "") or bundle.path.name


def _vectors(bundle: OkfBundle) -> dict[str, Any]:
    """The sidecar store's descriptor, or nothing. Never opens the store itself.

    The bundle's own `vectors` pointer is merged in underneath, because the chunk count
    is written there and nowhere else — but only underneath: the store describes itself,
    and a pointer left behind by an earlier index is the stale half of the pair.
    """
    descriptor = vector_store_path(bundle.path) / DESCRIPTOR_FILENAME
    if not descriptor.exists():
        return {}
    try:
        loaded = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        loaded = None
    if not isinstance(loaded, dict):
        return {}
    pointer = bundle.descriptor.get("vectors")
    return {**pointer, **loaded} if isinstance(pointer, dict) else loaded
