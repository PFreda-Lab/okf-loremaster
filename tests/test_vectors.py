"""The vector index, checked against the bundle it was derived from.

The step 8 gate is `test_the_store_declares_the_resolved_model_and_the_distance_metric`
plus `test_no_chunk_metadata_value_is_ever_null`: a store a consumer cannot attach, or
one Chroma refuses a row of, is a store that fails at the far end where nobody is
watching.

Everything here starts from the same golden bundle the emitter tests use, read back off
disk. Chunking a `ConceptRecord` would test the pipeline's opinion of a paper; chunking
the file tests what a retriever will actually be handed.

The embedder is a stub. The real one downloads a checkpoint on first use, and tests never
reach the network — which is also the reason `Embedder` is a protocol injected into
`build_index` rather than something the node constructs for itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from okf_loremaster.cli import app
from okf_loremaster.emitters.vectors import (
    COLLECTION,
    DISTANCE,
    REQUIRED_KEYS,
    ROW_LEVEL_KEYS,
    Chunk,
    build_index,
    chunks_for,
    index_descriptor,
)
from okf_loremaster.okf.layout import DESCRIPTOR_FILENAME, vector_store_path
from okf_loremaster.okf.reader import markdown_table, read_bundle
from okf_loremaster.okf.validate import Severity, validate_bundle

from graph_runs import TARGET, full_run

runner = CliRunner()

# Small and fixed. The dimension is never the thing under test, and a real one would cost
# a model download.
DIMENSIONS = 8


class StubEmbedder:
    """Deterministic vectors from the text, with a model id and a resolved revision.

    Deterministic rather than random so a test can assert that a given chunk got a given
    vector, and so a rebuild produces the same store — which is the property the pinned
    revision is supposed to buy in production.
    """

    def __init__(self, *, model: str = "stub/embedder", revision: str = "a" * 40) -> None:
        self._model = model
        self._revision = revision
        self.batches: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [
            [((hash(text) >> (bit * 4)) % 97) / 97 for bit in range(DIMENSIONS)] for text in texts
        ]


class RecordingStore:
    """A store that keeps what it was given instead of writing anything.

    Lets the metadata contract be asserted exactly as the store receives it, without a
    database in between deciding what it will and will not accept.
    """

    name = COLLECTION
    distance = DISTANCE
    replaced = False

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: list[Sequence[float]] = []

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def count(self) -> int:
        return len(self.chunks)


async def golden(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> Path:
    run = await full_run(settings_factory, tmp_path, monkeypatch, **overrides)
    return Path(run.state["bundle"])


async def indexed(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RecordingStore, Any]:
    """A golden bundle indexed into a recording store, with its result."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    store = RecordingStore()
    result = await build_index(bundle, embedder=StubEmbedder(), store=store)
    return bundle, store, result


# --- chunking ---------------------------------------------------------------


async def test_a_document_yields_one_concept_chunk_and_one_per_predictor_row(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = await golden(settings_factory, tmp_path, monkeypatch)

    rows_seen = 0
    for document in read_bundle(bundle).documents():
        chunks = chunks_for(document, root=bundle)
        table = markdown_table(document.section("Predictors reported") or "")
        assert len(chunks) == 1 + len(table), document.path.name
        assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
        assert chunks[0].level == "concept"
        assert all(chunk.level == "predictor" for chunk in chunks[1:])
        rows_seen += len(table)

    assert rows_seen, "no paper reported a predictor, so nothing was proved"


async def test_the_concept_chunk_leaves_the_table_to_the_row_chunks(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole document would overrun a sentence encoder's window and lose its tail in
    silence. The table is the largest section and the one already covered row by row."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for document in read_bundle(bundle).documents():
        concept = chunks_for(document, root=bundle)[0]
        assert "|" not in concept.text, document.path.name
        # But everything else about the paper is in it.
        assert document.title in concept.text
        for heading in ("Vocabulary hints", "Caveats", "Null or non-significant findings"):
            assert heading in concept.text, document.path.name


async def test_a_row_chunk_stands_alone(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieved on its own, a row is useless without the population it was measured in
    and the outcome it was measured against."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)

    checked = 0
    for document in read_bundle(bundle).documents():
        section = document.section("Predictors reported") or ""
        table = markdown_table(section)
        for row, chunk in zip(table, chunks_for(document, root=bundle)[1:], strict=True):
            assert document.title in chunk.text
            assert row["Predictor"] in chunk.text
            assert "Population:" in chunk.text
            checked += 1
    assert checked


async def test_a_row_chunk_carries_the_quote_keyed_to_its_number(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quote is the paper's own words and the reason a reader trusts the row. It is
    numbered against the table's `#` column, so a chunker that lost the numbering would
    attach the wrong sentence to the wrong predictor."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    document = next(iter(read_bundle(bundle).documents()))
    section = document.section("Predictors reported") or ""

    quoted = [
        line.strip() for line in section.splitlines() if line.strip()[:2].rstrip(".").isdigit()
    ]
    assert quoted, "the golden bundle carries no numbered quotes"

    chunks = chunks_for(document, root=bundle)
    for line in quoted:
        number, _, text = line.partition(". ")
        assert text in chunks[int(number)].text


# --- metadata ---------------------------------------------------------------


async def test_no_chunk_metadata_value_is_ever_null(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half the step 8 gate. Chroma raises `TypeError` on a null metadata value, so a
    missing field has to be `""` — checked over every chunk of a real bundle rather than
    over a document chosen for having every field."""
    _, store, _ = await indexed(settings_factory, tmp_path, monkeypatch)
    assert store.chunks

    for chunk in store.chunks:
        for key, value in chunk.metadata.items():
            assert value is not None, f"{chunk.id}: {key}"
            assert isinstance(value, str | int | float | bool), f"{chunk.id}: {key}={value!r}"


async def test_every_chunk_carries_the_four_keys_a_consumer_requires(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store, _ = await indexed(settings_factory, tmp_path, monkeypatch)

    ids = set()
    for chunk in store.chunks:
        for key in REQUIRED_KEYS:
            assert key in chunk.metadata, f"{chunk.id}: {key}"
        # `chunk_index` is legitimately 0 on a concept chunk; the other three are not.
        assert chunk.metadata["source"] and chunk.metadata["title"] and chunk.metadata["id"]
        assert (bundle_relative := str(chunk.metadata["source"])).endswith(".md"), bundle_relative
        ids.add(chunk.id)
    assert len(ids) == len(store.chunks), "two chunks share an id and would overwrite"


async def test_a_concept_chunk_carries_empty_strings_for_the_row_level_keys(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the gate. `timing`, `confidence` and `evidence_type` describe a
    predictor row; a filter on any of them that did not allow `""` would quietly drop
    every concept chunk, which is a paper's whole identity."""
    _, store, _ = await indexed(settings_factory, tmp_path, monkeypatch)

    concepts = [chunk for chunk in store.chunks if chunk.level == "concept"]
    rows = [chunk for chunk in store.chunks if chunk.level == "predictor"]
    assert len(concepts) == TARGET
    assert rows

    for chunk in concepts:
        for key in ROW_LEVEL_KEYS:
            assert chunk.metadata[key] == "", chunk.id
    # And the row chunks really do carry them, or the empty string would mean nothing.
    assert any(chunk.metadata["confidence"] for chunk in rows)
    assert any(chunk.metadata["timing"] for chunk in rows)


async def test_n_is_an_integer_where_the_paper_reported_one(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A numeric metadata filter needs a number. `""` rather than `0` where there is
    none: a study of zero participants is a different claim from one that never said."""
    _, store, _ = await indexed(settings_factory, tmp_path, monkeypatch)

    counts = [chunk.metadata["n"] for chunk in store.chunks]
    assert any(isinstance(value, int) and value > 0 for value in counts)
    assert all(isinstance(value, int) or value == "" for value in counts)


# --- the descriptors --------------------------------------------------------


async def test_the_store_declares_the_resolved_model_and_the_distance_metric(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 8 gate. Chroma's default space is L2; a consumer that assumed cosine gets
    a different order and no error, so the metric is declared rather than inferred. The
    revision is the *resolved* one, because `main` is a moving pointer."""
    bundle, _, result = await indexed(settings_factory, tmp_path, monkeypatch)

    descriptor = yaml.safe_load(
        (vector_store_path(bundle) / DESCRIPTOR_FILENAME).read_text(encoding="utf-8")
    )
    assert descriptor["kind"] == "rag"
    assert descriptor["embedding_model"] == "stub/embedder"
    assert descriptor["embedding_revision"] == "a" * 40
    assert descriptor["distance"] == DISTANCE
    assert descriptor["dimensions"] == DIMENSIONS
    assert descriptor["store"] == {"type": "chroma", "path": ".", "collection": COLLECTION}
    # The four keys a consumer requires, mapped from what we actually wrote.
    assert set(descriptor["metadata_map"]) == set(REQUIRED_KEYS)
    assert set(descriptor["metadata_keys"]) >= set(REQUIRED_KEYS) | set(ROW_LEVEL_KEYS)

    # The `""`-on-a-concept-chunk rule is stated where a consumer will read it. Without
    # it a metadata filter silently excludes half the corpus and looks like it found less.
    for key in ROW_LEVEL_KEYS:
        assert key in descriptor["notes"]
    assert descriptor["description"].startswith(f"{result.chunks} chunks over {TARGET} papers")


async def test_the_bundle_points_at_its_index_and_the_index_at_its_bundle(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer opens the bundle first. An index nothing points at is an index nobody
    attaches."""
    bundle, _, result = await indexed(settings_factory, tmp_path, monkeypatch)

    descriptor = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    pointer = descriptor["vectors"]
    assert (bundle / pointer["path"]).resolve() == result.path.resolve()
    assert pointer["embedding_model"] == "stub/embedder"
    assert pointer["distance"] == DISTANCE
    assert pointer["chunks"] == result.chunks
    # The bundle's own keys survived the patch.
    assert descriptor["kind"] == "okf"

    index_side = yaml.safe_load(
        (result.path / DESCRIPTOR_FILENAME).read_text(encoding="utf-8")
    )
    assert (result.path / index_side["source_bundle"]).resolve() == bundle.resolve()


async def test_the_store_is_a_sibling_of_the_bundle_and_not_a_shelf_inside_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_bundle` treats every directory at the root as a shelf. An index inside would
    validate as a shelf holding no papers, forever."""
    bundle, _, result = await indexed(settings_factory, tmp_path, monkeypatch)

    assert result.path.parent == bundle.parent
    assert result.path.name == f"{bundle.name}.chroma"
    assert bundle not in result.path.parents
    assert result.path.name not in {shelf.slug for shelf in read_bundle(bundle).shelves}


def test_an_unresolved_revision_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    """`main` is a moving pointer. A descriptor that recorded one would promise a
    reproducibility it cannot deliver, so the field stays empty and the run says so."""
    from okf_loremaster.emitters.vectors import IndexResult

    result = IndexResult(
        path=tmp_path / "b.chroma", bundle=tmp_path / "b", embed_model="stub/embedder"
    )
    descriptor = yaml.safe_load(index_descriptor(result))
    assert descriptor["embedding_revision"] == ""


# --- the store --------------------------------------------------------------


async def test_chroma_accepts_every_chunk_this_emitter_produces(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The metadata rules are Chroma's, so the only proof is Chroma. Everything above
    asserts what we believe it wants; this asserts that it agrees."""
    chromadb = pytest.importorskip("chromadb")
    from okf_loremaster.emitters.vectors import ChromaStore, chroma_settings

    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    result = await build_index(bundle, embedder=StubEmbedder())

    assert result.chunks == result.concept_chunks + result.predictor_chunks
    assert result.documents == TARGET
    assert not result.replaced

    client = chromadb.PersistentClient(path=str(result.path), settings=chroma_settings())
    collection = client.get_collection(COLLECTION)
    assert collection.count() == result.chunks
    # The distance metric survived the write, which is the thing a reopened store has to
    # agree about.
    assert collection.configuration_json["hnsw"]["space"] == DISTANCE

    fetched = collection.get(limit=1, include=["metadatas", "documents"])
    metadata = (fetched["metadatas"] or [{}])[0]
    assert set(REQUIRED_KEYS) <= set(metadata)

    # And re-indexing replaces rather than doubles.
    again = await build_index(bundle, embedder=StubEmbedder())
    assert again.replaced
    assert ChromaStore(result.path).count() == 0  # a third build starts from empty
    assert again.chunks == result.chunks


async def test_a_bundle_with_no_documents_is_reported_rather_than_indexed_empty(
    tmp_path: Path,
) -> None:
    """An empty store answers every query with nothing and looks like a broken one."""
    from okf_loremaster.okf.layout import INDEX_FILENAME

    bundle = tmp_path / "empty"
    bundle.mkdir()
    (bundle / INDEX_FILENAME).write_text(
        '---\ntitle: "Empty"\n---\n\n# Nothing\n', encoding="utf-8"
    )

    result = await build_index(bundle, embedder=StubEmbedder())
    assert result.chunks == 0
    assert not result.path.exists()
    assert any("no documents" in note for note in result.warnings)


# --- the validator ----------------------------------------------------------


async def test_the_validator_flags_a_pointer_at_an_index_that_is_not_there(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dangling pointer is silent at the far end: a consumer follows it, finds nothing,
    and attaches a bundle with no retrieval at all."""
    bundle, _, result = await indexed(settings_factory, tmp_path, monkeypatch)
    assert validate_bundle(bundle).ok

    for path in sorted(result.path.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    result.path.rmdir()

    report = validate_bundle(bundle)
    # Still a warning, never an error: the index is derived and rebuildable in one
    # command, and the bundle itself is untouched.
    assert report.ok
    assert any("points at a vector index" in finding.message for finding in report.warnings)
    assert all(finding.severity is Severity.WARNING for finding in report.warnings)


async def test_the_validator_flags_an_index_that_declares_no_distance(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, result = await indexed(settings_factory, tmp_path, monkeypatch)
    descriptor = result.path / DESCRIPTOR_FILENAME
    loaded = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    loaded["distance"] = "euclidean"
    loaded["embedding_revision"] = ""
    descriptor.write_text(yaml.safe_dump(loaded), encoding="utf-8")

    messages = [finding.message for finding in validate_bundle(bundle).warnings]
    assert any("`distance`" in note for note in messages)
    assert any("`embedding_revision`" in note for note in messages)


# --- the node ---------------------------------------------------------------


async def test_a_run_with_index_records_the_resolved_model_in_its_manifest(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`embed_model` is empty until something has actually embedded. The manifest
    promises the checkpoint that answered, which nobody knows at emit time."""
    pytest.importorskip("chromadb")
    stub = StubEmbedder()
    monkeypatch.setattr("okf_loremaster.run.embedder", lambda settings: stub)

    run = await full_run(settings_factory, tmp_path, monkeypatch, index=True)
    bundle = Path(run.state["bundle"])

    assert run.state["vector_index"] == str(vector_store_path(bundle))
    assert run.state["vector_chunks"] > TARGET
    manifest = run.state["manifest"]
    assert manifest is not None
    assert manifest.embed_model == "stub/embedder"
    assert manifest.embed_revision == "a" * 40
    assert stub.batches, "the node never reached the embedder"
    assert validate_bundle(bundle).ok


async def test_a_run_without_index_writes_no_store_and_says_nothing_about_one(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default. A bundle is complete without an index, so its absence is not a
    shortfall and does not warn."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)

    assert not vector_store_path(bundle).exists()
    report = validate_bundle(bundle)
    assert report.ok
    assert not [note for note in report.lines() if "vector" in note]


# --- the command ------------------------------------------------------------


def test_index_builds_a_store_from_a_bundle_that_already_exists(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route for a bundle somebody else built, or one edited by hand afterward.

    Synchronous on purpose: the command calls `asyncio.run`, which raises inside a loop
    that is already running — so the run that produces the bundle gets its own.
    """
    pytest.importorskip("chromadb")
    bundle = asyncio.run(golden(settings_factory, tmp_path, monkeypatch))
    monkeypatch.setattr("okf_loremaster.run.embedder", lambda settings: StubEmbedder())

    result = runner.invoke(app, ["index", str(bundle)])

    assert result.exit_code == 0, result.output
    assert vector_store_path(bundle).is_dir()
    assert DISTANCE in result.output
    # The caveat a metadata filter has to know about is printed where it will be read.
    assert "chunk_level" in result.output


def test_index_says_a_bundle_is_missing_rather_than_writing_an_empty_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("okf_loremaster.run.embedder", lambda settings: StubEmbedder())

    result = runner.invoke(app, ["index", str(tmp_path / "not-a-bundle")])

    assert result.exit_code == 1
    assert "not-a-bundle" in result.output
    assert not (tmp_path / "not-a-bundle.chroma").exists()


def test_the_missing_extra_is_named_before_a_run_starts_rather_than_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--index` is checked at assembly, not at the last node. Otherwise a run that took
    an hour would reach the end and only then discover it cannot do the thing it was
    asked for."""
    from okf_loremaster.config import ConfigError
    from okf_loremaster.run import embedder

    monkeypatch.setattr("okf_loremaster.run.importlib.util.find_spec", lambda name: None)

    with pytest.raises(ConfigError, match=r"\[vectors\]"):
        embedder(object())  # type: ignore[arg-type]


def test_index_is_refused_on_a_dry_run_rather_than_ignored(tmp_path: Path) -> None:
    """A dry run writes no bundle, and the index is built by reading one back."""
    result = runner.invoke(app, ["build", "a prompt", "--index", "--dry-run"])

    assert result.exit_code == 1
    assert "--index cannot be combined with --dry-run" in result.output
