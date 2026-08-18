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
    FALLBACK_WINDOW,
    REQUIRED_KEYS,
    ROW_LEVEL_KEYS,
    Chunk,
    build_index,
    chunks_for,
    index_descriptor,
    window_for,
)
from okf_loremaster.finalize import Finalize
from okf_loremaster.okf.layout import (
    DESCRIPTOR_FILENAME,
    OKF_DIRNAME,
    VECTORS_DIRNAME,
    vector_store_path,
)
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

    def __init__(
        self,
        *,
        model: str = "stub/embedder",
        revision: str = "a" * 40,
        window: int = 10_000,
    ) -> None:
        self._model = model
        self._revision = revision
        self._window = window
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

    @property
    def window(self) -> int:
        # Wide enough that nothing splits unless a test asks for a narrow one, so the
        # chunk-per-row and metadata assertions are not written around part boundaries.
        return self._window

    def count(self, text: str) -> int:
        # Whitespace-delimited words. Not what a real tokenizer does, and it does not need
        # to be: what is under test is that chunking respects whatever the embedder says.
        return len(text.split())

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


# --- fitting the window -----------------------------------------------------


async def test_a_chunk_too_long_for_the_window_is_split_rather_than_truncated(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An encoder truncates at its window and returns a vector anyway, so an over-long
    chunk is indexed by its head and its tail answers nothing — silently. Measured on a
    real corpus, half of all chunks overran the default checkpoint's 350-token window and
    a fifth of every token written was never embedded at all."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    window = window_for(StubEmbedder(window=40))

    split_something = False
    for document in read_bundle(bundle).documents():
        chunks = chunks_for(document, root=bundle, window=window)
        for chunk in chunks:
            assert window.fits(chunk.text), (chunk.id, window.count(chunk.text))
        split_something |= any(chunk.metadata["chunk_parts"] > 1 for chunk in chunks)

    assert split_something, "nothing split, so nothing was proved about splitting"


async def test_splitting_drops_no_word_of_the_chunk_it_split(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point. Splitting that lost text would be truncation with extra steps."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    window = window_for(StubEmbedder(window=40))

    for document in read_bundle(bundle).documents():
        whole = {chunk.id: chunk.text for chunk in chunks_for(document, root=bundle)}
        parts: dict[str, list[str]] = {}
        for chunk in chunks_for(document, root=bundle, window=window):
            parts.setdefault(chunk.id.split(".")[0], []).append(chunk.text)
        for handle, text in whole.items():
            rejoined = " ".join(parts[handle])
            missing = [word for word in text.split() if word not in rejoined]
            assert not missing, (handle, missing[:5])


async def test_every_part_says_which_paper_it_came_from(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part is retrieved on its own, and one that opened mid-caveat with no title would
    be a paragraph from nowhere."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    window = window_for(StubEmbedder(window=40))

    for document in read_bundle(bundle).documents():
        for chunk in chunks_for(document, root=bundle, window=window):
            assert chunk.text.startswith(document.title), chunk.id


async def test_a_chunk_that_fits_keeps_the_id_and_the_text_it_always_had(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting is for the chunks that need it. A part suffix on every id in the store
    would be a contract change charged to every consumer to fix a problem most chunks in
    most bundles do not have."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    window = window_for(StubEmbedder(window=10_000))

    for document in read_bundle(bundle).documents():
        before = chunks_for(document, root=bundle)
        after = chunks_for(document, root=bundle, window=window)
        assert [chunk.id for chunk in after] == [chunk.id for chunk in before]
        assert [chunk.text for chunk in after] == [chunk.text for chunk in before]
        assert all(chunk.metadata["chunk_parts"] == 1 for chunk in after)
        assert all(chunk.metadata["chunk_part"] == 0 for chunk in after)


async def test_the_parts_of_one_chunk_share_its_index_and_number_from_zero(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`chunk_index` stays the predictor's row number, which is what joins a chunk back to
    the table it came from. Renumbering parts through it would break that join, so parts
    are a second axis rather than more of the first."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    window = window_for(StubEmbedder(window=40))
    document = next(iter(read_bundle(bundle).documents()))

    groups: dict[str, list[Any]] = {}
    for chunk in chunks_for(document, root=bundle, window=window):
        groups.setdefault(chunk.id.split(".")[0], []).append(chunk)

    rows = len(markdown_table(document.section("Predictors reported") or ""))
    assert sorted(int(chunk[0].metadata["chunk_index"]) for chunk in groups.values()) == list(
        range(rows + 1)
    )
    for handle, chunks in groups.items():
        assert len({chunk.metadata["chunk_index"] for chunk in chunks}) == 1, handle
        assert [chunk.metadata["chunk_part"] for chunk in chunks] == list(range(len(chunks)))
        assert all(chunk.metadata["chunk_parts"] == len(chunks) for chunk in chunks)


def test_the_window_is_the_embedders_own_and_not_a_constant_here() -> None:
    """Two checkpoints have two different windows and two tokenizers that disagree about
    what the same sentence costs. The default one answers 350, not the 512 a BERT-family
    encoder is usually assumed to have — chunking to 512 would build every chunk a third
    too long for the model that embeds it."""
    assert window_for(StubEmbedder(window=128)).limit == 128
    assert window_for(StubEmbedder(window=512)).limit == 512
    # The stub counts words, so this is the stub's answer and not a character heuristic.
    assert window_for(StubEmbedder()).count("one two three") == 3


def test_an_embedder_that_cannot_say_gets_a_floor_rather_than_no_limit() -> None:
    """An embedder written against the older three-member protocol still indexes. It
    cannot be measured exactly, so it is measured pessimistically."""

    class Older:
        model_id = "older/embedder"
        revision = ""
        dimensions = 8

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

    window = window_for(Older())  # type: ignore[arg-type]
    assert window.limit == FALLBACK_WINDOW
    # Three characters a token is near the dense end of what real chunk text measured,
    # so the estimate over-counts prose rather than letting it overrun unnoticed.
    assert window.count("a" * 30) == 10


async def test_the_run_reports_what_it_split_instead_of_splitting_quietly(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store with 1.7x the chunks of the bundle it came from is a surprise worth
    explaining where it happens."""
    bundle = await golden(settings_factory, tmp_path, monkeypatch)
    result = await build_index(
        bundle, embedder=StubEmbedder(window=40), store=RecordingStore()
    )

    assert result.split
    assert result.window == 40
    assert f"{result.split} split to fit a 40-token window" in result.summary()

    unsplit = await build_index(
        bundle, embedder=StubEmbedder(window=10_000), store=RecordingStore()
    )
    assert unsplit.split == 0
    assert "split" not in unsplit.summary()


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


async def test_the_store_is_a_sibling_of_the_bundle_and_not_a_topic_inside_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_bundle` treats every directory at the root as a topic. An index inside would
    validate as a topic holding no papers, forever.

    Sibling rather than nested also makes the run folder the unit that moves: `okf/`
    and `vectors/` travel together under one `cp -r`, and either can be attached
    downstream on its own.
    """
    bundle, _, result = await indexed(settings_factory, tmp_path, monkeypatch)

    assert result.path.parent == bundle.parent
    assert result.path.name == VECTORS_DIRNAME
    assert bundle.name == OKF_DIRNAME
    assert bundle not in result.path.parents
    assert result.path.name not in {topic.slug for topic in read_bundle(bundle).topics}


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

    run = await full_run(settings_factory, tmp_path, monkeypatch, finalize=Finalize.BOTH)
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


def test_the_missing_extra_is_named_before_a_run_starts_rather_than_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The missing extra is named at assembly, not at the last node. Otherwise a run
    that took an hour would reach the end and only then discover it cannot do the thing
    it was asked for."""
    from okf_loremaster.config import ConfigError
    from okf_loremaster.run import embedder

    monkeypatch.setattr("okf_loremaster.run.importlib.util.find_spec", lambda name: None)

    with pytest.raises(ConfigError, match=r"\[vectors\]"):
        embedder(object())  # type: ignore[arg-type]


def test_choosing_what_to_keep_is_refused_on_a_dry_run_rather_than_ignored(
    tmp_path: Path,
) -> None:
    """A dry run writes nothing, so there is nothing for `--finalize` to keep."""
    result = runner.invoke(app, ["build", "a prompt", "--finalize", "both", "--dry-run"])

    assert result.exit_code == 1
    assert "--finalize cannot be combined with --dry-run" in result.output


def test_the_end_of_run_question_is_never_asked_into_a_terminal_textual_owns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This hung a finished 200-paper run until it was killed.

    Under `--tui` stdin is a tty and rich is enabled, so neither existing guard fires —
    but Textual holds the terminal in raw mode and reads the keystrokes itself. The
    prompt paints under a full-screen app that repaints over it, and blocks the main
    thread on a `readline` nothing can satisfy: pressing Enter does not help, because
    the keypress goes to Textual. `q` and `c` stop working for the same reason, so the
    log cannot even be copied out. `sample` on the live process showed
    `_textiowrapper_readline`, and the bundle was already complete on disk.

    Both guards are forced on so the test fails if `terminal_owned` stops being
    consulted, rather than passing because the harness has no tty anyway.
    """
    import sys

    from okf_loremaster.run import _ask_finalize

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("okf_loremaster.ui.plain.rich_enabled", lambda: True)

    def never(*args: object, **kwargs: object) -> object:
        raise AssertionError("blocked on stdin under a full-screen app; the run hangs here")

    monkeypatch.setattr("rich.prompt.IntPrompt.ask", never)

    kept = _ask_finalize(tmp_path / "okf", tmp_path / "vectors", console=None, terminal_owned=True)

    # Keeping both is keeping what the run already built — `finalize` is settled before
    # the graph so the embedder can be constructed — so no unasked-for deletion follows
    # from declining to ask.
    assert kept is Finalize.BOTH
