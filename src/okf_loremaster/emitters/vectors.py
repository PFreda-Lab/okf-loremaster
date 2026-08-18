"""Deriving the vector index from a finished bundle. Code, not an agent.

The index is built by **walking the bundle on disk**, never by extracting a second time
and never from the run state. That is what makes `okf-loremaster index <bundle>` a year
later produce the same store as the run did on the day, and it means there is
exactly one description of a paper in existence rather than two that can disagree.

Five choices here are load-bearing:

**Two chunk levels per paper, and the concept chunk deliberately omits the predictor
table.** One chunk carries the paper's identity — bottom line, null findings, vocabulary
hints, caveats — and one chunk carries each predictor row with the population, the
outcome definition and the bottom line around it for context. The table is left out of
the concept chunk because it is already covered row by row, and because a whole document
would overrun a sentence encoder's window and be silently truncated: the tail of a
truncated chunk is not indexed and nothing says so.

**Metadata is never `None`.** Chroma raises `TypeError` on a null value, so a missing
field is written as `""`. `n` is the exception that proves it: it is written as an `int`
when the paper reported one, so a numeric filter works, and `""` when it did not.

**`timing`, `confidence` and `evidence_type` describe a predictor row, so a concept chunk
carries `""` for all three.** That is why `chunk_level` exists. Without it a filter like
`confidence == "high"` would quietly exclude every concept chunk in the store — half the
corpus — and look like it had simply found less.

**The distance metric is declared, never defaulted.** Chroma's default space is L2. A
consumer that assumed cosine against an L2 store gets results in a different order and no
error at all, so the collection is created with the space set and the descriptor states
it.

**Telemetry is off.** Chroma phones home by default. A tool that reads the literature on
someone's behalf has no business reporting that it ran to a third party.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import yaml

from okf_loremaster.okf.layout import (
    ABSTRACT_SECTION,
    BOTTOM_LINE_SECTION,
    DESCRIPTOR_FILENAME,
    DISTANCES,
    NONE_CELL,
    PREDICTOR_COLUMNS,
    PREDICTORS_SECTION,
    QUOTE_LEAD,
    vector_store_path,
)
from okf_loremaster.okf.reader import OkfDocument, fact_list, markdown_table, read_bundle

__all__ = [
    "BATCH_SIZE",
    "COLLECTION",
    "DISTANCE",
    "REQUIRED_KEYS",
    "ROW_LEVEL_KEYS",
    "ChromaStore",
    "Chunk",
    "Embedder",
    "IndexResult",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "build_index",
    "chroma_settings",
    "chunks_for",
    "index_descriptor",
    "link_index",
]

MetadataValue: TypeAlias = str | int | float | bool

# One collection per store, named for what it holds rather than for the run: a consumer
# attaches the directory and has to be able to guess right.
COLLECTION = "papers"

# Declared into the collection *and* the descriptor. Chroma's own default is L2.
DISTANCE = "cosine"

# Texts per `encode` call, and rows per `add`. Both are about keeping a long run
# responsive — progress is reported between batches — rather than about throughput.
BATCH_SIZE = 32
STORE_BATCH = 256

CONCEPT_LEVEL = "concept"
PREDICTOR_LEVEL = "predictor"

# Metadata key to the table column it is read from. Per predictor row, so a concept chunk
# has no single value for any of them. Stated as one mapping so the keys a consumer
# filters on and the columns they are read from cannot drift apart.
#
# All three are closed vocabularies, and that is the entry requirement rather than a
# coincidence. A Chroma `where` clause tests equality, so a column worth putting here is
# one a consumer can name a whole value of. `Interacts with` holds a semicolon-joined list
# of variable names no filter could ever match exactly — it reaches the index through the
# chunk *text*, where a retriever can actually use it, and not through here.
_ROW_METADATA = {"timing": "Timing", "confidence": "Confidence", "evidence_type": "Type"}
ROW_LEVEL_KEYS = tuple(_ROW_METADATA)

# What a downstream RAG consumer requires of every chunk. Ours are named identically, so
# the descriptor's `metadata_map` is an identity map rather than a translation.
REQUIRED_KEYS = ("source", "title", "id", "chunk_index")

# Roughly a 512-token window at four characters a token — the size of the BERT-family
# encoders this tool defaults to. Only used to warn: the embedder truncates silently, and
# a chunk that lost its tail is worth knowing about.
TRUNCATION_CHARS = 2000

# The sections chunking treats specially are imported by name from `layout`. They were
# `BODY_SECTIONS[0]` and `BODY_SECTIONS[1]` for one version, which was a bomb with a long
# fuse: inserting `# Abstract` second would have redefined `PREDICTORS_SECTION` as the
# abstract, and every row chunk in every store would have been built from the wrong half
# of the document with no error raised anywhere.

# `1. <quote>` under the quote lead-in.
_NUMBERED = re.compile(r"^(?P<number>\d+)\.\s+(?P<text>.+)$")

# How a column reads inside a chunk. The chunk is prose a retriever embeds, so two
# headings are spelled out and the rest read as themselves. Derived from the columns
# rather than restated: a renamed column would otherwise stop matching in silence, and
# the row would embed with that field missing.
_LABELS = {"Type": "Evidence type", "p": "p-value", "Strength": "Evidence strength"}
_ROW_LABELS = tuple(
    (column, _LABELS.get(column, column)) for column in PREDICTOR_COLUMNS if column != "#"
)

# Bottom-line facts worth repeating inside a row chunk. A row retrieved on its own is
# useless without the population it was measured in.
_CONTEXT_FACTS = ("Population", "Outcome", "Design", "N")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One embedded unit: its store id, the text embedded, and what it can be filtered by."""

    id: str
    text: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    @property
    def level(self) -> str:
        return str(self.metadata.get("chunk_level") or "")


@dataclass(frozen=True, slots=True)
class IndexResult:
    """What `build_index` did, for the node and the CLI to report."""

    path: Path
    bundle: Path
    documents: int = 0
    chunks: int = 0
    concept_chunks: int = 0
    predictor_chunks: int = 0
    collection: str = COLLECTION
    distance: str = DISTANCE
    embed_model: str = ""
    embed_revision: str = ""
    dimensions: int = 0
    replaced: bool = False
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"{self.chunks} chunk(s) from {self.documents} document(s) "
            f"({self.concept_chunks} concept, {self.predictor_chunks} predictor)"
        )


# --- chunking ---------------------------------------------------------------


def chunks_for(document: OkfDocument, *, root: Path) -> list[Chunk]:
    """Every chunk one document contributes: one concept chunk, then one per row."""
    base = _document_metadata(document, root=root)
    handle = str(base["id"])

    chunks = [
        Chunk(
            id=_chunk_id(handle, 0),
            text=_concept_text(document),
            metadata={
                **base,
                "chunk_index": 0,
                "chunk_level": CONCEPT_LEVEL,
                **dict.fromkeys(ROW_LEVEL_KEYS, ""),
            },
        )
    ]

    section = document.section(PREDICTORS_SECTION) or ""
    facts = fact_list(document.section(BOTTOM_LINE_SECTION) or "")
    quotes = _numbered(section)
    for number, row in enumerate(markdown_table(section), start=1):
        chunks.append(
            Chunk(
                id=_chunk_id(handle, number),
                text=_row_text(document, row, facts=facts, quote=quotes.get(number, "")),
                metadata={
                    **base,
                    "chunk_index": number,
                    "chunk_level": PREDICTOR_LEVEL,
                    **{key: row.get(column, "") for key, column in _ROW_METADATA.items()},
                },
            )
        )
    return chunks


def _chunk_id(handle: str, index: int) -> str:
    """Unique within the store, and legible: `10000#3` is the fourth chunk of PMID 10000."""
    return f"{handle}#{index}"


def _document_metadata(document: OkfDocument, *, root: Path) -> dict[str, MetadataValue]:
    """The document-level half of every chunk's metadata. No value is ever `None`.

    `source` is the document's path inside the bundle rather than a URL, because that is
    the link between the two resources a consumer attaches: a chunk retrieved from the
    vector store names the OKF file to open, and the PubMed URI is still recoverable from
    `id`.
    """
    fields = document.fields
    return {
        "source": _relative(document.path, root),
        "title": document.title,
        "id": document.doc_id,
        "pmid": document.pmid,
        "domain": document.domain,
        "journal": _text(fields.get("journal")),
        "published": _text(fields.get("published")),
        "study_design": _text(fields.get("study_design")),
        "n": _count(fields.get("n")),
        "text_basis": _text(fields.get("text_basis")),
        "license": _text(fields.get("license")),
    }


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _count(value: Any) -> int | str:
    """`n` as an integer where the paper reported one, `""` where it did not.

    An integer so that a numeric metadata filter works at all; `""` rather than `0`
    because a study of zero participants is a different claim from one that never said.
    """
    text = _text(value).replace(",", "")
    try:
        return int(text)
    except ValueError:
        return ""


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _concept_text(document: OkfDocument) -> str:
    """The paper as one chunk: everything except what is covered row by row.

    `# Abstract` is left out too, and that is a deliberate trade rather than an oversight.
    A structured abstract runs 250 to 350 words; adding one would push almost every concept
    chunk past `TRUNCATION_CHARS`, where the embedder drops the tail *silently* — so the
    caveats and vocabulary hints at the end of the document would stop being retrievable
    at all, in exchange for prose that restates a bottom line written from the same source.
    The abstract stays in the bundle for an agent that opens the file; the index keeps its
    two levels, and both of them keep fitting in the window.
    """
    parts = [document.title, _text(document.fields.get("description"))]
    parts += [
        f"{heading}: {body}"
        for heading, body in document.sections()
        if heading not in (PREDICTORS_SECTION, ABSTRACT_SECTION)
    ]
    return "\n\n".join(part for part in parts if part)


def _row_text(
    document: OkfDocument,
    row: Mapping[str, str],
    *,
    facts: Mapping[str, str],
    quote: str,
) -> str:
    """One predictor row, with enough of its paper around it to stand alone."""
    lines = [document.title, ""]
    # `NONE_CELL` is skipped, not embedded. It is the table's way of writing "nothing
    # here", and `Interacts with: —` on the nineteen rows in twenty that state no
    # interaction is a sentence that means nothing and still moves the vector.
    lines += [
        f"{label}: {row[column]}"
        for column, label in _ROW_LABELS
        if row.get(column) and row[column] != NONE_CELL
    ]
    lines += [f"{label}: {facts[label]}" for label in _CONTEXT_FACTS if facts.get(label)]
    bottom = _first_paragraph(document.section(BOTTOM_LINE_SECTION) or "")
    if bottom:
        lines += ["", bottom]
    if quote:
        lines += ["", f"Quoted from the paper: {quote}"]
    return "\n".join(lines)


def _first_paragraph(text: str) -> str:
    """The prose above a fact list, which is the sentence a person would quote."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("-") or not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def _numbered(section: str) -> dict[int, str]:
    """The quotes under `QUOTE_LEAD`, keyed by the table row number they belong to."""
    _, lead, tail = section.partition(QUOTE_LEAD)
    if not lead:
        return {}
    quotes: dict[int, str] = {}
    for line in tail.splitlines():
        match = _NUMBERED.match(line.strip())
        if match is not None:
            quotes[int(match.group("number"))] = match.group("text").strip()
    return quotes


# --- the store --------------------------------------------------------------


class VectorStore(Protocol):
    """Where chunks land.

    A protocol because the store is the one part of this that is genuinely swappable —
    LanceDB or Qdrant would satisfy it — and because a test can then assert on the exact
    metadata handed over without a database in the way.
    """

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def count(self) -> int: ...


def chroma_settings() -> Any:
    """Chroma's client settings, with anonymized telemetry off.

    Not a preference. The default posts run events to a third-party endpoint, and a tool
    that reads the literature on someone's behalf must not report that it ran.
    """
    from chromadb.config import Settings as ChromaSettings

    return ChromaSettings(anonymized_telemetry=False)


class ChromaStore:
    """A persistent Chroma collection, created with its distance metric set.

    Re-indexing replaces the collection rather than adding to it. That is the one place
    this package deletes something it did not just write, and it is deliberate: the index
    is derived and a rebuild costs minutes, while a store still answering with papers the
    bundle no longer holds is wrong in a way nobody would notice.
    """

    def __init__(
        self,
        path: Path,
        *,
        collection: str = COLLECTION,
        distance: str = DISTANCE,
    ) -> None:
        import chromadb

        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {', '.join(DISTANCES)}, not {distance!r}")

        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.name = collection
        self.distance = distance
        self._client = chromadb.PersistentClient(path=str(path), settings=chroma_settings())

        existing = {getattr(item, "name", item) for item in self._client.list_collections()}
        self.replaced = collection in existing
        if self.replaced:
            self._client.delete_collection(collection)
        configuration: Any = {"hnsw": {"space": distance}}
        self._collection = self._client.create_collection(
            name=collection,
            configuration=configuration,
            # We supply vectors; without this Chroma would fetch its own ONNX embedder.
            embedding_function=None,
        )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunk(s) but {len(embeddings)} embedding(s) — they are "
                f"positional and a mismatch would file text under another paper's vector"
            )
        for start in range(0, len(chunks), STORE_BATCH):
            window = chunks[start : start + STORE_BATCH]
            vectors: list[Sequence[float]] = [
                list(vector) for vector in embeddings[start : start + STORE_BATCH]
            ]
            self._collection.add(
                ids=[chunk.id for chunk in window],
                embeddings=vectors,
                documents=[chunk.text for chunk in window],
                metadatas=[dict(chunk.metadata) for chunk in window],
            )

    def count(self) -> int:
        return int(self._collection.count())


# --- the embedder -----------------------------------------------------------


class Embedder(Protocol):
    """Whatever turns text into vectors.

    Injected rather than constructed, because the default one downloads a model on first
    use and the test suite never reaches the network.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """The default embedder: a local sentence-transformers checkpoint.

    Loaded lazily, so building it costs nothing until something is actually embedded —
    which matters because it is constructed when the run starts and used only at the very
    end, and because importing `sentence_transformers` pulls torch.

    `revision` is *resolved*, not echoed. The descriptor has to name the checkpoint that
    produced the vectors, and `main` is a moving pointer that promises a reproducibility
    it cannot deliver, so an unresolvable revision is reported empty rather than guessed.
    """

    # A hub commit is a 40-character hex sha. Anything else in that position is a
    # directory name, not a revision.
    _SHA = re.compile(r"^[0-9a-f]{40}$")

    def __init__(
        self, model_name: str, pinned_revision: str | None = None, *, batch_size: int = BATCH_SIZE
    ) -> None:
        self.model_name = model_name
        self.pinned_revision = pinned_revision or None
        self.batch_size = batch_size
        self._model: Any = None
        self._revision = ""

    def load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, revision=self.pinned_revision)
            self._revision = self._resolve_revision()
        return self._model

    def _resolve_revision(self) -> str:
        if Path(self.model_name).expanduser().exists():
            # A checkpoint on disk has no revision to record, and inventing one would be
            # a claim about provenance nobody can check.
            return ""
        try:
            from huggingface_hub import snapshot_download

            # Already downloaded by the line above; this is a cache lookup, not a fetch.
            snapshot = Path(
                snapshot_download(
                    self.model_name, revision=self.pinned_revision, local_files_only=True
                )
            )
        except Exception:  # the hub raises several different ways for "not cached"
            return self.pinned_revision or ""
        return snapshot.name if self._SHA.match(snapshot.name) else (self.pinned_revision or "")

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def revision(self) -> str:
        self.load()
        return self._revision

    @property
    def dimensions(self) -> int:
        return int(self.load().get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.load()
        vectors = model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True)
        return [[float(value) for value in vector] for vector in vectors]


# --- building ---------------------------------------------------------------


async def build_index(
    bundle: Path,
    *,
    embedder: Embedder,
    store: VectorStore | None = None,
    store_path: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> IndexResult:
    """Walk a finished bundle, embed it, and write the store and its descriptor.

    Async only so that embedding can run off the event loop: the encoder is synchronous
    and CPU-bound, and a run that blocked the loop for minutes would freeze the live
    meter that says how far along it is.
    """
    read = read_bundle(bundle)
    target = store_path if store_path is not None else vector_store_path(bundle)
    documents = list(read.documents())
    chunks = [chunk for document in documents for chunk in chunks_for(document, root=bundle)]

    if not chunks:
        return IndexResult(
            path=target,
            bundle=bundle,
            embed_model=embedder.model_id,
            warnings=(
                f"{bundle} holds no documents, so no vector index was built — an empty "
                f"store answers every query with nothing and looks like a broken one",
            ),
        )

    warnings: list[str] = []
    overlong = sum(1 for chunk in chunks if len(chunk.text) > TRUNCATION_CHARS)
    if overlong:
        warnings.append(
            f"{overlong} of {len(chunks)} chunk(s) are over {TRUNCATION_CHARS} characters "
            f"and may be truncated by the embedder, which drops their tail silently"
        )

    vectors = await _embed(chunks, embedder, on_progress=on_progress)
    sink = store if store is not None else ChromaStore(target, distance=DISTANCE)
    sink.add(chunks, vectors)

    if not embedder.revision:
        warnings.append(
            f"the revision of {embedder.model_id} could not be resolved, so the index "
            f"descriptor records it unpinned — a rebuild is not guaranteed to reproduce "
            f"these vectors"
        )

    result = IndexResult(
        path=target,
        bundle=bundle,
        documents=len(documents),
        chunks=len(chunks),
        concept_chunks=sum(1 for chunk in chunks if chunk.level == CONCEPT_LEVEL),
        predictor_chunks=sum(1 for chunk in chunks if chunk.level == PREDICTOR_LEVEL),
        collection=getattr(sink, "name", COLLECTION),
        distance=getattr(sink, "distance", DISTANCE),
        embed_model=embedder.model_id,
        embed_revision=embedder.revision,
        dimensions=embedder.dimensions,
        replaced=bool(getattr(sink, "replaced", False)),
        warnings=tuple(warnings),
    )

    target.mkdir(parents=True, exist_ok=True)
    (target / DESCRIPTOR_FILENAME).write_text(
        index_descriptor(result, bundle_descriptor=read.descriptor), encoding="utf-8"
    )
    note = link_index(bundle, result)
    if note:
        warnings.append(note)
        result = replace(result, warnings=tuple(warnings))
    return result


async def _embed(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    on_progress: Callable[[int, int], None] | None,
) -> list[list[float]]:
    """Embed in batches, reporting between them from the event loop's own thread.

    The bus is an asyncio queue and is not thread-safe, so progress is emitted here
    rather than from inside the worker thread.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        window = [chunk.text for chunk in chunks[start : start + BATCH_SIZE]]
        vectors.extend(await asyncio.to_thread(embedder.encode, window))
        if on_progress is not None:
            on_progress(len(vectors), len(chunks))
    if len(vectors) != len(chunks):
        raise ValueError(
            f"the embedder returned {len(vectors)} vector(s) for {len(chunks)} chunk(s)"
        )
    return vectors


# --- the descriptors --------------------------------------------------------


def index_descriptor(
    result: IndexResult, *, bundle_descriptor: Mapping[str, Any] | None = None
) -> str:
    """`resource_descriptor.yaml` for the store — what a consumer reads on attach.

    Carries only what a prebuilt vector resource is required to declare — the embedding
    model, the distance metric, where the store is, and how our metadata keys map onto
    the ones a consumer expects — plus optional keys anything generic can ignore. The
    resource has to be attachable by a consumer that has never heard of this tool.
    """
    source = dict(bundle_descriptor or {})
    identifier = str(source.get("id") or result.bundle.name)
    name = str(source.get("name") or result.bundle.name)
    payload: dict[str, Any] = {
        "kind": "rag",
        "id": f"{identifier}-vectors",
        "name": f"{name} — vector index",
        "description": (
            f"{result.chunks} chunks over {result.documents} papers, embedded from the "
            f"OKF bundle beside this store."
        ),
        "store": {"type": "chroma", "path": ".", "collection": result.collection},
        "embedding_model": result.embed_model,
        "embedding_revision": result.embed_revision,
        "dimensions": result.dimensions,
        "distance": result.distance,
        # Ours are already named what a consumer asks for, so this is an identity map. It
        # is written out rather than omitted because a consumer is entitled to read it
        # and a missing map is indistinguishable from a forgotten one.
        "metadata_map": {key: key for key in REQUIRED_KEYS},
        "metadata_keys": sorted(_METADATA_KEYS),
        "chunk_levels": {
            CONCEPT_LEVEL: "one per paper: everything except the predictor table",
            PREDICTOR_LEVEL: (
                "one per predictor row, with the population, outcome and bottom line "
                "around it"
            ),
        },
        "source_bundle": f"../{result.bundle.name}",
        "notes": (
            f"{', '.join(ROW_LEVEL_KEYS)} describe a predictor row: a {CONCEPT_LEVEL} "
            f"chunk carries \"\" for all three, so a filter on any of them must either "
            f"allow \"\" or select chunk_level == \"{PREDICTOR_LEVEL}\". Missing values "
            f"are \"\" everywhere, never null; n is an integer when the paper reported "
            f"one."
        ),
    }
    return str(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))


# Every metadata key a chunk can carry, declared for a consumer building a filter.
_METADATA_KEYS = (
    *REQUIRED_KEYS,
    *ROW_LEVEL_KEYS,
    "chunk_level",
    "pmid",
    "domain",
    "journal",
    "published",
    "study_design",
    "n",
    "text_basis",
    "license",
)


def link_index(bundle: Path, result: IndexResult) -> str:
    """Point the bundle's own descriptor at the index. Returns a warning, or `""`.

    The bundle descriptor is what a consumer opens first, so it is where "there is also
    a vector store, built from this, with this model" belongs. The root `index.md` is
    left alone: it is a map of the literature, and it was written before the index
    existed.
    """
    path = bundle / DESCRIPTOR_FILENAME
    if not path.exists():
        return (
            f"no {DESCRIPTOR_FILENAME} in {bundle}, so the bundle does not point at its "
            f"vector index — the store is still complete and usable on its own"
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return f"{path} could not be updated to point at the vector index: {exc}"
    if not isinstance(loaded, dict):
        return f"{path} is not a YAML mapping, so it was left alone"

    loaded["vectors"] = {
        "path": f"../{result.path.name}",
        "descriptor": DESCRIPTOR_FILENAME,
        "collection": result.collection,
        "embedding_model": result.embed_model,
        "embedding_revision": result.embed_revision,
        "distance": result.distance,
        "chunks": result.chunks,
    }
    path.write_text(
        str(yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True, width=100)),
        encoding="utf-8",
    )
    return ""
