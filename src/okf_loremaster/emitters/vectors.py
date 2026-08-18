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
the concept chunk because it is already covered row by row.

**A chunk that does not fit the encoder's window is split, never allowed to overrun it.**
An encoder truncates at its window and returns a vector anyway, so the tail of an
over-long chunk is simply not indexed and nothing says so — text that is plainly in the
bundle answers no query. The window is measured in *tokens, by the embedder's own
tokenizer*, because a character budget cannot do this job: across three real bundles
chunk text ran from 2.1 to 6.8 characters per token, so any single constant is either
too tight for most chunks or too loose for the dense ones. Splitting is at natural
boundaries — section, then paragraph, then line, then word — and every part repeats the
paper's title so it can still be read on its own when it comes back alone.

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
    "Window",
    "build_index",
    "chroma_settings",
    "chunks_for",
    "index_descriptor",
    "link_index",
    "window_for",
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

# Where the window lands when an embedder cannot say what its own is. 350 tokens is the
# default checkpoint's, and is on the small side of the BERT family — guessing small
# over-splits, which costs a little retrieval precision; guessing large loses text.
FALLBACK_WINDOW = 350

# Room left inside the window for the special tokens an encoder adds ([CLS], [SEP]) and
# for the tokenizer disagreeing with itself about a boundary a split landed on.
WINDOW_MARGIN = 8

# Packing is *balanced*, not greedy: a chunk needing two parts is split near the middle
# rather than filled to the brim and left with a remainder. Greedy packing of a 400-token
# chunk into a 350-token window yields a 50-token second part, which is a fragment that
# matches on the strength of the repeated title alone. This is the fraction of the window
# a balanced target may be relaxed by to avoid landing a boundary mid-section.
BALANCE_SLACK = 0.15

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
class Window:
    """How much text one embedded unit may carry, and how to measure it.

    Both halves come from the embedder rather than from a constant here, because the
    answer is a property of the checkpoint somebody configured: the default one has a
    350-token window, another has 512, and their tokenizers disagree about how many
    tokens the same sentence is. `count` is the embedder's own tokenizer, so the measure
    and the thing being measured cannot drift apart.
    """

    limit: int
    count: Callable[[str], int]

    def fits(self, text: str) -> bool:
        return self.count(text) <= self.limit


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
    split: int = 0
    window: int = 0
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        line = (
            f"{self.chunks} chunk(s) from {self.documents} document(s) "
            f"({self.concept_chunks} concept, {self.predictor_chunks} predictor)"
        )
        # Said out loud because it is the difference between the store holding the whole
        # bundle and holding the first 350 tokens of each piece of it.
        if self.split:
            line += f", {self.split} split to fit a {self.window}-token window"
        return line


# --- chunking ---------------------------------------------------------------


def chunks_for(document: OkfDocument, *, root: Path, window: Window | None = None) -> list[Chunk]:
    """Every chunk one document contributes: one concept chunk, then one per row.

    With a `window`, a chunk too long for it becomes several parts that each fit; without
    one, each is emitted whole and whatever embeds it decides what to keep. The parameter
    is optional so that this stays a pure function of the document for anything that only
    wants to see what a paper contributes.
    """
    base = _document_metadata(document, root=root)
    handle = str(base["id"])

    chunks = list(
        _parts(
            _concept_blocks(document),
            handle=handle,
            index=0,
            lead=document.title,
            joiner=_JOINER,
            window=window,
            metadata={
                **base,
                "chunk_index": 0,
                "chunk_level": CONCEPT_LEVEL,
                **dict.fromkeys(ROW_LEVEL_KEYS, ""),
            },
        )
    )

    section = document.section(PREDICTORS_SECTION) or ""
    facts = fact_list(document.section(BOTTOM_LINE_SECTION) or "")
    quotes = _numbered(section)
    for number, row in enumerate(markdown_table(section), start=1):
        chunks += _parts(
            _row_blocks(document, row, facts=facts, quote=quotes.get(number, "")),
            handle=handle,
            index=number,
            lead=document.title,
            joiner=_JOINER,
            window=window,
            metadata={
                **base,
                "chunk_index": number,
                "chunk_level": PREDICTOR_LEVEL,
                **{key: row.get(column, "") for key, column in _ROW_METADATA.items()},
            },
        )
    return chunks


def _chunk_id(handle: str, index: int, part: int = 0) -> str:
    """Unique within the store, and legible: `10000#3` is the fourth chunk of PMID 10000.

    A part suffix is appended only when there is more than one, so the id of a chunk that
    fits its window is the id it has always had — and `10000#3.1` reads as the second part
    of that chunk rather than as some fourteenth one.
    """
    return f"{handle}#{index}" if part == 0 else f"{handle}#{index}.{part}"


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


def _concept_blocks(document: OkfDocument) -> list[str]:
    """The paper as blocks: everything except what is covered row by row, minus the title.

    The title is the *lead* rather than a block, because it is repeated at the head of
    every part this becomes — so a part that arrives on its own still says which paper it
    is from.

    `# Abstract` is left out, and that is a trade rather than an oversight: it is prose
    restating a bottom line written from the same source, so indexing it spends window on
    a second telling of what the concept chunk already says. It stays in the bundle for an
    agent that opens the file.
    """
    blocks = [_text(document.fields.get("description"))]
    blocks += [
        f"{heading}: {body}"
        for heading, body in document.sections()
        if heading not in (PREDICTORS_SECTION, ABSTRACT_SECTION)
    ]
    return [block for block in blocks if block]


def _row_blocks(
    document: OkfDocument,
    row: Mapping[str, str],
    *,
    facts: Mapping[str, str],
    quote: str,
) -> list[str]:
    """One predictor row, with enough of its paper around it to stand alone.

    The row's own fields are a single block on purpose. They are what the chunk *is*, and
    a split that put `Predictor:` in one part and `Effect:` in another would produce two
    halves that each read as a different claim than the row makes.
    """
    # `NONE_CELL` is skipped, not embedded. It is the table's way of writing "nothing
    # here", and `Interacts with: —` on the nineteen rows in twenty that state no
    # interaction is a sentence that means nothing and still moves the vector.
    lines = [
        f"{label}: {row[column]}"
        for column, label in _ROW_LABELS
        if row.get(column) and row[column] != NONE_CELL
    ]
    lines += [f"{label}: {facts[label]}" for label in _CONTEXT_FACTS if facts.get(label)]
    blocks = ["\n".join(lines)]
    blocks.append(_first_paragraph(document.section(BOTTOM_LINE_SECTION) or ""))
    blocks.append(f"Quoted from the paper: {quote}" if quote else "")
    return [block for block in blocks if block]


# --- fitting the window -----------------------------------------------------

# Blocks are joined, and split, at the coarsest boundary that gets a part under the
# window. A blank line separates sections and paragraphs; a newline separates the lines
# within one; a space is the last resort before giving up on a boundary altogether.
_JOINER = "\n\n"
_BOUNDARIES = ("\n\n", "\n", " ")


def _parts(
    blocks: Sequence[str],
    *,
    handle: str,
    index: int,
    lead: str,
    joiner: str,
    window: Window | None,
    metadata: Mapping[str, MetadataValue],
) -> list[Chunk]:
    """One chunk per part: same metadata throughout, differing only in id and part."""
    texts = _pack(blocks, lead=lead, joiner=joiner, window=window)
    return [
        Chunk(
            id=_chunk_id(handle, index, part),
            text=text,
            metadata={**metadata, "chunk_part": part, "chunk_parts": len(texts)},
        )
        for part, text in enumerate(texts)
    ]


def _pack(blocks: Sequence[str], *, lead: str, joiner: str, window: Window | None) -> list[str]:
    """The text of each part, in order. One part whenever the whole thing fits."""
    units = [block for block in blocks if block]
    whole = joiner.join([lead, *units]) if units else lead
    if window is None or window.fits(whole):
        return [whole]

    # Room for the lead on every part, and for the tokenizer counting a join boundary
    # differently than it counted the two sides separately.
    budget = window.limit - WINDOW_MARGIN - window.count(lead)
    if budget < 1:
        # A title alone fills the window. Nothing here can help, and splitting into
        # title-only parts would be worse than one over-long chunk.
        return [whole]

    atoms = [piece for unit in units for piece in _atoms(unit, budget=budget, window=window)]
    filled = _fill(atoms, budget=budget, joiner=joiner, window=window)

    # Balanced rather than greedy: refill against an even share of the same number of
    # parts, and keep it only if it still costs no more parts than filling to the brim.
    share = sum(window.count(atom) for atom in atoms) / len(filled)
    relaxed = min(budget, int(share * (1 + BALANCE_SLACK)) + 1)
    if relaxed < budget:
        balanced = _fill(atoms, budget=relaxed, joiner=joiner, window=window)
        if len(balanced) <= len(filled):
            filled = balanced

    return [joiner.join([lead, *part]) for part in filled]


def _fill(atoms: Sequence[str], *, budget: int, joiner: str, window: Window) -> list[list[str]]:
    """Group atoms into parts, each under `budget`. Never empty, never an empty group.

    Sized from per-atom counts plus a token for each join rather than by re-counting every
    candidate part, which would be quadratic in the number of atoms for an answer that
    `WINDOW_MARGIN` already covers.
    """
    parts: list[list[str]] = []
    current: list[str] = []
    size = 0
    for atom in atoms:
        cost = window.count(atom) + 1
        if current and size + cost > budget:
            parts.append(current)
            current, size = [], 0
        current.append(atom)
        size += cost
    if current:
        parts.append(current)
    return parts or [[]]


def _atoms(text: str, *, budget: int, window: Window) -> list[str]:
    """`text` as the largest runs that each fit `budget`, split at the coarsest boundary.

    A run is rejoined with the separator it was split on, so a paragraph broken at spaces
    comes back as prose rather than as a column of words.
    """
    if window.count(text) <= budget:
        return [text]
    for separator in _BOUNDARIES:
        pieces = [piece for piece in text.split(separator) if piece.strip()]
        if len(pieces) < 2:
            continue
        runs = _join_runs(pieces, separator=separator, budget=budget, window=window)
        if all(window.count(run) <= budget for run in runs):
            return runs
        return [
            atom
            for run in runs
            for atom in (
                _atoms(run, budget=budget, window=window)
                if run != text and window.count(run) > budget
                else [run]
            )
        ]
    # One token longer than the whole window. Embedding it truncated is the only option
    # left, and it is a single word, so there is no tail of meaning to lose.
    return [text]


def _join_runs(pieces: Sequence[str], *, separator: str, budget: int, window: Window) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = [*current, piece]
        if current and window.count(separator.join(candidate)) > budget:
            runs.append(separator.join(current))
            current = [piece]
        else:
            current = candidate
    if current:
        runs.append(separator.join(current))
    return runs


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

    @property
    def window(self) -> int:
        """How many tokens one text may carry before the rest is dropped, or 0 if unknown.

        Optional in the sense that 0 is a legal answer — an embedder behind an HTTP API
        may genuinely not know — and chunking then falls back to `FALLBACK_WINDOW`.
        """
        ...

    def count(self, text: str) -> int:
        """Tokens `text` occupies, by this embedder's own tokenizer."""
        ...

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

    @property
    def window(self) -> int:
        """`max_seq_length`, which is the checkpoint's own answer and not always 512.

        The default checkpoint says 350. Assuming the BERT architectural maximum instead
        would have every chunk built a third too long for the model that embeds it.
        """
        return int(getattr(self.load(), "max_seq_length", 0) or 0)

    def count(self, text: str) -> int:
        # `add_special_tokens` because [CLS] and [SEP] occupy the same window the text
        # does, and a chunk sized to the window without them is two tokens over it.
        tokenizer = self.load().tokenizer
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.load()
        vectors = model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True)
        return [[float(value) for value in vector] for vector in vectors]


def window_for(embedder: Embedder) -> Window:
    """What to chunk against: the embedder's own window and tokenizer where it has them.

    Read through `getattr` rather than off the protocol directly, so an embedder written
    against the older three-member `Embedder` still indexes. It gets the fallback window
    and a characters-per-token estimate, which is worse than an exact count and much
    better than emitting chunks nothing measured at all.
    """
    limit = int(getattr(embedder, "window", 0) or 0)
    count = getattr(embedder, "count", None)
    return Window(
        limit=limit or FALLBACK_WINDOW,
        count=count if callable(count) else _estimate_tokens,
    )


def _estimate_tokens(text: str) -> int:
    """Tokens, for an embedder that cannot count its own. Deliberately pessimistic.

    Real chunk text measured 2.1 to 6.8 characters per token against the default
    checkpoint. Three is near the dense end, so this over-counts typical prose and
    over-splits rather than letting a dense chunk overrun a window unmeasured.
    """
    return -(-len(text) // 3)


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
    # Off the loop because the first call to either loads the model, and chunking needs
    # the answer before anything is embedded.
    window = await asyncio.to_thread(window_for, embedder)
    chunks = [
        chunk
        for document in documents
        for chunk in chunks_for(document, root=bundle, window=window)
    ]

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
    overlong = [chunk for chunk in chunks if not window.fits(chunk.text)]
    if overlong:
        # Only reachable when a single word is longer than the whole window, since
        # everything else was split to fit. Worth saying rather than swallowing.
        warnings.append(
            f"{len(overlong)} of {len(chunks)} chunk(s) could not be split under the "
            f"{window.limit}-token window of {embedder.model_id} and will be truncated"
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
        # Chunks that became more than one part, counted as the parent chunks they came
        # from rather than as the parts they became.
        split=len(
            {chunk.id.split(".")[0] for chunk in chunks if chunk.metadata["chunk_parts"] != 1}
        ),
        window=window.limit,
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
                "one per predictor row, with the population, outcome and bottom line around it"
            ),
        },
        "source_bundle": f"../{result.bundle.name}",
        "notes": (
            f"{', '.join(ROW_LEVEL_KEYS)} describe a predictor row: a {CONCEPT_LEVEL} "
            f'chunk carries "" for all three, so a filter on any of them must either '
            f'allow "" or select chunk_level == "{PREDICTOR_LEVEL}". Missing values '
            f'are "" everywhere, never null; n is an integer when the paper reported '
            f"one. A chunk too long for the embedding window is split into chunk_parts "
            f"parts numbered by chunk_part, each repeating the paper title, so (id, "
            f"chunk_index) identifies a chunk and (id, chunk_index, chunk_part) "
            f"identifies a row of this store; retrieving several parts of one chunk is "
            f"expected and they are contiguous text in that order."
        ),
    }
    return str(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100))


# Every metadata key a chunk can carry, declared for a consumer building a filter.
_METADATA_KEYS = (
    *REQUIRED_KEYS,
    *ROW_LEVEL_KEYS,
    "chunk_level",
    "chunk_part",
    "chunk_parts",
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
