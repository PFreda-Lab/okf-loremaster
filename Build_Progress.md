# Build Progress

Reverse-chronological. One entry per build step or material change, datestamped.
Measurements, rejected alternatives, and forensics live here — **not** in `CLAUDE.md`.

Build steps: 0 scaffold · 1 config/router/events · 2 NCBI clients · 3 schemas ·
4 charter→rank · 5 screen→curate · 6 fulltext→reconcile · 7 emit→validate · 8 vectors ·
9 AFCE docs + TUI · 10 export/inspect/README/example.

---

## Start here — the build is finished

**Steps 0–10 are all done and green.** 1425 tests pass offline in 5m18s; `ruff check src/ tests/`
and `mypy src/` (69 files, strict) are clean; all seven commands run. Everything below this line is
history. There is no next build step; this block is what is left over.

**Three things are still open, and all three are the user's call.** None is ever done unprompted:

1. **The license.** `pyproject.toml` declares none and `README.md` says "not yet chosen". Until one
   is set the package is unlicensed and internal.
2. **`git init`.** Deferred by the user until the build finished. It has.
3. **The folder rename** (`Loremaster-OKF` → something else). Also deferred. Safe to do, but the
   editable install records this directory's absolute path — after a rename, run
   `conda run -n okf-loremaster pip install -e .` from the new location or imports stop resolving.
   Nothing else depends on the folder name.

**Nothing has ever been run against live PubMed.** Every test, every fixture and the step 10 gate use
`tests/fake_ncbi.py` or the recorded cassette. The first real run needs `.env` filled in
(`okf-loremaster init`), and `--dry-run` first is the cheap way to see what it will cost.

**If you are picking up maintenance rather than a step:** `CLAUDE.md` holds the invariants,
`README.md` holds the contracts a consumer may rely on, and the entries below hold the measurements
and the alternatives that were rejected and why.

---

## 2026-08-03 — README: the schematic redrawn, and a pause documented in the wrong place

Post-step-10 fix, prompted by the diagram being hard to follow on GitHub.

**A real error, not just a layout complaint: pause 2 was drawn and described after *curation*.** It is
after **`rank`** — the graph is compiled `interrupt_after=["charter", "rank"]`. The placement is the
entire point of that pause: it lands *before* `screen`, which is the highest-volume call in the run,
so approving the pool is what bounds the bill. Documented after curation, it reads as a formality
that happens once the money is already spent. Fixed in the diagram and in the prose, which now also
says why the pauses are interrupts rather than prompts inside a node.

**The ASCII schematic is now Mermaid**, which GitHub renders natively. Amber nodes are agents, gray
is code, blue is you — so "which of these is an agent" is carried by color and an explicit
`· DEEP agent` label, rather than by a bare role tag the reader has to decode.

- **Laid out as four left-to-right rows stacked top-to-bottom**, not one vertical column. Thirteen
  nodes in a single column rendered 3200 px tall — a scroll marathon nobody reads. Rows bring it to
  1584 × 823. A pure `LR` flow is the other trap: it comes out wide enough that GitHub scales the
  text down to unreadable.
- **First draft put the table's detail in the node labels**, which is what made the boxes tall. The
  table below the diagram already carries the detail; the diagram carries the shape.
- **`p2` must live inside the loop subgraph.** Outside it, with `rank` and `screen` inside, Mermaid
  pushed it out to the far left and drew a long lateral edge across the figure.
- Rendered with `@mermaid-js/mermaid-cli` and inspected before committing rather than pushed and
  eyeballed on GitHub — `<b>` and `<br/>` both survive, and the cluster fill needed overriding
  because Mermaid's default yellow fights the amber agent nodes.

**`search` is now described accurately.** It had read "writes the PubMed query plan", which hides the
two-stage split: `charter` (DEEP) produces each shelf's **seed terms**, and `search` (MID) turns
those into real queries. Both the diagram and the node table now say so, and both record that a
re-query round is `queries.gap_plan` — built **in code** from the curator's gap list, not a second
agent call.

**Caveat for whoever publishes this:** `pyproject.toml` sets `readme = "README.md"`, and PyPI does
not render Mermaid — it would show the block as source. Private and unpublished, so GitHub
readability wins for now; revisit at publish time.

---

## 2026-08-03 — Step 10: `export`, `inspect`, the README, and the downstream gate

**Status:** complete, and the build with it. The gate is met: a finished bundle dropped into a
`resources/` directory resolves under FE_Demo2's real `okf_*` resolver — `okf_list_domains`,
`okf_browse`, `okf_search` and `okf_read` all return, and all three reference forms of AFCE §6.4.1
rule 6 read the same document. Suite 1396 → 1425 offline, 238 s → 318 s; `ruff` and `mypy --strict`
(65 → 69 files) clean. `_pending` is gone — every declared command now does something.

### What shipped

| Module | Holds |
|---|---|
| `emitters/export.py` | `export_bundle`, `ExportResult` — a filtered copy that is a bundle in its own right |
| `okf/overview.py` | `read_overview`, `BundleOverview`, `ShelfOverview` — what `inspect` counts |
| `ui/overview.py` | `render_overview` — what `inspect` prints |
| `okf/markdown.py` | `cell` / `table_row` / `table_rule` / `facts` / `inline` — the writer both emitters share |
| `okf/reader.py` | `OkfDocument.n`, `.full_text`, `.export_safe` — readers that were about to be duplicated |
| `tests/test_export.py` | 17 tests: the filter, the refusals, the round trip, both commands |
| `tests/test_afce_contract.py` | 7 tests: a consumer written from the spec, not from our writer |

### Design decisions

- **The exporter copies retained documents byte for byte** rather than re-rendering them. The body
  holds quotes reproduced exactly as published; a re-render is precisely where "exactly as published"
  stops being true, and it would not fail — it would just quietly differ.
- **A document is kept only when its `export_safe` flag and its recorded license agree.** They are
  derived from each other at emit time, so disagreement means a hand edit. Whether to hand something
  to a third party is the wrong question to answer optimistically, so the export takes the
  conservative side and names the file in a warning rather than dropping it silently.
- **The catalog is filtered, never rebuilt.** `unmapped_vocab` exists only there — deliberately not in
  frontmatter — so a rebuilt catalog would lose it with no way to notice.
- **The copy gets its own descriptor `id`** (`<source-id>-permissive`) and carries `derived_from`,
  `export_filter` and `omitted`. Two corpora sharing a resource id is a collision a consumer cannot
  detect. The `vectors` pointer is dropped for the same class of reason: the store embeds documents
  the filter just removed, so a pointer to it would be a lie about what is in the copy.
- **An emptied shelf keeps its directory and an index that says why.** "No papers survived the
  filter" and "this shelf does not exist" are different claims, and only one of them is true.
- **The export refuses before it writes anything** — into itself, into a parent of itself, into a
  non-empty directory, into a file, or with a filter that would keep nothing. All five are checked up
  front, so a refusal never leaves a half-written bundle behind.
- **`inspect` reads the catalog as its spine and the documents for what only they carry.** Shelf
  sizes, designs, sample sizes and tags come from `_catalog.jsonl`, because that is the file a
  downstream consumer reads and summarizing something else would summarize the wrong thing.
  `text_basis`, `export_safe` and the state of each effect size are only in the files. The last is
  the point of the command: `effect sizes: 11 verified, 13 row(s) reported none` is the one line that
  says how much of a corpus can be quoted with a number attached, and it exists nowhere but in the
  `Effect` column of each table.
- **A shared `okf/markdown.py` instead of a second private `_cell`.** Two writers now produce tables
  that `reader.markdown_table` has to read back cell for cell. Two copies of the escaping would be two
  places for it to drift, and the drift would be invisible: a table that round-trips wrong still
  renders fine.

### Three defects caught during the step

- **`"full text"` vs `"full_text"`.** Both new modules compared `text_basis` against the prose
  spelling. `TextBasis.FULL_TEXT` is `"full_text"`, so every document in every corpus would have
  counted as abstract-only — no error, no warning, just a wrong number in a summary and a wrong split
  in an export. Fixed by naming the value once (`layout.FULL_TEXT_BASIS`), routing both modules
  through `OkfDocument.full_text`, and pinning the two spellings to each other in a test rather than
  to a literal.
- **`export_safe` is the string `"true"`/`"false"` on disk, and `"false"` is truthy.** Every flat
  scalar is quoted, including bools. Anything that reads the key and tests it for truth exports the
  entire bundle while reporting it as filtered — which is a redistribution of documents that may not
  be redistributed, with a reassuring message. `OkfDocument.export_safe` compares as a string and
  documents why; `test_export.py` asserts `0 < kept < TARGET`, which is what catches it.
- **`Counter.update(dict)` adds the dict's values, not its keys.** `inspect`'s vocabulary tally
  called it on a fact list, so the "papers" column rendered as `A00A00A00A00…` — the ICD-10 codes
  concatenated 24 times. It rendered without erroring, which is how it survived to be seen. The test
  now asserts the exact pair, `("icd10", 24)`.

### The gate

The bundle was built from the synthetic corpus, copied to `resources/okf/`, and read by FE_Demo2's
own `tools/impl/okf.py` — loaded from its path with a stubbed `config` module, read-only, nothing
written into FE_Demo2. All four tools resolved: four domains with the right counts, a browse listing
matching the shelf, a fuzzy search returning hits, and `okf_read` finding the same document by
`'10000'`, `'10000_Author00.md'` and `'alpha/10000_Author00.md'`.

**One field degrades, and it is theirs rather than ours.** FE_Demo2's builder stamps `domain_title`
into every document, so its `okf_read` returns `domain_title: ""` for our files. AFCE §6.4.1 rule 5
puts that value in the shelf's `index.md` and rule 8 puts it in the descriptor, which is where we
write it — and the spec's own frontmatter example does not carry it. Duplicating a shelf title into
250 documents would also go stale the moment a shelf is retitled. Left as is, and recorded here.

**The gate is also a test, not only a one-off script.** `tests/test_afce_contract.py` re-implements
the consumer from the spec — its own dependency-free line parser, its own three-form resolver, its
own title+description+tags+journal haystack — and checks a finished bundle against it. Importing
FE_Demo2 from the suite was rejected: it would break when that repository moves, and it would prove
the bundle matches one implementation rather than the contract. Every rule it covers fails
*silently* on a real corpus, which is why it is worth a test at all.

### The README

Rewritten as a GitHub shopfront, to the user's brief of 2026-08-03: scope first (*this feeds
downstream feature-construction agentic systems in the clinical and EHR domain*, and the five
specific ways an abstract dump fails those systems), then the bundle with a real annotated document,
then a schematic of the agentic system with an agent-vs-code table for all thirteen nodes, then the
interface, then every command with every flag, then the two output contracts — the eight OKF rules
and the Chroma metadata including the concept-chunk `""` caveat — then a worked example that ends in
`cp -R` into a downstream `resources/`.

The `inspect` sample in it is the synthetic corpus's real output, reproduced verbatim and labeled as
such. Dressing it up as a clinical run would have meant inventing numbers; the one invented number
that did slip into a draft (an `n` on the cited AIDS Care paper) was removed.

---

## 2026-08-03 — Step 9: AFCE doc updates + the Textual TUI

**Status:** complete. Both halves of the gate met — AFCE §6.4.1 now describes the format we actually
ship, and `--tui` runs a build full screen where `q` stops the run, flushes the checkpointer, and
prints the `--resume <id>` that continues it. Suite 1384 → 1396 (11 new in `test_tui.py`; the 12th is
`test_domain_agnostic.py` picking up `ui/tui.py` by itself), offline, 238 s; `ruff` and
`mypy --strict` (64 → 65 files) clean. `[tui]` installed into `okf-loremaster` and nowhere else:
textual 8.2.8.

### The AFCE half

Nine describe-what-shipped edits across `specs/6.4-resources.md`, `AFCE_BUILD_PLAN.md`,
`specs/15-paper.md` and `CLAUDE.md`. The optional-frontmatter table now lists `license`,
`export_safe`, `text_basis`, `n`, `study_design`, `generated`, `verified` and `sources`; the
structured-body example is a real emitted document; the distribution caveat is weakened in the good
direction, because `license` is recorded verbatim from BioC and `export_safe` is computed, so a
`--permissive-only` subset is genuinely shippable.

- **Loremaster appears three times in AFCE, always as an example, never as a dependency.** The user
  was explicit: people will use AFCE without ever running this tool. Each mention reads "one example
  of an external producer that emits this shape", `afce build-okf` stays first-class for the
  PDF-directory case, hand-authoring stays equally valid, and §6.4.1 closes with "AFCE detects any
  conforming directory identically and depends on no producer in particular." The plan's own wording
  ("name Loremaster-OKF as the upstream producer") would have made it *the* path; it is one path.
- **The frontmatter list was checked against `okf/layout.py` before it was written**, and the
  `resource_descriptor.yaml` paragraph against what `emitters/okf.py::descriptor` and
  `emitters/vectors.py::index_descriptor` actually write. A spec that describes an intention rather
  than an artifact is the failure mode this step exists to close.
- **The example's quotes are reproduced exactly as published**, typography and all, and the spec now
  says why: a cleaned-up quote is worse than none, because the quote is what lets a reviewer check an
  extracted number against its source.

### The TUI half

| Module | Holds |
|---|---|
| `ui/tui.py` | `LoremasterApp`, `ConfirmScreen`, `TuiPause`, `TuiReviewer`, `build_run_tui` |
| `ui/pauses.py` | `charter_view` / `retrieve_view` — the renderables both front ends show |
| `ui/review.py` | `signoff_view` / `signoff_caption` — the same split for sign-off |
| `run.py` | `RunInterrupted`, `require_textual`, and the `attach` / `pause` / `reviewer` injection points |
| `graph/build.py` | `NODES` — the pipeline in order, for a renderer that wants to draw what has not happened yet |
| `cli.py` | `--tui`, its three refusals, and the stopped-run exit path |

**Design decisions**

- **`Pause` and `Reviewer` became `async`.** Asking a human is I/O, and I/O here is awaited. The
  alternative — keeping them synchronous and bridging with `app.call_from_thread` — would have put
  the graph on a second thread, and `EventBus.emit` is `put_nowait`, which is not thread-safe. One
  loop, one thread, no bridge. The cost is `async def` on four implementations and five call sites;
  it bought the whole TUI without touching the graph.
- **Every decision surface is a function returning renderables.** `charter_view`, `retrieve_view`,
  `signoff_view` build a `list[RenderableType]`; the console prints them and the modal mounts them
  as `Static` widgets. Two front ends, one description of what a pause *says* — so a bug in the
  charter summary is fixed once, and the two can't drift into showing different things.
- **The TUI is a third subscriber, not a second way to run.** `build_run` gained `attach`, `pause`
  and `reviewer` parameters, each defaulting to what it already did. `--tui` supplies all three.
  `attach` was the right seam because it preserves the existing lifecycle exactly: it returns a task
  that ends when the bus closes, which is the contract `_start_renderer` already satisfied. **Nodes
  never print** stays true by construction rather than by discipline.
- **`q` cancels the worker; it does not kill the process.** The run is a Textual worker in the app's
  own loop, so cancelling it raises `CancelledError` inside `run_build` — a `BaseException`, so it
  passes straight through `except Exception` and unwinds `async with checkpointer(...)` cleanly. The
  saver flushes on the way out, which is what makes the reported id worth anything. Cancelling while
  a modal is up is safe for the same reason: the cancellation unwinds `push_screen_wait`.
- **The run id is taken off `RunStarted`, not invented.** It is the thread id the checkpointer uses
  and the one `--resume` wants; deriving it a second time in the app would be a second place to be
  wrong.
- **A stopped run is exit code 130 with a resume hint, not a failure.** `RunInterrupted` carries the
  id; `cli.py` prints `okf-loremaster build --resume <id>` in the same breath. 130 because to
  whatever ran us this is the same event as a Ctrl-C.
- **The app stays open after the run finishes** ("finished — press q to close") rather than
  self-closing on the last event. Closing would wipe the log the moment it became worth reading.
- **`--tui` has three refusals, and two of them are notes rather than errors.** `--json` is a hard
  refusal (a full-screen app and a machine-readable stream cannot both own stdout); `--dry-run` and
  a non-terminal both print a one-line note and fall back to the console renderer. A missing `[tui]`
  extra raises before the first search, like `--index` does — the same reasoning as step 8.
- **`NODES` is declared in `graph/build.py`, not read off the compiled graph.** `StateGraph.nodes` is
  a dict in insertion order, not topological order, and a renderer should not have to build a graph
  to draw a list. `test_tui.py` asserts the two agree, so a node added to the graph and not to
  `NODES` fails a test instead of silently never appearing on screen.
- **`q` is re-bound on the modal.** `ModalScreen` sets `_modal = True`, which blocks App-level
  bindings — so without the second binding, the one key documented as always available would be dead
  in exactly the state a user is most likely to want it. The modal also carries a `q stops the run
  and checkpoints it` hint line.

**Testing** — `tests/test_tui.py`, 11, driven with Textual's `run_test()` against the same fake NCBI
corpus the dry-run tests use, so what is exercised is a real graph run answered through the real
modal. A full run approving both pauses reaches the same state a console run does; declining via
`escape` stops without failing and leaves `charter.yaml` on disk; `q` leaves `checkpoints.sqlite` and
a usable run id; `build_run_tui` raises `RunInterrupted` when stopped and re-raises the real
exception when failed; and five CLI tests cover the flag's refusals and fallbacks.

**Forensics**

- **`_attach` was an accidental override of `MessagePump._attach`.** mypy caught it as a Liskov
  violation, which is the friendly version of the bug — a method named for what it does to the bus,
  landing on a Textual internal named for what it does to widgets. Renamed to `_subscribe`.
- **Rich was eating the one word the user needs to type.** `pip install 'okf-loremaster[tui]'` prints
  as `pip install 'okf-loremaster'` — Rich parses `[tui]` as a markup tag and drops it. The
  `[vectors]` hint from step 8 had the same defect and nobody had noticed. Fixed in the shared error
  handler with `rich.markup.escape`, so every message that names an extra survives being printed.
  The test asserts against ANSI-stripped output, because `CliRunner` interleaves color codes between
  the escaped brackets.
- **Widgets are torn down when the `run_test()` context exits**, so `query_one("#log")` afterward
  raises `NoMatches`. Read anything off the app *inside* the context; assert on plain attributes
  after.
- **`ruff --fix` and `pytest.importorskip` disagree about import order.** The skip has to run before
  `from okf_loremaster.ui.tui import ...`, so the imports below it are late by construction; ruff
  sorts them into a second block and flags the `# noqa: E402` as unused, because E402 is not enabled
  here. Let it have its way — the noqa was cargo.

**Cost of the new coverage:** 237 s → 238 s. Eleven Textual tests for one second, because
`run_test()` is headless and the pilot never waits on a real frame.

**Next:** Step 10 — `export`, `inspect`, the README, and one worked end-to-end example.

---

## 2026-08-03 — Step 8: `index_vectors` — the derived Chroma store

**Status:** complete. Gate met — a real Chroma store on disk, a `resource_descriptor.yaml` carrying
the **resolved** model id and revision, no null metadata anywhere (asserted over every key of every
chunk), and concept chunks carrying `""` for the three row-level keys. Suite 1360 → 1384 (22 new;
the other 2 are `test_domain_agnostic.py` picking up the two new modules by itself), offline, 237 s;
`ruff` and `mypy --strict` (62 → 64 files) clean. `[vectors]` installed into `okf-loremaster` and
nowhere else: chromadb 1.5.9, sentence-transformers 5.6.1, torch 2.13.0.

**Built**

| Module | Holds |
|---|---|
| `emitters/vectors.py` | Chunking, `Embedder` and `VectorStore` protocols, `ChromaStore`, `build_index`, `index_descriptor`, `link_index` |
| `graph/nodes/index_vectors.py` | The last node; degrades to a warning, never fails a run |
| `okf/layout.py` | `DISTANCES` — the metric names both sides may say |
| `okf/validate.py` | `_check_vector_index` — six warnings about a store that disagrees with its bundle |
| `run.py` | `embedder(settings)` — resolves the model, and refuses early if the extra is absent |
| CLI | `index <bundle>`, and `build --index` |

**Design decisions**

- **The store is a sibling, `<bundle>.chroma/`, not a subdirectory.** `read_bundle` treats every
  directory at the bundle root as a shelf, so an index inside would validate as a shelf with no
  papers. It is also binary, so anything that copies a bundle would drag it along, and it is derived,
  so deleting it costs a rebuild and nothing else.
- **Two chunk levels, and the concept chunk omits the predictor table.** Not "the whole document":
  the table is already covered row by row, and including it twice both dilutes the concept
  embedding and makes the same sentence win twice in a result set. Each row chunk carries the
  population, outcome, design, N and bottom line around it, so a row retrieved alone still means
  something.
- **chromadb and sentence-transformers are imported inside the functions that use them.** The
  module is therefore importable without `[vectors]`, which is what lets the graph wire
  `index_vectors` in unconditionally instead of branching its shape on an installed package. The
  node returns `{}` when no embedder was built.
- **The missing-extra check runs at run assembly, not at the last node.** `run.embedder` checks
  `importlib.util.find_spec` for both packages and raises a `ConfigError` naming the fix. A run that
  spends an hour and a real amount of money before discovering it cannot do the last step is the
  failure mode worth designing out; `--index` now fails in a sentence, before the first search.
- **Metadata is never `None` — Chroma rejects it — but `n` is an `int` where the paper reported
  one.** A numeric filter has to work on the one field anybody would filter numerically; everything
  else missing is `""`.
- **`timing`, `confidence` and `evidence_type` are `""` on a concept chunk, and that is said in
  three places**: the descriptor's `notes`, the README (as a blockquote), and the CLI's own output
  after `index` runs. A filter on any of them that neither allows `""` nor selects
  `chunk_level == "predictor"` silently drops half the corpus and looks like it merely found less —
  the quietest failure in the whole bundle, so it is stated where each audience will actually be.
- **The distance metric is declared, never defaulted.** Chroma's default is L2; a consumer that
  guessed wrong gets a different order and no error at all. `DISTANCES` lives in `okf/layout.py`
  because the validator checks the same list — writer and checker must agree on which names are even
  sayable.
- **Row labels are derived from `PREDICTOR_COLUMNS`, not restated.** The index parses the predictor
  table back out of a finished bundle, so a column renamed on the writing side alone would produce
  chunks whose metadata is silently empty. One tuple, read by both.
- **Every index finding is a warning, never an error.** The store is derived and rebuildable; a
  bundle whose index is stale or absent is not a bad bundle. Making it an error would let a
  regenerable artifact fail the OKF gate.
- **Embedding runs on `asyncio.to_thread`, one batch at a time, with progress emitted between
  awaits.** `EventBus.emit` is `put_nowait` and is not thread-safe, so nothing may emit from the
  worker thread. Batching is what makes the seam natural rather than a workaround.
- **An unresolved revision is reported empty, not guessed.** `SentenceTransformerEmbedder` asks
  `huggingface_hub.snapshot_download(..., local_files_only=True)` and accepts the answer only if it
  is 40 hex characters; otherwise the descriptor carries `""` and the run warns. A descriptor that
  states a revision it did not verify is worse than one that admits it does not know.
- **A rebuild deletes the collection and recreates it.** An upsert would leave chunks belonging to
  papers a later curation dropped, and those chunks would keep answering queries with no document
  behind them.
- **`index_vectors` runs after `validate` even when the gate failed**, and turns any exception into
  a warning. The bundle is already written; refusing to index it, or crashing on the way out, would
  destroy nothing but would cost the user the whole run's output for a defect in a derived artifact.

**Testing** — `tests/test_vectors.py`, 22. A `StubEmbedder` (deterministic hash vectors, 8 dims, a
fake 40-char revision) and a `RecordingStore` carry most of them, so the suite stays offline; a
sentence-transformers download is network and `conftest.py` blocks it. Chunk counts and levels; the
concept chunk not containing the table; a row chunk standing alone; quotes keyed to row `#`; no null
metadata; the four required keys and unique ids; `""` on concept chunks; `n` int-or-`""`; every
descriptor field; the bundle↔index cross-pointers; sibling-not-shelf layout; an unresolved revision;
an empty bundle reported as not indexed; two validator injections (a pointer at a store that is not
there, a bad distance); the node recording the resolved model in the manifest; and four CLI tests.
Three of them run **real Chroma** — accepted metadata, `hnsw.space` still `cosine` after reopening
the store, and a rebuild replacing rather than appending.

**Forensics**

- **chromadb 1.5.9 ships its own types and mypy checks them.** Two rewrites came out of that:
  `configuration` is a `CreateCollectionConfiguration`, not a plain dict, and `Collection.add` wants
  `Sequence[float]` rows rather than `list[float]`. Both are the library being right — worth
  recording because the obvious code typechecks nowhere.
- **`ruff --fix` quietly deleted the `PREDICTOR_COLUMNS` import** while sorting `__all__`, because
  at that moment the column names were restated as literals a few lines below. The fix was not to
  re-add the import but to derive the labels from the tuple, which is what removed the possibility of
  the desync in the first place.
- **A CLI test written `async def` failed with "asyncio.run() cannot be called from a running event
  loop."** `index` calls `asyncio.run` itself, so the test that drives it cannot already be inside a
  loop; it is synchronous now and calls `asyncio.run(golden(...))` to build its fixture. The
  `validate` CLI tests are async and fine — `validate` never starts a loop. Worth knowing before
  writing the next CLI test.

**Cost of the new coverage:** 180 s → 237 s. Three tests exercise real Chroma and account for most
of it; the rest drive full graph runs, as every end-to-end test here does.

**Next:** Step 9 — AFCE doc updates and the Textual TUI.

---

## 2026-08-03 — Step 7: `emit_okf → validate`, plus `--review` sign-off

**Status:** complete. Gate met — the golden bundle passes the hard gate with zero errors and zero
warnings, the warnings block prints, and `--review` writes
`verified: [{by: "human:tester", at: "...Z"}]` on one line. Suite 1298 → 1360 (50 new tests; the
other 12 are `test_domain_agnostic.py` parametrizing over `src/**.py` and picking up the new
modules by itself), offline, 180 s; `ruff` and `mypy --strict` (50 → 62 files) clean.

**Built**

| Module | Holds |
|---|---|
| `okf/layout.py` | Every filename, reserved heading and load-bearing key, stated once |
| `okf/frontmatter.py` | `render` / `parse` / `split` / `stamp` — the single writer *and* reader |
| `okf/reader.py` | Reading a bundle back off disk; never raises, collects `problems` |
| `okf/validate.py` | `validate_bundle`, `Finding`, `Severity`, the vocabulary aggregate |
| `emitters/okf.py` | The whole bundle: documents, shelf indexes, root index, log, catalog, descriptor |
| `review.py` | `Reviewer` protocol, `Signoff`, `NoReview`, `signer_id` |
| `ui/review.py` | `ConsoleReviewer` — prints a rendered specimen, then asks |
| `graph/nodes/{review,emit_okf,validate}.py` | The three new nodes |
| CLI | `validate <bundle>`, `--review`, and a nonzero exit when the gate fails |

**Design decisions**

- **The flow-style discipline is enforced in one module, not asked for at call sites.** There is
  exactly one way to get it wrong and it is invisible in a diff. `parse` is deliberately *stricter*
  than `yaml.safe_load` — it refuses an indented continuation, a keyless line, a duplicate key —
  and `validate` runs every block through both readers and requires them to agree. That agreement
  is the only check that actually proves the discipline held: a block that reads one way for a spec
  consumer and another for a line-parser is a bundle that means two different things.
- **Every flat scalar is quoted, numbers and booleans included.** This resolved a real
  contradiction: the emitter was writing `n: 1454` and `export_safe: true` while the validator
  rejected any value not starting `"`, `[` or `{`. AFCE §6.4.1 rule 7 settles it — "strictly quoted
  flat scalars + string lists", against a dependency-free parser that hands back strings anyway. An
  unquoted `n: 1454` is an `int` to a YAML consumer and `"1454"` to a line-parser, which is the one
  divergence the whole discipline exists to rule out. Quoted, both readers get `"1454"`.
- **An effect size is printed only when verification kept it.** `PredictorRow.downgraded()` strips
  `effect` but keeps `effect_raw`, so printing the raw string unconditionally would put the exact
  fabricated number back into the bundle, in the one place a reader would ever see it — undoing
  step 6 entirely. A stripped magnitude reads `unverified`; a row that never carried one reads `—`.
  The two are told apart by whether the raw string contains digits at all: prose like "not
  significant" is the extractor's own words and no check ever touched it.
- **Nothing is deleted.** Overwriting warns and replaces the files the emitter owns; a stale shelf
  directory from an earlier taxonomy is reported, not removed. A tool that tidies a directory it was
  pointed at eventually tidies the wrong one.
- **An empty shelf still gets a directory and an index saying so.** An absent shelf and a shelf that
  retained nothing are different claims about the literature; collapsing them loses the interesting
  one.
- **`validate` reads the bundle back off disk rather than checking the objects that wrote it.** A
  run validating its own in-memory records proves the pipeline agrees with itself and nothing about
  the files a downstream agent will open — and every defect this gate exists to catch lives in that
  gap. Same code path as `okf-loremaster validate <bundle>`, because it is the same question.
- **Failures are reported, never raised.** A bundle that fails is still on disk and still worth
  looking at; the exit code carries the verdict. Likewise the reader: a malformed file lands in
  `problems` and is simply absent, so one bad document does not hide every file after it.
- **Errors break a consumer; warnings make a bundle worse without making it wrong.** A missing
  `domain`, a `domain` that does not match its folder, a moved section, a dead link — each makes a
  document silently unreachable or misfiled. An untagged document or an empty shelf is a warning:
  AFCE's haystack is title, tags and journal, so untagged is findable only by title.
- **`review` is an unconditional node, not a conditional edge and not a third interrupt.** Routing
  on a dep would make the graph's shape depend on runtime config; the node already answers "nobody
  was asked" by returning an empty update. And the two existing interrupts sit where resuming saves
  real money — `reconcile → emit_okf` is neither slow nor expensive.
- **Declining is not a failure.** The bundle is written without a `verified` block and sits at OKF's
  `unverified` tier, which is what it is. The only thing a decline loses is a claim we were never
  entitled to make.
- **`--review` is refused outright alongside `--yes`, `--dry-run` or `--json`.** Each of those means
  nobody is going to look, and signing anyway writes `by: "human:<id>"` naming a person who never
  saw the bundle. That is a false attestation, not a weak one, so it is refused rather than degraded.
- **The reviewer is shown a whole rendered concept file before being asked.** Counts and a
  verification line say the run behaved; they say nothing about whether the files are any good. The
  specimen is chosen, not sampled — the paper with the most predictor rows shows the most format per
  screen, and a run whose tables are wrong is wrong there first.
- **The bundle directory is decided in `run.py` before the graph starts** and carried on `Deps`, so
  a resumed run cannot write somewhere other than the run it resumed. `run.py` still writes
  `charter.yaml` itself as well, so a run stopped at a pause leaves a file to edit.
- **Verbatim quotes go below the table, numbered against its `#` column.** A quote is a whole
  sentence and a table cell is not where a sentence is readable.

**Testing**

- `tests/test_emit_okf.py` — 17, all against a golden bundle read back off disk rather than the
  objects that wrote it. The hard gate; every flat value quoted and every nested one flow style;
  both parsers seeing the same document; `domain` equal to its folder with "shelf" nowhere in
  frontmatter; section order; the `null_findings` sentinel; quotes keyed to row `#`; catalog
  integrity both directions; and the step-6 tie-in — a run driven by the fabricating extractor must
  show `unverified` where the invented `4.44` would have gone, and must not print the number.
  Sign-off is driven through a `StubReviewer` monkeypatched over `ConsoleReviewer`.
- `tests/test_okf_validate.py` — 27. Frontmatter unit tests (round-trip, newline collapse, empty vs
  `False`, the four line-parser refusals, a missing or unclosed fence, naive vs aware timestamps),
  then **defect injection**: a bundle is built, asserted clean, then broken one way at a time — domain
  mismatch, a bare scalar, a `shelf` key, a moved section, an empty section, a catalog that
  disagrees in either direction, a dead link, a duplicate id, a reserved name, a missing root index,
  a remote embedder. The unmapped-vocabulary aggregate is checked both under and over the 15%
  threshold, and the over case asserts the exact rerun string `--vocab icd10,snomed`.
- `tests/test_cli_validate.py` — 6. Exit codes are the point of a gate: 0 on a good bundle, 1 naming
  `index.md` on a broken one, 1 on a bundle that is not there. Plus the `--review` refusal,
  parametrized over `--yes`, `--dry-run` and `--json`.

**Forensics**

- **The bare-scalar contradiction** above was a live bug carried in from the plan, not a test
  artifact: the emitter and the validator disagreed about the same line, and every bundle would have
  failed its own gate on the first integer. Found by the gate test, settled by reading the
  downstream contract rather than by picking a side.
- `test_an_empty_shelf_is_emitted_and_warned_about` was first written by deleting documents off
  disk, which left the shelf index pointing at files that no longer existed — the test then failed
  on "does not resolve" errors that had nothing to do with the empty shelf. Rewritten as a run whose
  charter carries a shelf the corpus cannot fill, which is how an empty shelf actually arises.

**Cost of the new coverage:** the three files add 108 s to a 180 s suite, because each end-to-end
test drives a full graph run. Acceptable for now — the alternative is a shared session-scoped bundle
fixture, which would let one test's mutation leak into another's assertions. Revisit if the suite
crosses ~5 minutes.

**Next:** Step 8 — `index_vectors`.

---

## 2026-08-03 — Step 6: `fulltext → extract → reconcile`, with deterministic numeric verification

**Status:** complete. Gate met — an injected fake odds ratio is caught end to end: `effect=None`,
confidence downgraded HIGH → MEDIUM, one warning naming both the paper and the number, and the run
finishes with all 24 records and every shelf intact. Suite 1259 → 1298, offline, 68 s; `ruff` and
`mypy --strict` (50 files) clean.

**Built**

| Module | Holds |
|---|---|
| `graph/nodes/fulltext.py` | BioC retrieval, section priority, the length budget, abstract fallback |
| `graph/nodes/extract.py` | One DEEP call per paper, a per-shelf cached prefix, one schema repair |
| `graph/nodes/reconcile.py` | Budgets → verification → vocabulary partition → record assembly |
| `verification.py` | `Quantity`, `Source`, `verify_extraction` — the whole check, no model call |
| `prompts.py` | `EXTRACT_SYSTEM`, `extract_context`, `extract_user` |
| `schemas/` | `PaperText`, `VerificationSummary`, `MAX_SOURCE_CHARS = 24_000` |

**Design decisions**

- **The scope of the check is the text the extractor actually read.** `PaperText.text` is the
  finished prompt block, header and all, and the length budget is applied in `fulltext` rather than
  in `extract`. Checking against text the model was never shown would report correct extractions as
  fabricated — the one failure that would make the check worse than having none.
- **A number that is not in the source is removed, not rejected.** `PredictorRow.downgraded()`
  keeps the predictor, its operationalization and its timing — all of which the paper did report —
  and drops only the magnitude. Discarding the row would throw away good evidence to punish one
  field; discarding the paper would let one bad number cost everything else it said.
- **Precision is asymmetric: a claim may be less precise than its source and never more.** A source
  reading `1.84` supports a claim of `1.8`, which is a rounding. The reverse would let any bare
  integer in the text — a year, a count, a table number — support a claimed effect of `4.44`.
- **A row that quoted its source is checked against that sentence alone.** A full text has hundreds
  of numbers, so a document-wide match is a coincidence waiting to happen; one sentence makes it
  unlikely rather than merely uncommon. A row without a quote falls back to the whole document.
- **`effect` is also checked against `effect_raw`, needing no source at all.** A row claiming
  `effect: 3.91` beside `effect_raw: "1.82 (95% CI 1.21-2.74)"` contradicts itself, and one of the
  two is wrong whatever the paper said. This is what catches a silent unit conversion.
- **Section priority is about where the numbers are**, not page order: title, abstract, results,
  tables and figures, conclusions, methods, discussion, introduction. Discussion and introduction go
  last because they restate other people's papers, which is the fastest route to attributing another
  study's effect size to this one. `REF` and seven other non-content types never reach the prompt.
- **A section that does not fit is skipped, not a stopping point**, so one enormous methods section
  cannot cost a paper its conclusions. Reading order is restored after selection — a prompt whose
  sections arrive shuffled by priority reads as a different paper than the one published.
- **License is recorded verbatim from BioC and left empty on the abstract path.** PubMed serves no
  license with an abstract, and an inferred one is how a bundle becomes undistributable unnoticed.
- **Length budgets run before verification, not after**, so nothing is checked that a budget was
  about to drop and the verification counts describe the bundle rather than the model's reply.
- **One schema repair, not two.** `SchemaError.hint` names the field that was wrong and a model
  told which field it broke usually fixes it; a second failure is a model that cannot satisfy the
  schema, and a third DEEP call to confirm that is the most expensive way to learn nothing.
- **Warnings are one per category, never one per row.** Five offending rows are named as examples
  and the rest are counted, so a run where everything failed says so in one line.

**Testing**

- `tests/test_verification.py` — 18. Ten drive `verify_extraction` on shapes a synthetic corpus
  never produces: a Lancet middle dot, a true minus sign, an interval whose hyphen is not a minus
  (attached, en-dashed and spaced), a hyphenated word, a claim less precise than its source, a bare
  integer against a claimed decimal, `repr(0.00003)` losing its decimals, a quote scoping the check
  to one sentence. Then the gate, as a pair: the same corpus read once faithfully and once by an
  extractor that invents a single odds ratio. **The control matters as much as the gate** — without
  it a check that deleted every number would pass.
- `tests/test_extraction.py` — 16, one node at a time on the paths a healthy run never takes:
  section selection under a monkeypatched budget, an oversized section skipped while a lower-priority
  one below it is kept, the reference list never reaching the extractor, the abstract fallback, a
  paper already read not being fetched again, a reply repaired once, a reply that never parses, no
  router at all, and — the invariant the whole check rests on — that the model is shown byte for byte
  the string verification will later check against.
- `tests/graph_runs.py` — new. The charter, the scripted screening/curation policy, `full_run` and
  `node_deps`, moved out of `test_screening.py` so screening, extraction and verification interrogate
  the same run rather than three that have drifted apart.
- `tests/fake_ncbi.py` — now serves BioC. Two papers in five are open access (matching
  `llm.estimate.OPEN_ACCESS_RATE`, so the fixture's reality and the projection's assumption are one
  claim rather than two guesses); the rest answer the way BioC really answers outside the subset —
  HTTP 200 with a plain-text `[Error]` body — so the trap `bioc.py` exists to contain is exercised on
  every run. Licenses cycle `CC BY / CC BY-NC / CC0 / NO-CC CODE` so both answers of `is_export_safe`
  occur. Every open-access paper prints one real result sentence, and `REFERENCE_ONLY = 8.77` appears
  only in the reference list.
- `tests/fake_llm.py` — a default extractor that parses the numbers back out of the prompt with
  generic regexes rather than importing them from the fixture, so a fabricating extractor is a
  contrast with a working one rather than the only case the suite exercises.

**Forensics — the new tests found two real defects in `verification.py`, both fixed**

1. **`beta −0.44` lost its minus.** `_is_sign` skipped whitespace before deciding, then saw the `a`
   of `beta` and called the dash a hyphen. Rewritten around three cases: attached to a digit or a
   letter it is not a sign (`1.21-2.74` is an interval, `follow-up` is a word); spaced away from a
   digit it is still not a sign (`1.21 - 2.74` is the same interval); anywhere else it is a sign.
2. **The gate itself failed: `assert 4.44 is None`.** `_agree` compared at
   `min(a.decimals, b.decimals)`, so a claimed `4.44` rounded to zero places matched any bare `4` in
   the text — and a paper is full of years, counts and table numbers. Replaced with the asymmetric
   `_supports(claim, found)`, which rounds the source to the claim's precision and never the reverse.
   A production correctness fix, not a test fix: without it the check passed almost everything.

Also: adding 24 BioC fetches per end-to-end run cost ~60 s at the keyless 2.5 rps. The test settings
now carry a placeholder `ncbi_api_key`, purely for the 8 rps limiter it selects — these runs hit an
in-process transport and the courtesy limiter is not what is under test. `test_screening.py` went
from a large share of a 124 s suite to ~23 s.

**Known limitation:** a row with no quote is checked against the whole document, where a
coincidental match is plausible — a fabricated `2.1` in a paper that prints `2.1` anywhere passes.
Accepted: the alternative is dropping every unquoted number, and the extraction prompt asks for a
quote precisely so the strong check is the common one. `VerificationSummary` counts what was
checked, so a corpus of unquoted rows is visible rather than silent.

**Next:** Step 7 — `emit_okf → validate`, plus `--review` sign-off.

---

## 2026-08-03 — Step 5: `screen → curate`, and the conditional re-query edge

**Status:** complete. Gate met — the shelf floor/ceiling property tests pass, 971 of them across
120 seeds. Suite 265 → 1259, offline, 59 s; `ruff` and `mypy --strict` (45 files) clean.

**Built**

| Module | Holds |
|---|---|
| `graph/nodes/screen.py` | One FAST call per pooled paper, global screen budget, borderline flag |
| `graph/nodes/curate.py` | One MID call per shelf, three-tier reserve, `pending_gap_plan` |
| `curation.py` | `enforce_bounds` — ceiling, floor, target, `Placement`, `MAX_ROUNDS = 2` |
| `queries.gap_plan` | Gap → search syntax, no model call; skips a term an earlier round ran |
| `graph/build.py` | `curate → search` conditional edge, `_drive` resume loop |
| `ui/summary.py` | The final shelf table, with `still missing` beside a short shelf |
| `prompts.py` | `SCREEN_SYSTEM` / `screen_user` / `screen_context`, `CURATE_SYSTEM` / `curate_user` |
| `schemas/` | `ScreenVerdict`, `ShelfCuration`, `CurationDecision`, `CurationResult`, `ShelfGap` |

CLI: `build` gains `--screen-budget --shelf-min --shelf-max --max-rounds`; `--max-rounds` above
`MAX_ROUNDS` exits 1 rather than being silently clamped.

**Design decisions**

- **One screening call per paper, not one per batch.** Batching forty abstracts is cheaper per
  token and returns forty verdicts whose alignment to the input is the model's to get right. One
  dropped row shifts every verdict after it onto the wrong paper, silently, in the one node whose
  output nothing downstream can check. One paper per call cannot misalign, and it is the shape
  `llm.estimate` already projects, so the call count printed at the retrieve pause is the count
  the run makes.
- **Curation is per shelf.** The screener saw each paper alone; the curator seeing a shelf whole
  is the only point in the run where "these four report the same result from the same cohort" is
  noticeable. Worth a MID model at a few calls where the screener is FAST at a few hundred.
- **Judgment and arithmetic are separate modules.** `curate.py` asks; `curation.py` fits. That
  split is why shelf sizes are reproducible across runs even though the judgment behind them is
  not — and why the property tests can hammer `enforce_bounds` 971 times without a model.
- **The three bounds are applied in an order that decides which one yields:** `shelf_max` trims
  hard, `shelf_min` backfills from the reserve, `target_papers` trims the widest shelves and is
  the only bound allowed to fail. A floor is a statement that a shelf below it is not worth
  having, so when the floors cannot fit inside the target the target gives way *and warns*.
- **The floor is backfilled from the reserve before it is allowed to become a gap.** Three tiers,
  best first: screener-included papers the shelf had no room to offer, then excluded-but-relevant,
  then papers the curator turned down. A curator's rejection is the best-informed "no" in the run,
  so it is reached for last. A reserve backfill is cheaper and better informed than a search round.
- **The re-query loop is bounded three independent ways:** `rounds` against `max_rounds`, a CLI
  hard cap at `MAX_ROUNDS = 2`, and a gap plan that must contain a query no round has already run.
  The third is the one that matters — without it a shelf that is thin because the *literature* is
  thin re-runs the same searches, reaches the same shortfall, and pays twice.
- **A second round screens only what it has not already screened, and the budget is global across
  rounds.** The retrieve pause is where a person approved a number of papers to spend on; a
  second round that quietly doubled it would make that approval mean nothing.
- **A re-query round re-curates only the shelves that came up short**, and re-offers a gapped
  shelf whole rather than only its new papers — a shelf under its floor is small, so seeing it
  entire costs almost nothing. Shelves that were fine keep their first-round decisions.
- **`gap_plan` drops the population anchor and ORs the curator's phrases beside the shelf's own
  seeds.** The round that came up short already searched the narrow shape. `MAX_GAP_TERMS = 6`.
- **A shelf still under its floor is printed as a finding, not swallowed.** `ui/summary.py` puts
  the curator's own `missing` line beside the count. The alternative is a bundle whose thin
  shelves are discoverable only by browsing it.

**Testing**

- `tests/test_curation.py` — 971 tests. Eight properties over `SEEDS = range(120)`, each seed a
  random charter (1–6 shelves, floor 1–6, ceiling floor+0–8, target 1–`count*ceiling+4`) against a
  60-PMID universe with cross-shelf duplicates, a 25% chance of an invented slug, and PMIDs
  deliberately absent from `rank`. **P1** nothing over the ceiling · **P2** under the floor ⟺
  reported as a gap, with `gap.missing` round-tripping the curator's words · **P3** over target
  only when every shelf is at or under its floor *and* the warning fired · **P4** no paper on two
  shelves · **P5** every placed paper came from that shelf's own `kept ∪ reserve` · **P6**
  determinism · **P7** `total == offered − duplicates − trimmed + backfilled` · **P8** every
  charter shelf present, each list rank-ordered. Plus 11 targeted tests for the tie-breaks.
- `tests/test_screening.py` — 19 tests. Eleven drive the two nodes on the paths a happy run never
  reaches: budget reached, an unreadable reply, a shelf the charter does not have, a curator
  answering about a paper nobody offered, silence about one that was, a failed shelf call, and
  `router=None` on both nodes. Eight run the whole graph twice over.
- `tests/fake_llm.py` — a scripted model that **reads the real prompts**: which node is asking
  comes from the system constant, what it is asking about is parsed back out of the user message.
  A `prompts.py` change that drops the shelf slug, the offered PMIDs or the paper text fails here
  loudly, instead of passing against a transcript that no longer corresponds to anything sent.
  Tests write a policy, not a reply order, so concurrency cannot reorder them into failure.

**Forensics**

- **"The second round ran" is not the claim worth testing; "it asked something new" is.**
  `fake_ncbi` now withholds a 12-paper slice of `delta`, numbered past `PER_TOPIC` so it is
  invisible to `all_pmids` and to every per-topic query. It answers only to `UNLOCK_PHRASE`
  (`"rescue cohort"`), and the only route from a curator to that phrase is `ShelfGap.missing` →
  `gap_plan` → `esearch`. A re-query edge that quietly re-ran the first round's searches comes
  back with the corpus it already had and fails. Rejected alternatives: making the withheld papers
  compete for a capped pool (measures the ranker, not the edge); boosting their RCR (same problem,
  and year-sensitive); withholding *existing* `delta` papers (breaks Step 4's assertions).
  `POOL_SIZE = 200 > 168` so `quota_select` retains everything and the scripted judgment alone
  drives the gap.
- **The end-to-end arithmetic is asserted exactly, not approximately.** Round 1: 160 − 4 dedupe
  drops = 156 screened; offers 24/24/24/2; curator keeps 8/8/8/2 = 26; target trims 2 (widest
  first, ties to the later slug) → 8/7/7/2 = 24, with `delta` 2 short of its floor of 4. Round 2:
  the unlocking query returns the 12 withheld papers, 168 screened with no repeats, only `delta`
  re-curated, ceiling trims its 14 to the 8 best-ranked, target trims 6 → 6/6/6/6 = 24, no gaps.
- **The screener is never shown a PMID, so the scripted policy is not either.** `fake_ncbi.identify`
  reads the topic and index back out of the abstract text. Keying a test policy on the PMID would
  let it recognize papers by something the model never sees.
- **`ScreenVerdict.borderline` is derived, not asked for.** `(not include and relevance >= 2) or
  (include and confidence is LOW)` — a model asked to self-report borderline-ness answers yes
  about everything.
- **A screening call that failed yields `relevance 0` and `borderline False`**, so a paper the
  model never actually judged cannot be pulled into a floor backfill. One aggregated warning per
  batch, plus a second when more than half failed.

**Known limitation:** `max_rounds` counts rounds, not queries. A charter whose shelves all come up
short sends every gap to `gap_plan` in one round, which is correct, but the second round's cost is
bounded only by `max_queries=12` and the global screen budget — not by how many shelves were short.

**Next:** Step 6 — `fulltext → extract → reconcile`, with deterministic numeric verification.

---

## 2026-08-03 — Step 4: `charter → search → dedupe → rank`, both pauses, `--dry-run`

**Status:** complete. Gate met on every clause — zero LLM calls under `--dry-run`, both pauses
print what they were specified to print, `--yes` bypasses both, `--vocab` overrides the charter
and lands in `charter.yaml`, and MMR + quota demonstrably change the retained set with the
comparison printed. 265 tests pass offline in 42 s; `ruff` and `mypy --strict` (41 files) clean.

**Built**

| Module | Holds |
|---|---|
| `graph/state.py` | `RunState` TypedDict, `Deps`, the `span()` context manager |
| `graph/build.py` | `StateGraph` wiring, `AsyncSqliteSaver`, `interrupt_after=["charter","rank"]` |
| `graph/nodes/charter.py` | DEEP call, `--charter` load, `--vocab` override, skeleton fallback |
| `graph/nodes/search.py` | MID query plan, filter application, one `esearch` per query, one `efetch` |
| `graph/nodes/dedupe.py` | Retractions, missing abstracts, normalized-title collisions |
| `graph/nodes/rank.py` | iCite batch, relevance scoring, MMR + per-shelf quota, `selection_diff` |
| `queries.py` | `tiab`/`phrase`, `with_filters`, `deterministic_plan`, `inspect_translation` |
| `ranking.py` | `Weights`, `relevance`, `mmr`, `quota_select`, `SelectionComparison` |
| `llm/estimate.py` | `project_spend` — per-node projection, priced through the router's three stages |
| `ui/pauses.py` | `render_charter`, `render_retrieve`, `ConsolePause` / `AutoApprove` |
| `run.py` | `build_run`, `draft_charter`, `parse_vocab`, `RunOptions` |

CLI: `build` gains `--dry-run --vocab --pool-size --target-papers --yes --json`; `charter`
renders through the same `render_charter` the pause uses, minus the question.

Tests added: `tests/test_dry_run.py` (20, the gate as executable spec) · `tests/test_ranking.py`
(18) · `tests/test_queries.py` (15) · `tests/test_domain_agnostic.py` +15 over the new modules ·
`tests/fake_ncbi.py`, a 160-paper synthetic PubMed + iCite over `httpx.MockTransport`. Suite
197 → 265.

**Design decisions**

- **The pauses are graph interrupts, not prompts inside a node.** `interrupt_after` plus the
  SQLite checkpointer means the state a human reviews is the state on disk, and `--resume` gets
  a real resume point rather than a replay. A node that asked its own question would violate
  *nodes never print* and would not survive a restart.
- **Zero LLM calls is enforced twice.** `test_dry_run` monkeypatches `Router` to raise on
  construction *and* asserts no `LLMCall` event was emitted. Either alone can be defeated by a
  future refactor; both together cannot be passed by accident.
- **The dry-run spend projection measures the corpus, not a guess of it.** Screening cost is
  dominated by abstract length, which varies about fivefold across the literature, and the pool
  is already retrieved by the time the projection runs. Only the prompt overheads for nodes that
  do not exist yet are allowances, and they are named as such in the module docstring.
- **MMR uses lexical Jaccard over title + MeSH + keywords, never the abstract.** An abstract is
  long enough that any two papers in one shelf overlap heavily on function words and method
  boilerplate, which flattens the similarity matrix exactly where diversity is supposed to
  discriminate. `λ = 0.7`.
- **Diversification is reported as a delta, not asserted.** `selection_diff` returns the
  per-shelf counts under pure relevance rank and under MMR + quota, plus which shelves were
  helped. If the two selections agree, the pause says so; a claim that cannot come out false is
  not evidence.

**Forensics**

- **The bad-field-tag test could not be made to fire on the recorded term, and that is
  correct.** `inspect_translation` only flags an `[All Fields]` rewrite when nothing else
  explains it — `untagged_clauses(term) == []`. The recorder searched
  `postoperative respiratory failure[nosuchfield]` *unquoted*, so `postoperative respiratory`
  is untagged and automatic term mapping accounts for the rewrite on its own. Rather than
  re-record eight interactions over the network for one test's convenience, the test asserts
  both branches against the same recorded response: the bare term is correctly not flagged, and
  the quoted shape `tiab()` actually emits *is*. The recorded numbers stand — 14,382 hits
  against 79 for the clean query, with an empty `errorlist` either way.
- **`retmax` is part of the cassette key.** `CLEAN_RETMAX = 10` / `BAD_FIELD_RETMAX = 5` are the
  values the recorder used, not free choices; a mismatch is a `CassetteMiss`, not a stale value.
- **The first live dry run returned 0 hits on all 6 queries.** The charter's `population` was
  `adults undergoing major noncardiac surgery`, which becomes one exact-phrase `[tiab]` clause
  and matches nothing. Two fixes, because either alone leaves the failure silent: `CHARTER_SYSTEM`
  now states that `population` and `outcome` are searched verbatim as quoted phrases and must be
  ~4-word noun phrases a paper would print; and `search.py::_report_empty` warns when queries
  come back empty, naming exact-phrase matching as the cause. PubMed reports a zero-result search
  as a complete success, so without the warning the run simply arrives at the retrieve pause with
  an empty pool and no account of itself.
- **Pure relevance rank does not sweep a single shelf, and the test says the true thing.** On the
  synthetic corpus, citation impact is stacked by topic and everything else is uniform, yet pure
  rank still lands `{alpha: 15, beta: 14, gamma: 7, delta: 4}` — `position` (weight .30, spread
  .24) dominates `citation` (weight .15, spread ~.12). The assertion was rewritten to the
  measured property (lopsided before, level at the quota after, `shelves_helped` == the starved
  ones) rather than rigging the corpus to produce a tidier number.
- **A bare `Callable[[RunState], Awaitable[...]]` makes mypy infer `NodeInputT = Never`.**
  LangGraph's `add_node` overloads key on a *named* `state` parameter. Fixed with a `BoundNode`
  callback `Protocol` as `_bind`'s return type.
- **`--yes` and `--dry-run` get a non-interactive `ConsolePause`, not `AutoApprove`.** Both still
  render everything; they only skip the question, and print "not asking" where the prompt would
  be. `--json` gets `AutoApprove` because there is no console to render to. Silently rendering
  nothing would make `--dry-run` useless for the one thing it exists for.

**Verified live** (`build --charter /tmp/okf-demo/charter.yaml --dry-run --pool-size 60`):
6 queries · 13 hits · 8 unique · 8 pooled; charter pause printed the shelf taxonomy,
`vocabularies  icd10, cpt, loinc` and the `--vocab` hint; retrieve pause printed the query table,
totals, top titles, the shelf-affinity delta, and a projected spend of $0.2183 over 21 calls
priced by LiteLLM; final meter `tokens 0 · cost $0.00`. Diversification reported "changed
nothing" because only 8 candidates were retrieved, well under the pool size — the measurable
demonstration is `test_mmr_and_the_quota_change_the_retained_set` against the 160-paper fake.

**Known limitation:** `deterministic_plan` ANDs the charter's parts, so a `--dry-run` on a narrow
charter retrieves far less than the real run's MID-planned queries would. This is deliberate — a
dry run must not call a model — but it means the projected spend is a floor when the pool comes
back below `--pool-size`. The pause prints the pool size it measured, so the shortfall is visible
rather than folded into the estimate.

**Next:** Step 5 — `screen → curate`, with the conditional re-query edge capped at 2 rounds.

---

## 2026-08-03 — Step 3: `schemas/` — every typed object that moves between nodes

**Status:** complete. Gate met — *"`mypy --strict src/` clean."* 26 files, no issues.

**Built**

`src/okf_loremaster/schemas/`, 1601 lines across nine modules:

| Module | Holds |
|---|---|
| `common.py` | `Model` base, `Slug`, `Confidence`, `EvidenceType`, `Direction`, `TextBasis`, `is_export_safe`, `slugify`, `filename_token` |
| `limits.py` | Length budgets and the pure functions that enforce them |
| `charter.py` | `Shelf`, `Charter` — YAML round-trip, `digest()`, advisory `problems()` |
| `candidates.py` | `PlannedQuery`, `QueryPlan`, `ExecutedQuery`, `Candidate`, `ScoredCandidate` |
| `screening.py` | `ScreenVerdict`, `CurationDecision`, `ShelfGap`, `CurationResult` |
| `concept.py` | `PredictorRow`, `NullFinding`, `Extraction`, `ConceptRecord`, `partition_vocabulary` |
| `manifest.py` | `BundleCounts`, `ShelfSummary`, `CostSummary`, `RunManifest` |
| `parse.py` | `SchemaError`, `extract_json`, `parse_model`, `repair_hint`, `response_format_for` |
| `__init__.py` | 46 public names |

Tests: `tests/test_schemas.py` (78) · `tests/test_domain_agnostic.py` (32). Suite now 197,
still fully offline, 12.9 s.

**Design decisions**

- **`Extraction` split out of `ConceptRecord`.** Everything under `record.extraction` came from
  a model; everything beside it was read verbatim from an API or decided by the pipeline. No
  reader has to guess which is which, and the emitter writes bibliographic frontmatter without
  ever consulting model output.
- **Two invariants became code, not prompt text.** A `model_validator` inserts the
  `none reported` null-findings sentinel, so omission is *impossible* rather than *checked* — a
  missing section and a section reporting nothing render identically as an empty table, which is
  exactly the failure a prompt instruction cannot catch. `CostSummary` refuses to persist a
  display string that reads as complete when calls were unpriced.
- **Budgets truncate and warn; they never reject.** An over-long extraction is a good one that
  ran on, and re-asking costs a DEEP call to fix formatting. Dropped predictor rows are named in
  the warning, because silent truncation is indistinguishable from a paper that reported less.
  Rows are cut from the tail: the model's ordering is its judgment of importance.
- **`p_value` is a string.** `<0.001` and `NS` are how papers report it; a float loses the
  difference between "very small" and "not given".
- **`PredictorRow.downgraded()` keeps the claim and drops the number.** Numeric verification in
  step 6 nulls `effect` and the CIs, lowers confidence one step, and leaves `effect_raw`
  visible. Discarding the row would throw away a real reported relationship to punish one field.
- **`Candidate.screening_text` excludes MeSH.** Indexing lags publication by months, so
  including it would systematically favor older papers in a judgment about content.
- **`CostSummary.display` is stored pre-rendered** by the router's `format_cost`. The manifest
  then cannot disagree with what the run printed, and `schemas` does not import `llm.router`.
- **`schemas` depends on `clients`, never the reverse.** `Candidate.from_record` /
  `with_metrics` / `with_concepts` adapt `PubMedRecord`, `CitationMetrics`, `AnnotatedDocument`.

**Two defects caught in the writing**

1. **A pydantic field named `construct` shadows `BaseModel.construct`.** mypy flagged it as an
   `[assignment]` error; a runtime check confirmed pydantic emits `UserWarning: Field name
   "construct" ... shadows an attribute in parent "BaseModel"` **at class-definition time** —
   so every `okf-loremaster --help` would have printed two warnings before doing anything.
   Renamed to `predictor` in both `PredictorRow` and `NullFinding`, verified clean under
   `python -W error`. `CLAUDE.md`'s invariant updated to match.
2. **`slugify` was lowercasing author filenames**, giving `33745404_ferrari-silva.md` where the
   downstream convention is `34228066_Courtney.md`. Added `filename_token()` — same folding,
   capitalization preserved — because the filename is what an agent sees in a shelf index and
   cites back at us.

**Domain-agnosticism scan**

`tests/test_domain_agnostic.py` greps every `.py` under `src/` for six categories: disease,
drug/drug class, lab, specialty, cohort/registry, and the six shelf slugs of the hand-built
bundle this tool replaces. Whole-token matching, so `keystroke` does not trip on `stroke`.

Three guardrails on the scan itself, because the scan is the thing a future agent is most
likely to "fix" into breaking the search node:

- `test_infrastructure_terms_are_not_flagged` — 29 explicit negatives (`pubmed`, `pmc`, `mesh`,
  `tiab`, `majr`, `bioc`, `pubtator`, `icite`, `icd10`, `atc`, `loinc`, `pubmedbert`, …), plus a
  paragraph of prose using them all together. Widening the blocklist fails **here**, loudly,
  rather than in a node that quietly stops finding papers.
- `test_the_scan_would_actually_catch_something` — positive control; without it a broken matcher
  reads as a clean codebase.
- `test_the_scan_actually_has_files_to_scan` — a glob that finds nothing would make the whole
  file vacuous.

Plus two behavioral checks of the invariant itself: `Charter` ships no default `vocabularies`,
and `partition_vocabulary` has no implicit allowlist — with an empty charter list, *everything*
is unmapped.

**Known limitation**

`sentences()` is not a general-purpose tokenizer. It guards seventeen abbreviations and treats a
single letter before a period as an initial. A missed boundary joins two sentences, which counts
as one and lets slightly more text through — acceptable because every caller only truncates.

**Verified**

`mypy src/` → 26 files, no issues · `ruff check src/ tests/` → clean · `pytest` → 197 passed,
offline · `python -W error -c "import okf_loremaster.schemas"` → silent.

**Next.** Step 4 — `charter → search → dedupe → rank`, `--dry-run`, both confirmation pauses,
`--vocab`, MMR + per-shelf quota.

---

## 2026-08-03 — Step 2: NCBI clients, rate limiting, disk cache, record/replay

**Status:** complete. Gate met — *"fixtures replay fully offline."*

**Built**

- `clients/_http.py` (354 lines) — `RateLimiter` (token bucket, injectable clock/sleep),
  `DiskCache` (content-addressed, write-then-rename), `HttpClient` (retries with full jitter,
  `Retry-After` honored, `HttpStats`). Every outbound request in the package goes through it.
- `clients/cassette.py` — `CassetteTransport`, an `httpx.AsyncBaseTransport` implementing
  record/replay. It sits **below** the cache and the limiter, so a replayed test runs the same
  code path as a live call rather than a shortcut around it.
- `clients/eutils.py` (443 lines) — `esearch`, `efetch`, and the PubMed XML parser.
- `clients/bioc.py`, `clients/pubtator.py`, `clients/icite.py` — full text + license, concept
  annotations, citation metrics.
- `clients/__init__.py` — `build_clients()` and the limiter topology.
- `scripts/record_fixtures.py` — the only file permitted to touch the network.
- `tests/test_http.py` (18) · `tests/test_clients.py` (20) · `tests/test_cassette.py` (8);
  `conftest.py` gains `replay_clients` and the `no_network` autouse fixture.
- Fixture `tests/fixtures/ncbi.jsonl` — 8 interactions, 227 KiB, recorded live 2026-08-03.

**Probing the live APIs before writing parsers was the whole value of this step**

Four traps, none of which reading the documentation would have surfaced. Each is now a test.

1. **BioC answers "not in the open-access subset" with HTTP 200 and a plain-text body**:
   `[Error] : No result can be found.` `raise_for_status()` sails past it and `json.loads` then
   fails a long way from the cause. Most of any corpus is not open access, so this is an *ordinary*
   outcome — `is_unavailable()` checks the body and `fetch()` returns `None`.
2. **PubMed rewrites unknown field tags instead of rejecting them.**
   `postoperative respiratory failure[nosuchfield]` becomes `"postoperative respiratory
   failure"[All Fields]` — 14382 hits against 79 for the real query — and `errorlist` comes back
   **empty**. A generated query can be malformed and successful at the same time. Only
   `query_translation` reveals it, which is why it is a field on `ESearchResult` rather than
   discarded; the step-4 search node will check it.
3. **`PubmedBookArticle` is a sibling element of `PubmedArticle`, not a variant.** A parser that
   looks only for the latter returns fewer records than ids requested and raises nothing. Fixed by
   iterating root children in document order; `efetch` now emits a `WarningEvent` on any shortfall,
   and records carry `source_type` (`journal` / `book`).
4. **LiteLLM fetches its model-price map over HTTP at import time.** Only visible because the
   network blocker caught it. `LITELLM_LOCAL_MODEL_COST_MAP=True` in `conftest.py` for tests;
   production keeps the live fetch, since a fresher price map means fewer unpriced calls.

**Two real defects, found by tests against recorded payloads**

- **`.//ArticleIdList/ArticleId` also matches every cited reference.** On PMID 30035690 that is 19
  matches instead of 4, and the last wins — yielding `PMC1421137`, a *cited* paper's id, in place
  of the correct `PMC6340782`. That would have fetched the wrong article's full text and filed it
  under this PMID: silent, plausible, and undetectable downstream. Ids are now read from the
  scoped `PubmedData/ArticleIdList/ArticleId`. Locked by
  `test_ids_come_from_this_record_not_its_references`.
- **`authors[0].split()[0]` truncates compound surnames** — "Ferrari" for PubMed's `LastName`
  "Ferrari Silva", which then becomes the concept filename. Replaced with an `Author` dataclass
  keeping surname and initials separate; `first_author_surname` returns the surname verbatim and
  slugification is deferred to the emitter, where it belongs.

**Decisions**

- **One `HttpClient`, and therefore one token bucket, for all of NCBI.** E-utilities, BioC and
  PubTator are all `*.ncbi.nlm.nih.gov` and the limit is enforced **per IP across all of them**.
  A limiter per client is the obvious design and it is wrong: three clients at 8 rps each is 24 rps
  from NCBI's side. iCite is a different host and gets its own. Locked by
  `test_ncbi_clients_share_one_rate_limiter`.
- **Run under the published ceiling on purpose** — 8 rps with a key against a ceiling of 10, 2.5
  against 3. The limit is per IP and a shared or NAT'd address may already be carrying traffic we
  cannot see. `ncbi_rate()` clamps to the ceiling regardless of configuration.
- **`RateLimiter.acquire()` holds its lock across the sleep.** Deliberate: it serializes waiters
  into arrival order and prevents the thundering herd that releasing-then-sleeping produces.
- **Cache keys strip credentials and identity parameters** (`api_key`, `tool`, `email`), so
  rotating a key does not orphan the cache and a cached response is not tied to who fetched it.
  Recorded cassettes are redacted on write for the same reason.
- **`esummary` omitted.** `efetch` returns a superset of what it offers, and an untested client
  method with no caller is a liability rather than a convenience.
- **Tests cannot reach the network, as a guarantee rather than a convention.** The `no_network`
  autouse fixture blocks both httpx transports. A test that reaches a live API passes for the
  wrong reason, is slow, depends on someone else's uptime, and fails on a plane — and it does that
  to tests written months from now, by someone who never read this file.
- **`blocked_sync` and `blocked_async` are separate functions.** An `async def` patched over a
  synchronous method does not raise; it returns a coroutine nobody awaits, so the call quietly
  succeeds and surfaces only as a `RuntimeWarning`. That is exactly how trap 4 above was found.
- **Fixture ids are fixed literals, never search-derived**, so cassette keys stay stable across
  re-recordings. The recorded topic is deliberately unrelated to the downstream project's domain,
  which makes the suite a standing check on domain agnosticism.
- **`CitationMetrics.rcr_or_default` returns 1.0 — the field average — when RCR is `None`.**
  A paper published too recently to be scored must not rank below every scored paper in the pool.

**Lint fixes** (each a real finding, none suppressed): `RUF022` unsorted `__all__`; `RUF100` on a
`noqa: S314` for a rule not in the select list; two mypy `call-overload` / `unused-ignore` errors
in `icite._int` and `cassette.py`, fixed by rewriting the coercers and widening a dict type rather
than by silencing them.

**Not a concern, checked rather than assumed:** `xml.etree.ElementTree` does not expand internal or
external entities — it raises on an undefined one — so neither XXE nor billion-laughs applies to
the PubMed parser and no `defusedxml` dependency is warranted.

**Verified**

```
conda run -n okf-loremaster ruff check src/ tests/ scripts/   # All checks passed
conda run -n okf-loremaster mypy src/                         # no issues in 17 files, strict
conda run -n okf-loremaster pytest -q                         # 87 passed
conda run -n okf-loremaster okf-loremaster selftest           # exit 0
```

Live recording run, which is what confirmed the parsers against real payloads:

```
efetch       4 record(s)
               9500320  1998  Ileal-lymphoid-nodular hyperplasia, non- journal RETRACTED
              20301425  1993  BRCA1- and BRCA2-Associated Hereditary B book
              33745404  2022  Effects of exercise modality and intensi journal
              30035690  2019  Perioperative risk factors for postopera journal PMC6340782
bioc         PMC13424880 license='CC BY' 23 sections, 470 words
bioc/missing None  <- 200 OK with an [Error] body
```

**Open items** — both carried from step 1, unchanged: `.env` holds a live key inside a
OneDrive-synced folder (the user has declined to move it; `OKF_LOREMASTER_ENV_FILE` remains the
one-line escape hatch), and there is still no git repo (the user will `git init` when the build is
done).

**Next.** Step 3 — `schemas/`: `Charter`, `Candidate`, `ScreenVerdict`, `ConceptRecord`,
`PredictorRow` (including the new required `evidence_type`), `NullFinding`, and runtime-keyed
`vocabulary_hints` / `unmapped_vocab`. Gate: `mypy --strict src/` clean. The binding constraint is
that no vocabulary key may be a literal in `src/` — the set is derived from the charter at runtime.

---

## 2026-08-03 — Step 1: config, event bus, LLM router, cost meter

**Status:** complete. Gate met — *"a fake node emits events; live token/USD meter renders;
unknown-model path shows 'cost unavailable', not `$0.00`."*

**Built**

- `config.py` — every environment variable the package reads, declared once. Nothing else in
  `src/` touches `os.environ`. `ConfigError` messages name the offending variable.
- `events.py` — frozen slotted dataclasses (`RunStarted` `NodeStarted` `NodeFinished` `Progress`
  `LLMCall` `WarningEvent` `ErrorEvent` `RunFinished`) plus `EventBus`. This is the mechanism
  behind the *nodes never print* invariant.
- `llm/router.py` — role-bound completions, per-role concurrency, retries, `CostLedger`.
- `llm/fake.py` — scriptable completion callable; canned replies, scripted failures, call count.
- `ui/plain.py` — Rich renderer with a pinned live meter, auto-falling back to sequential lines.
- `selftest.py` + hidden `okf-loremaster selftest` — scripted two-pass run, no network.
- `cli.py` rewritten: `init` performs a real preflight and prints a status table.
- `tests/` — `test_config.py` (11) · `test_events.py` (5) · `test_router_cost.py` (13) ·
  `test_ui_plain.py` (8), replacing the step-0 placeholder.

**The `$0.00` problem, and why it needed structure rather than care**

`litellm.completion_cost()` returns `0.0` for a model it does not recognize rather than raising,
and this project's endpoint is an Azure-style gateway whose deployment names are absent from
LiteLLM's price map. So "free" and "no idea" arrive as the same float. Reporting the wrong one
is the worst kind of failure: it looks like good news.

Pricing therefore runs three stages, the third being explicit ignorance:

1. LiteLLM's price map — with `0.0` treated as *no answer*, not as zero.
2. `OKF_LOREMASTER_PRICE_<ROLE>_IN` / `_OUT`, USD per 1M tokens. Both must be set; a
   half-configured pair returns `(None, None)` rather than silently undercounting.
3. `usd = None`. Tokens still counted, call tallied as unpriced.

Rendering is a single function, `format_cost(usd, *, calls, unpriced)`, shared by the ledger and
every renderer so no display path can drift. `$0.00` is reachable only when `calls == 0`;
all-unpriced renders `cost unavailable`; a mixed run renders `$1.2345 + 3 unpriced` rather than
quietly reporting a partial total as if it were complete.
`test_zero_dollars_only_ever_means_zero_calls` brute-forces the combinations to keep it that way.

**Decisions**

- **Dropped `tenacity`.** The retry loop must emit a `WarningEvent` per attempt — a retry that is
  invisible is indistinguishable from a slow network. Wiring that through tenacity's callbacks was
  more code than the loop it would have replaced. Removed from `pyproject.toml` rather than left
  as an unused dependency. Backoff is exponential with full jitter
  (`min(30, 2**attempt) * random()`), so parallel calls hitting one rate limit do not retry in
  lockstep.
- **Transient and permanent failures are classified, not lumped.** Retrying an auth failure or a
  malformed request just burns rate limit and delays the real error, so those raise on the first
  attempt. Exception classes resolve lazily from `litellm` by name and are cached, keeping the
  import out of `--help`.
- **`LLMCall` carries both per-call and cumulative figures.** Renderers stay stateless with
  respect to arithmetic; a late subscriber or a second renderer cannot disagree with the first
  about the running total.
- **`PlainRenderer` subscribes in `__init__`, not in `run()`.** Events emitted between wiring and
  starting the consumer task would otherwise be dropped.
- **`_handle` has no fallback branch.** The `match` covers the `Event` union exhaustively; with no
  `case _`, mypy reports a missing return the moment an event type is added without a case. A
  fallback would silently swallow it instead. The trailing `return []` was removed once mypy
  flagged it unreachable — that diagnostic is the feature working.
- **Renderer writes to stderr**, leaving stdout clean for piped data.
- **Live display is opt-out by signal**, checked strongest-first: `NO_COLOR`, then `CI`, then a
  dumb/empty `TERM`, then `isatty()`.

**Snag — a real defect, found by tests**

Three tests failed on `Settings(hf_home=...)` being silently ignored. Cause: a field carrying
`validation_alias` cannot be populated by its field name unless told otherwise, so the constructor
kwarg was discarded and the ambient environment won instead. That affected `api_key`, `api_base`,
and `hf_home` — meaning the selftest's own overrides were being dropped and a real key could have
leaked into a test run. Fixed with `validate_by_name=True, validate_by_alias=True`. The older
`populate_by_name=True` also works; both were checked under `-W error` and the current spelling is
the non-deprecated one.

**Lint fixes** (each a real finding, none suppressed): two `RUF100` unused `noqa: BLE001` —
`BLE` is not in the select list, so the suppressions were noise; one `SIM102` nested `if` in
`_price_from_litellm`; two `E501`.

**Verified**

```
conda run -n okf-loremaster ruff check src/ tests/   # All checks passed
conda run -n okf-loremaster mypy src/                # no issues in 10 files, strict
conda run -n okf-loremaster pytest -q                # 41 passed in 9.64s
conda run -n okf-loremaster okf-loremaster selftest  # exit 0
```

The selftest runs the same scripted graph twice against `gateway/custom-deployment-name`, a model
LiteLLM cannot price by design:

| pass | prices set | rendered cost |
|---|---|---|
| `selftest-unpriced` | none | `cost unavailable` + a line naming the variables to set |
| `selftest-priced` | `PRICE_*_IN/_OUT` | `$0.0064` |

Both passes emit two retry warnings from scripted transient failures
(`! screen: fast call failed (ConnectionError), retry 1/3 in 0.8s`), confirming retries surface
rather than hide. Verified under a pty (live meter path) and piped (fallback path); `NO_COLOR=1`
forces the fallback. Final assertion line:

```
PASS unpriced -> cost unavailable   priced -> $0.0064
```

**Open items**

- **`.env` holds a live API key inside a OneDrive-synced folder.** `config.py` already searches
  `~/.config/okf-loremaster/.env` before the project-local one, and `OKF_LOREMASTER_ENV_FILE`
  pins a single path. Moving it is a one-line change with no code impact — recommended, not yet
  done.
- **No version control.** There is no git repo here, so an editor buffer holding a stale copy of a
  file can silently revert work on save. `git init` would make that recoverable.

**Next.** Step 2 — `clients/eutils.py` and `clients/bioc.py` with a token-bucket limiter (8 rps
with an API key, 2.5 without) and a disk cache, plus `pubtator` and `icite`, and record/replay
fixtures so the suite runs fully offline. Rules that bind here: E-utilities/BioC/PubTator/iCite
only, never a scraped page; never `oa.fcgi` or the retired PMC FTP layout.

---

## 2026-08-03 — Step 0: scaffold + environment

**Status:** complete. 0 of 10 build steps remaining before step 1.

**Built**

- `pyproject.toml` — hatchling, src layout, extras `[vectors] [tui] [dev] [all]`, ruff + mypy
  strict + pytest config.
- `src/okf_loremaster/__init__.py`, `src/okf_loremaster/cli.py` — full Typer command surface
  (`init charter build index validate export inspect`) with every planned flag declared, so
  `--help` shows the real shape of the tool from day one.
- `CLAUDE.md` — operating manual, invariants, behavior. 6.4 KB against a ~10 KB cap.
- `README.md` — install, configure, use, conduct. Completed properly in step 10.
- `.env.example` — every variable, annotated.
- `.gitignore`, `cspell.json`, `tests/test_scaffold.py` (placeholder; real tests land in step 1).
- Conda env `okf-loremaster` (Python 3.11.14), editable install with all extras.

**Naming.** Project renamed from "Loremaster-OKF" to **OKF Loremaster** before any code was
written. One name in three mechanical spellings, because hyphens are illegal in Python
identifiers and in shell variable names:

| | |
|---|---|
| Display name | OKF Loremaster |
| PyPI dist | `okf-loremaster` |
| Import package | `okf_loremaster` |
| CLI | `okf-loremaster`, alias `loremaster` |
| Env prefix | `OKF_LOREMASTER_` |
| Conda env | `okf-loremaster` |

The `loremaster` alias is a second console script pointing at the same app. `okf-loremaster` is
canonical in all documentation; the alias exists only to save typing.

**Decisions**

- **Dedicated conda env, not `fe_demo2`.** `chromadb` + `litellm` together pull `onnxruntime`,
  `tiktoken`, `openai`, `posthog`, and the resolver has a history of moving pydantic bounds. A
  bump under `fe_demo2`'s live demo is not reliably undone by a `pip freeze` restore. One
  `conda create` makes that structurally impossible. Confirmed by the resolved set below, which
  installed torch 2.13 and pydantic 2.13.4 — exactly the kind of change that would have been
  unwelcome in a shared env.
- **`[vectors]` is an extra, not a base dependency.** It pulls torch. The OKF bundle is the
  product; the vector index is derived and optional, and `--index/--no-index` already gates it.
- **Flow-style frontmatter.** OKF v0.2 nests `generated` / `verified` / `sources` and derives
  trust tiers from `verified`; flattening forfeits conformance, and multi-line YAML breaks the
  naive line-parser downstream. One key per line with YAML flow style for nested values satisfies
  both. Rejected: flat `generated_by` / `generated_at`.
- **Dropped `status` and `retracted` from frontmatter**; moved `stale_after` to the root
  `index.md`; `verified` is written only by `--review`. Each was audited against whether anything
  actually consumes it. Absent `verified` means spec tier `unverified`, which is the honest tier
  for machine extraction — a self-attestation on every file discriminates nothing.
- **Rich by default, Textual behind `--tui`.** The GUI was cut from scope.
- **Tool caches redirected to `~/.cache/okf-loremaster/`.** This project lives in a OneDrive
  folder. A single `mypy` run over two files produced 4.5 MB of SQLite in `.mypy_cache/`, rewritten
  on every invocation — gitignored, but OneDrive syncs it regardless. `ruff`, `mypy`, and `pytest`
  all expand `~` in their cache path settings (verified empirically for mypy, which does not
  document it), so all three are configured in `pyproject.toml` rather than left to env vars that
  a future run would forget to set.

**Known limitation.** Dropping per-concept `status` means a paper retracted *after* a bundle is
built cannot be marked `deprecated` without a full rebuild. Accepted: retracted papers are
dropped at `dedupe`, so this only affects post-build retractions.

**Open items**

- **License not chosen.** `pyproject.toml` declares no license and `README.md` says so. Treat the
  package as unlicensed and internal until one is set.
- **Folder still named `Loremaster-OKF`.** Renaming it later is safe, but the editable install
  records this directory's absolute path — after a rename, run
  `conda run -n okf-loremaster pip install -e .` from the new location or imports stop resolving.
  Nothing else in the code depends on the folder name: all path defaults are relative to the
  working directory or rooted at `$HOME`.

**Verified**

```
conda run -n okf-loremaster okf-loremaster --help      # full command table renders
conda run -n okf-loremaster okf-loremaster --version   # OKF Loremaster 0.1.0.dev0
conda run -n okf-loremaster loremaster --version       # alias resolves to the same app
conda run -n okf-loremaster okf-loremaster validate .  # exits 2, names build step 7
conda run -n okf-loremaster ruff check src/            # All checks passed
conda run -n okf-loremaster mypy src/                  # no issues, strict
conda run -n okf-loremaster pytest -q                  # 1 passed
```

No `.mypy_cache`, `.ruff_cache`, or `.pytest_cache` is left in the project directory after all
three run.

**Resolved versions** (2026-08-03, macOS arm64, Python 3.11.14) — recorded because several are
majors that later steps are written against: `langgraph` 1.2.10 · `langgraph-checkpoint-sqlite`
3.1.1 · `litellm` 1.95.0 · `chromadb` 1.5.9 · `sentence-transformers` 5.6.1 · `transformers`
5.14.1 · `torch` 2.13.0 · `pydantic` 2.13.4 · `pydantic-settings` 2.14.2 · `typer` 0.27.1 ·
`textual` 8.2.8 · `httpx` 0.28.1 · `rich` 15.0.0 · `mypy` 2.3.0 · `ruff` 0.16.1 · `pytest` 9.1.1.

**Snags**

- First editable install failed with `OSError: Readme file does not exist: README.md`. The install
  was backgrounded before `README.md` was written, and hatchling reads `readme` during metadata
  generation. Re-ran after writing it. No code change needed.

**Next.** Step 1 — `config.py` (pydantic-settings, failures name the variable), `llm/router.py`
(LiteLLM, FAST/MID/DEEP roles, retries, token + USD accounting, per-role concurrency),
`events.py`, `ui/plain.py`. The configured endpoint is an Azure-style gateway, which is precisely
the case where LiteLLM's `completion_cost()` returns `0.0` for an unrecognized model — so the
`OKF_LOREMASTER_PRICE_<ROLE>_IN/_OUT` fallback and the "cost unavailable" marker need to work on
the first run, not as a later patch.
