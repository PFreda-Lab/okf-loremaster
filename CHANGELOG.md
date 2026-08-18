# Changelog

Notable changes to OKF Loremaster. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version 0.x means the OKF bundle layout and the CLI may still change between minor releases.
The bundle contract downstream reads is the part to watch — it is called out here when it moves.

## [Unreleased]

### Added

- **Per-tier reasoning effort, set in `.env`.** `OKF_LOREMASTER_EFFORT_FAST`, `_BALANCED` and
  `_REASONING` take one vocabulary across providers — `minimal low medium high xhigh max none` —
  which LiteLLM turns into a thinking budget for Anthropic and the native parameter for OpenAI.
  Left commented out, nothing is sent and the provider's own default applies; `none` is a different
  request, and a real one, since it asks a model that would otherwise reason not to. The budget is
  *added* to each node's reply allowance rather than taken out of it: screening asks for 256 tokens
  and every thinking budget is larger than that, Anthropic requires `max_tokens` above
  `budget_tokens`, and LiteLLM applies its own cap only on the legacy `thinking` path and never on
  this one. `--dry-run` counts the full budget on every call, so cost it there first — effort on
  BALANCED is the expensive one, because extraction lives there at a call per paper.
- **`--no-abstract` leaves the `# Abstract` section out of every document.** Not to be confused
  with `--basis`: that decides what a paper is *read from*, this only decides what the finished
  document *carries*. The search, the screening, the curation and the extraction are identical
  either way, and all that changes is that the publisher's own words are not copied into the file
  at the end — about a fifth of a document's bytes, which is worth having when a downstream agent
  pays per token to read the corpus. It does not make a bundle redistributable, because every
  predictor row still quotes the paper verbatim. Since a missing `# Abstract` already means "PubMed
  served none" on roughly one paper in ten, a bundle built this way says so in `log.md`, in its root
  `index.md`, and as `abstracts: false` in `resource_descriptor.yaml` — otherwise a corpus without
  them reads as an unlucky one rather than a deliberate one.

### Changed

- **A price you configure is now used in preference to LiteLLM's table, not after it.**
  `OKF_LOREMASTER_PRICE_<ROLE>_IN`/`_OUT` were consulted only when LiteLLM had no opinion, which
  made them live for gateway deployment names it cannot recognize and dead for every published
  model whose price has since moved. LiteLLM prices from a static file shipped inside its own
  wheel: it was still answering the introductory $2/$10 for `claude-sonnet-5` well after that
  became $3/$15, so every cost this tool printed for that model was two thirds of the truth. An
  unpriceable call still reports tokens and "cost unavailable" rather than `$0.00`.
- **`--dry-run` projects far closer to what a run actually costs** — about 80% of it, from about
  27%. Two independent errors: the `response_format` schema was never counted as input, though it
  is billed once per call and came to 12% of a 199-paper run, and the reply allowances were
  reasoned about rather than measured. Extraction's was 700 tokens against a measured mean of
  5,034.
- **Six calls per model in flight instead of three**, which roughly halves the wall clock of a
  build — about 30 minutes of extraction for 200 papers, against about 50. Rate limits are counted
  per model per minute, so the tiers were never sharing one budget as the previous default assumed.
  `OKF_LOREMASTER_CONCURRENCY_BALANCED` still lowers it if a deployment's quota is tighter.
- **`--target-papers` defaults to 150 rather than 200**, the middle of the 120–250 browsability
  band rather than the top of it. Extraction alone sets both the price and the wall clock of a run,
  and the default is what somebody gets before they have any idea what either will be.
- **Reply ceilings raised on extraction, query planning and curation.** A reply cut off by its
  ceiling is billed in full, discarded and re-asked, so the room costs nothing unspent and the lack
  of it costs a whole generation. Extraction tripped its old ceiling on 22% of papers in one run.

### Fixed

- **Reasoning effort and temperature could not be sent together, and were.** Every call this tool
  makes asks at temperature 0, and no provider allows that alongside reasoning: Anthropic refuses
  any temperature but 1 once thinking is on, and OpenAI's reasoning models refuse the parameter
  outright. LiteLLM's `drop_params` is no help — it drops what a model does not *support*, and
  temperature is fully supported; only the combination is rejected. Setting
  `OKF_LOREMASTER_EFFORT_FAST` therefore turned screening into a wall of 400s, 30 calls out of 30,
  and the run went on to emit a bundle that validated cleanly on the papers nothing had screened.
  A tier with effort set now sends no temperature at all — dropped rather than pinned to 1.0, since
  1.0 satisfies Anthropic and fails OpenAI. A tier left unset is untouched and still asks at 0,
  which is what an open-weight deployment gets: those take a temperature and not a
  `reasoning_effort`. The trade belongs to whoever sets the variable and `.env.example` now says
  so: that tier stops sampling deterministically, by the provider's rule rather than ours.
- **A two-letter language code took an entire run to zero hits.** The charter's `languages` went
  into PubMed's `[la]` field verbatim, and that field takes three-letter ISO 639-2/B codes — `eng`,
  not `en`. It is the last place a mistake like this shows up, because `[la]` is a real field tag:
  PubMed does not rewrite it the way it rewrites an unknown one, it reports an empty `errorlist`,
  and it answers `Count 0`. Since the filter is appended to every query in a plan, twelve
  well-formed queries off a well-drafted charter returned nothing at all, 84 seconds and one
  reasoning call in. `en`, `english`, `deu` and the rest now normalize to the code PubMed answers
  to, duplicate spellings collapse to one clause, a language PubMed does not index is refused by
  name while the charter is still being parsed, and the field description tells the drafting model
  the format — the charter prompt never asks for this field, but a model handed the schema fills it
  in anyway, and what it wrote unprompted was `en`. The error a run with no hits raises now points
  at the filters every query shares before it blames any single clause.
- **A fifth of the vector index was never indexed.** A sentence encoder drops whatever runs past
  its window and returns a vector anyway, with no error and nothing in the response to say it
  happened, so a chunk longer than the window was embedded from its opening only. The default
  model's window is 350 tokens rather than the 512 usually assumed of a BERT-family checkpoint,
  which is what made this so much worse than it looked: on a 199-paper bundle 601 of 1,204 chunks
  were truncated and **103,089 of 491,705 tokens — 21% — never reached the store**; on another it
  was 30%. The old warning reported about 16% because it counted characters against a budget
  derived from the window it assumed. Chunks are now measured with the embedder's own tokenizer and
  split at the coarsest boundary that fits — section, then paragraph, then line, then word — with
  every part repeating the paper's title so it still reads on its own when it comes back alone. A
  character budget cannot do this job: real chunk text ran from 2.1 to 6.8 characters per token, so
  no single figure is both safe and efficient. Re-measured on three bundles: zero truncated, zero
  words lost. A chunk that already fit keeps the id and the text it always had — 1,097 of them are
  byte-identical — and a split one becomes `<handle>#<row>.1`, `.2` and so on, carrying
  `chunk_part` and `chunk_parts` beside the metadata it already had. Rebuild an index to pick this
  up; the OKF bundle itself is unchanged.
- **A run under `--tui` could finish its bundle and then hang forever.** The end-of-run question
  about what to keep was asked with a `rich` prompt while Textual held the terminal in raw mode:
  the question is painted underneath a full-screen app and blocks on a read no keypress can reach,
  so the run sits there complete, with `q` and `c` both dead because the loop that handles them is
  the loop that is blocked. Runs that do not own the terminal are unaffected. Both resources are
  now kept without asking, which is what an unattended run already did.
- **The vector index gave no sign of life while its embedding model loaded.** The model loads
  lazily and a first run downloads several hundred megabytes, which fell between two progress
  messages on the last node of a long build — and being a blocking load, it stalled the renderer
  as well. The load is now announced before it starts and runs off the event loop.
- **Schemas with too many optional fields silently stopped constraining replies.** A provider
  compiling a schema into a decoding grammar caps how many optional parameters it will branch on;
  past that it rejects the schema, the rejection reads as a refusal, and every later call goes out
  unconstrained. Every property is now marked required, which keeps the meaning — nullable fields
  still accept null — and takes the count to zero.

## [0.2.0] — 2026-08-17

**The bundle contract moved.** Every per-paper document gains a twelfth finding-table column and a
verbatim abstract, and papers that report one gain a section for it. The change is additive — no
column was renamed or removed, no file moved, and the folder layout is identical — so a reader that
keys on the header row is unaffected. A reader that parses the finding table by column *position*
is not, which is what makes this a minor bump rather than a patch.

### Added

- **`--basis any|abstract|full-text` decides what each paper is read from.** The default `any` is
  the previous behavior: open-access full text where PMC has it, the abstract otherwise. `abstract`
  reads every paper the same way, so nothing in the corpus is deeper than anything else and nothing
  is dropped for being paywalled. `full-text` keeps only papers whose full text is open access — a
  smaller, slower, costlier corpus that answers more per document. Enforced in `rank`, upstream of
  the re-query edge, so a policy holds across search rounds. `resource_descriptor.yaml` records the
  policy and `index.md` says what the corpus was read from, both silent under `any` rather than
  claiming a policy nobody set.
- **An `# Abstract` section on every document**, carrying the source abstract verbatim. For a
  full-text document it is the only place the authors' own framing survives; for an abstract-only
  one it is what the structured sections above it were derived from, so a row that reads oddly can
  be checked without leaving the file. It sits under the document's existing `license` and
  `export_safe` like every other quoted span.
- **`Interacts with`, a twelfth finding-table column, and an `# Interactions` section.** The column
  is on every table and reads `—` when a paper reports nothing, which is the common case; the
  section appears only where a paper reports something. It records how two predictors relate rather
  than how one relates to the outcome — correlated, mutually exclusive, modifying, confounding,
  mediating or derived from — each on its own line, joined to the finding table by row number, with
  a magnitude and the evidence for it. It is a note for whoever selects features downstream, since
  two collinear predictors are two things a model should probably not carry separately, and it says
  nothing about how good the study is. Papers volunteer this far more often in full text than in an
  abstract.

### Fixed

- **`init` reported `ready` on a machine whose first fetch would fail.** Its readiness check was
  built only from the model-side requirements, and the NCBI contact address is enforced where the
  HTTP clients are wired rather than by any model call — so with an API key set and
  `OKF_LOREMASTER_NCBI_EMAIL` empty, `init` printed `ready` and exited 0 while a build raised
  `ConfigError` on its first request. It now asks both halves, and names the variable.
- **A provider that refuses schema-constrained output in different words no longer kills a run at
  the first node.** The router already falls back to prose JSON when a provider rejects structured
  outputs, but it recognized only one wording. An Azure AI Foundry Anthropic deployment answers
  `Grammar compilation timed out.`, which is the same refusal said differently — a provider compiles
  the schema into a decoding grammar before the model sees anything, and one it cannot compile in
  time it will not honor. Because "timed out" reads as transient, this took the retry path and
  failed identically every attempt, ending the run at `charter`. It is now treated as a refusal and
  learned once per model rather than rediscovered per call.

### Notes

- The abstract crosses into a bundle whatever the paper's license; `export_safe` is what says
  whether the document may be redistributed. The README previously said only quoted spans crossed,
  which stopped being true with this release.

## [0.1.3] — 2026-08-15

Configuration and documentation. Existing `.env` files keep working — both Anthropic spellings
remain accepted aliases.

0.1.2 was tagged and never published: its release run stopped at the smoke test, which asserted on
a variable name this release moved. Nothing reached PyPI under that number.

### Fixed

- **Configure counted the wrong two variables.** It said a first run needs an API key and an
  email, meaning NCBI's — and skipped model routing, which is the part that costs money and the
  part most likely to be wrong. It is now three things: models, a key for them, and the NCBI
  contact address.
- **Every instruction assumed Anthropic.** Configure carries a provider table — Anthropic, OpenAI,
  Azure OpenAI, Azure AI Foundry, Bedrock, Ollama, and OpenAI-compatible servers such as vLLM and
  LM Studio — with the model string and base URL each one wants.
- **Azure had no instructions at all.** Two facts that are not guessable from the outside: LiteLLM
  defaults to api-version `2025-02-01-preview`, and pinning another needs `AZURE_API_VERSION`
  exported in the shell, because nothing here copies `.env` into `os.environ`. Also that LiteLLM
  cannot price a deployment name, so an Azure run reports "cost unavailable" until
  `OKF_LOREMASTER_PRICE_*` is set.
- A local model needs a non-empty `OKF_LOREMASTER_API_KEY` even though nothing checks the string.
  Undocumented, and it stops a run before it starts.
- **`init` named `ANTHROPIC_API_KEY` as the variable to go and set**, while the template it had
  just written asks for `OKF_LOREMASTER_API_KEY`. Somebody configuring Azure or a local server was
  pointed at the one spelling that is wrong for them. It now names ours.
- **`HF_HOME` shipped uncommented as `/Users/<you>/.cache/huggingface`.** A literal placeholder,
  not an example: an untouched `.env` sent the embedding model download to a directory named after
  the placeholder. Commented out, so Hugging Face's own default applies until it is set
  deliberately.

### Changed

- `.env.example` leads with `OKF_LOREMASTER_API_KEY` and `OKF_LOREMASTER_API_BASE` rather than the
  Anthropic spellings, with a provider menu above the model block. `ANTHROPIC_API_KEY` and
  `ANTHROPIC_BASE_URL` remain accepted aliases, so no existing configuration breaks.
- The tier table gives how often each tier is called instead of example model names.

## [0.1.1] — 2026-08-14

Documentation only. No code changed; the bundle layout and the CLI are identical to 0.1.0.

### Fixed

- **The README named the wrong variable for the API key.** It said the key is read "under the
  provider's own name, not ours", which is not what the code does — the router passes `api_key`
  to LiteLLM explicitly and refuses to start when it is empty. Anyone on a non-Anthropic provider
  who set `OPENAI_API_KEY` and nothing else was told a required variable was unset with no
  indication why. The key goes in `ANTHROPIC_API_KEY` or `OKF_LOREMASTER_API_KEY` whatever the
  provider.

### Changed

- Configure is a four-step walkthrough — pick a directory, `init`, fill in two lines, `init`
  again — rather than a block of prose that assumed the reader already knew the shape of things.
  It says where `.env` and `bundles/` land, what each required value is for and where to get it,
  and shows the report `init` prints when a machine is ready.
- Install says what a `pip install` does and does not need, and points at Configure instead of
  ending on a list of extras.

## [0.1.0] — 2026-08-14

First public release.

### Added

- Build a task-scoped biomedical literature bundle from PubMed/PMC in Open Knowledge Format
  v0.2, from a plain-language research question: `okf-loremaster build "..."`.
- A thirteen-node graph — `charter → search → dedupe → rank → screen → curate → fulltext →
  extract → reconcile → review → emit_okf → validate → index_vectors`. Five agents make the
  judgment calls; HTTP, dedup, ranking, MMR, license logic, validation and indexing are code.
- Per-paper markdown carrying predictor rows with effect sizes, operationalization, timing,
  evidence type, null findings and vocabulary hints, plus a `predictors.md` index that points
  into them and a `search.md` that reproduces the search.
- Deterministic numeric verification: an effect the source text does not contain becomes
  `effect=None` with a downgraded confidence and a logged warning, and the run continues.
- A computed `strength` per row from study design, adjustment and sample-size scale — never
  asked of the model.
- Optional Chroma vector index over the finished bundle (`[vectors]`), built by walking what
  was emitted rather than by a second extraction pass.
- Resume from a checkpoint (`--resume`), a `runs` listing, a Textual TUI (`[tui]`), JSONL
  event output, and cost reporting that says "cost unavailable" rather than `$0.00` for a
  model LiteLLM cannot price.
- `okf-loremaster init` writes an annotated `.env` and reports what is still unset.

### Notes

- Python 3.11 and 3.12.
- NCBI access is E-utilities, BioC, PubTator and iCite only; no page is ever scraped.
- Apache-2.0 covers this code, not the bundles it builds. Each emitted document records the
  `license` its publisher reported, and `export_safe` says whether it may leave.

[Unreleased]: https://github.com/PFreda-Lab/okf-loremaster/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/PFreda-Lab/okf-loremaster/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/PFreda-Lab/okf-loremaster/compare/v0.1.1...v0.1.3
[0.1.1]: https://github.com/PFreda-Lab/okf-loremaster/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PFreda-Lab/okf-loremaster/releases/tag/v0.1.0
