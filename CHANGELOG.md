# Changelog

Notable changes to OKF Loremaster. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version 0.x means the OKF bundle layout and the CLI may still change between minor releases.
The bundle contract downstream reads is the part to watch — it is called out here when it moves.

## [Unreleased]

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

[Unreleased]: https://github.com/PFreda-Lab/okf-loremaster/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/PFreda-Lab/okf-loremaster/compare/v0.1.1...v0.1.3
[0.1.1]: https://github.com/PFreda-Lab/okf-loremaster/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PFreda-Lab/okf-loremaster/releases/tag/v0.1.0
