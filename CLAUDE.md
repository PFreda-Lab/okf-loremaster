# CLAUDE.md — OKF Loremaster

Operating manual for agents working in this repo. One line plus a pointer, never a paragraph.
Cap ~10 KB. **Adding something means shortening something.**

## What this is

A standalone, pip-installable tool that builds a **task-scoped biomedical literature knowledge
bundle** from PubMed/PMC in Open Knowledge Format (OKF v0.2), plus an optional derived Chroma
vector index.

It sits **upstream of everything**. AFCE (`../AFCE`, currently `../FE_Demo2`) consumes what we
produce: its Evidence Reviewer browses the OKF shelf via `okf_*` tools, a second one queries the
vector store. **Coupling is files on disk only** — no shared import, interpreter, or state in
either direction.

We replace hand-built abstract dumps with structured evidence: predictor rows with effect sizes,
operationalization, timing, evidence type, null findings, vocabulary hints — curated to a
browsable 120–250 papers.

## Docs

| File | For |
|---|---|
| `README.md` | How to install and run it |
| `Build_Progress.md` | Datestamped log; measurements, rejected alternatives, forensics |
| `.env.example` | Every configurable variable, annotated |
| `../AFCE/specs/6.4-resources.md` §6.4.1 | The downstream contract our output must satisfy |
| `~/.claude/plans/claude-code-build-splendid-tarjan.md` | The build plan |

## Environment

Dedicated conda env. **Never install into `fe_demo2`** — a resolver bump there breaks a live demo.

```bash
conda run -n okf-loremaster okf-loremaster --help
conda run -n okf-loremaster pytest
conda run -n okf-loremaster mypy src/
conda run -n okf-loremaster ruff check src/ tests/
```

`HF_HOME=~/.cache/huggingface` — shared with other envs so the embedding model downloads once.
**Never inside OneDrive**: the hub cache links `snapshots/` into `blobs/` with symlinks, which the
sync client mangles. Not the deprecated `TRANSFORMERS_CACHE`.

The editable install records an absolute source path. **If this folder is renamed or moved, re-run
`conda run -n okf-loremaster pip install -e .` from the new path** or imports stop resolving.

## Stack

Typer + Rich · pydantic / pydantic-settings · httpx · LangGraph 1.x (`langgraph-checkpoint-sqlite`
is a **separate package**) · LiteLLM · Chroma + sentence-transformers (extra `[vectors]`) · Textual
(extra `[tui]`).

Graph: `charter → search → dedupe → rank → screen → curate → fulltext → extract → reconcile →
review → emit_okf → validate → index_vectors`.

Three LLM roles bound in config: **FAST** screening · **MID** query planning, curation · **DEEP**
charter, extraction.

## Invariants

Broken by accident, so stated explicitly.

- **Biomedical scope is a given; clinical specifics are not.** This package is biomedical by
  construction. PubMed, PMC, E-utilities, BioC, MeSH and `[tiab]` query syntax, PubTator, the
  standard clinical coding systems, and a biomedical embedding model are all in scope and belong
  in `src/`. What must never appear as a constant in `src/` is anything *below* that level: a
  specific disease, condition, specialty, drug, drug class, lab, cohort, registry, shelf name, or
  a fixed list of vocabulary keys. Those are derived at runtime from the charter. Test: could this
  constant be wrong for a project on a different condition? If yes, it belongs in the charter.
- **Agents only for judgment** — charter, query planning, screening, curation, extraction. HTTP,
  dedup, ranking, MMR, license logic, file writing, validation, embedding, indexing are code.
- **Nodes never print.** They emit typed events; renderers subscribe.
- **Frontmatter is one key per line, YAML flow style for nested values.** OKF v0.2 nests
  `generated` / `verified` / `sources`, and derives trust tiers from `verified`; flattening
  forfeits conformance, multi-line breaks naive line-parsers downstream. Flow style satisfies both.
- **The frontmatter key is `domain`; the human word is "shelf".** `domain` must equal the folder
  name. Never let "shelf" leak into frontmatter.
- **Embeddings resolve through config**, default local and pinned by revision — never a
  module-level literal. Downstream rejects remote embedders on attach.
- **The vector index is built by walking the finished bundle**, never a second extraction pass.
- **`null_findings` is never omitted** — `predictor: "none reported"` when there is none. A
  validator inserts the sentinel, so omission is impossible rather than merely checked. The field
  is `predictor`, not `construct`: pydantic warns at import on a field that shadows
  `BaseModel.construct`, and every CLI invocation would print it.
- **Numeric verification is deterministic post-processing.** A number the source text does not
  contain becomes `effect=None` with a downgraded confidence and a logged warning; the run
  continues.
- **Never scrape PubMed or PMC web pages.** E-utilities, BioC, PubTator, iCite only.
- **Never use `oa.fcgi` or the retired PMC FTP layout.** It still returns HTTP 200 (verified
  2026-08-03), so this is a design choice, not a workaround: BioC returns full-text JSON per
  article with `infons.license`, while `oa.fcgi` only ever returned package locations for
  FTP/cloud download. Do not "fix" this back.
- **HTTP 200 is not success.** BioC answers "not in the open-access subset" with 200 and a
  plain-text `[Error]` body; `raise_for_status()` sails past it. Availability is checked on the
  body, and an unavailable article returns `None` — most of any corpus is not open access.
- **One rate limiter for all of NCBI.** E-utilities, BioC and PubTator share an IP-enforced
  limit, so they share one `HttpClient`. A limiter per client is 3× the configured rate.
- **PubMed rewrites unknown field tags instead of rejecting them.** `x[nosuchfield]` becomes
  `"x"[All Fields]`, returns far more hits, and reports an *empty* `errorlist`. A generated
  query can be malformed and successful at once; only `query_translation` reveals it.
- **Identifiers are read from `PubmedData/ArticleIdList`, never `.//ArticleIdList`.** The
  unscoped path also matches every cited reference, silently yielding another paper's PMC id.
- **Tests never reach the network.** `conftest.py` blocks the httpx transports outright;
  fixtures are recorded by `scripts/record_fixtures.py`, the one file allowed to call out.
- **Never report `$0.00` for an unpriced model** — say "cost unavailable". LiteLLM's
  `completion_cost()` returns `0.0` for unknown models rather than raising.
- **Never write to `../FE_Demo` or `../FE_Demo2`.** Reference only. `../AFCE` docs are edited, but
  only in build step 9.
- **Never share, sync, or print `.env` secrets.**

## Behavior

- **Stop after each numbered build step** and show what was built before continuing.
- Update `Build_Progress.md` on every material change, with a datestamp.
- Do not over-engineer. Prioritize provenance correctness, determinism, low token cost, clean
  observability, and a bundle that is pleasant for a downstream LLM agent to navigate.
- 120–250 papers is a **browsability ceiling, not a recall target**.
- Config failures are loud and name the variable.
- **American English only — no exceptions.** Prose, identifiers, comments, docstrings, CLI help,
  commit messages, and emitted OKF content alike. `operationalized` not `operationalised`,
  `prioritize` not `prioritise`, `summarize` not `summarise`, `behavior` not `behaviour`,
  `analyze` not `analyse`, `catalog` not `catalogue`. The one thing you never "correct": text
  quoted verbatim from a source paper, which is reproduced exactly as published.
- Keep the conda env outside this directory so it does not bloat cloud sync.
