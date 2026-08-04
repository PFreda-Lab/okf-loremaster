# OKF Loremaster

**Turn a research question into a browsable, cited, machine-readable evidence library.**

OKF Loremaster searches PubMed and PMC, screens and curates the results down to a corpus a person
could actually read, extracts structured evidence from full text where the license allows it, and
writes a directory of markdown documents in
[Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing)
(OKF v0.2) — plus an optional vector index derived from the same files.

```bash
okf-loremaster build "predictors of 30-day readmission after heart failure hospitalization" --index
```

- [Who this is for](#who-this-is-for)
- [What you get](#what-you-get)
- [The agentic system](#the-agentic-system)
- [The interface](#the-interface)
- [Install](#install)
- [Configure](#configure)
- [Commands](#commands)
- [Output contracts](#output-contracts)
- [Worked example](#worked-example)
- [Conduct and provenance](#conduct-and-provenance)

---

## Who this is for

This tool sits **upstream of feature-construction agentic systems in the clinical and EHR domain** —
systems where a panel of LLM agents proposes, critiques and assembles candidate features from
patient data. Those agents are only as good as the evidence they can reach, and the usual way to
feed them is a folder of abstracts scraped into a prompt. That fails in a specific, quiet way:

- an abstract says *"was associated with"* and stops, so the agent cannot rank a candidate by effect
  size, or tell an odds ratio of 1.05 from one of 4.2;
- it rarely says **how** a predictor was measured, or **when** relative to the outcome — which is
  exactly what a feature-engineering agent has to decide, and exactly where leakage comes from;
- **null findings are invisible**, so the agent proposes what the literature already ruled out;
- there is no vocabulary bridge from the paper's words to ICD/LOINC/RxNorm/SNOMED codes in a warehouse;
- nothing records what was read from full text and what was read from an abstract, so every claim
  carries the same apparent weight.

OKF Loremaster produces the structured version of all five: predictor rows with effect sizes,
operationalization, timing and direction; a separate table of nulls; vocabulary hints; and
provenance down to `text_basis: "full_text"` per document. Drop the result into a downstream
system's `resources/` directory and its evidence agents browse a shelf instead of grepping a blob.

**It is not a systematic-review tool and does not pretend to be one.** The target is 120–250
papers — a *browsability ceiling, not a recall target*. If an agent cannot read the shelf, the shelf
is too big.

Nothing about the format is proprietary to us: OKF is a public format, the bundle is plain markdown
and YAML, and any consumer that reads OKF reads this. The coupling to a downstream system is
**files on disk only** — no shared import, no shared interpreter, no shared state.

---

## What you get

```
<bundle>/
  index.md                  # the charter, a shelf table, the run manifest, cost totals
  log.md                    # what happened, in order, with counts
  _catalog.jsonl            # one JSON row per document — for code, not for reading
  resource_descriptor.yaml  # what a downstream tool reads on attach
  cardiac-function/
    index.md                # browse table for this shelf
    31234567_Okonkwo.md     # one document per paper
    33745404_Ferrari-Silva.md
  medications/
    index.md
    ...
<bundle>.chroma/            # optional, derived, and beside the bundle — never inside it
  resource_descriptor.yaml  # embedding model, resolved revision, dimensions, distance metric
```

Every document is YAML frontmatter plus five ordered sections. The frontmatter is the citation and
the provenance; the body is the evidence:

````markdown
---
type: "Literature Evidence"
title: "Effects of exercise modality and intensity on the CD4 count in people with HIV: a systematic review and meta-analysis."
description: "Meta-analysis of randomized trials: aerobic and high-intensity training raise CD4 count; other modalities did not."
resource: "https://pubmed.ncbi.nlm.nih.gov/33745404/"
domain: "labs-biomarkers"
id: "33745404"
pmid: "33745404"
journal: "AIDS Care"
authors: "Ferrari Silva B, Oliveira GH, Ferraz Simões C, ..."
published: "2021"
tags: ["CD4 count", "aerobic training", "meta-analysis"]
study_design: "systematic review and meta-analysis"
text_basis: "abstract"
license: "publisher copyright"
export_safe: "false"
generated: {by: "okf-loremaster/extract/<model-id>", at: "2026-03-21T11:31:58Z"}
sources: [{id: "pmid:33745404", resource: "https://pubmed.ncbi.nlm.nih.gov/33745404/"}]
---

# Bottom line

Aerobic training and high-intensity training raised CD4 count; the other modalities and
intensities did not.

- **Design** — systematic review and meta-analysis
- **Population** — people living with HIV
- **Outcome** — absolute CD4 count, cells/mm³
- **Read from** — the abstract

# Predictors reported

| # | Predictor | Operationalization | Timing | Outcome | Type | Effect | p | Direction | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | aerobic training | modality subgroup, pooled mean difference vs. control | measured after the training period | CD4 count | intervention | 79.91 cells/mm³ (95% CI 19.30-140.52) | ≤0.01 | increases | high |

Quoted from the paper, by row:

1. …only aerobic exercise has proved to have a significant effect on CD4 (MD 79.91 cell/ml³ [CI 95% 19.30-140.52],=< 0.01).

# Null or non-significant findings

| # | Predictor | Outcome | Detail |
|---|---|---|---|
| 1 | non-aerobic training modalities | CD4 count | no significant effect in the modality subgroup analysis |

# Vocabulary hints

- **mesh** — CD4 Lymphocyte Count, Exercise Therapy

# Caveats

Pooled across trials with heterogeneous protocols and durations.
````

Three things about that document are the whole point:

**The quote line under the table.** Every effect size is reproduced beside the sentence it came
from, **exactly as published** — typography, brackets, the mangled `=< 0.01` and all. That is the
one thing a producer must never tidy: a cleaned-up quote cannot be checked against the source, so it
is worse than no quote. A number the source text does not contain is removed by a deterministic
post-processing pass, which downgrades the row's confidence and logs a warning rather than failing
the run.

**`# Null or non-significant findings` is never omitted.** A paper that reported none says so
explicitly. An absent section and a null result are different claims, and a validator inserts the
sentinel so omission is impossible rather than merely discouraged.

**`text_basis` and `license` are per document.** Most of PubMed is abstract-only under publisher
copyright; a minority is open access. The bundle records which is which per paper, which is what
makes `export --permissive-only` possible and what stops an agent from weighing an abstract-derived
claim like a full-text one.

*(Example content from [PubMed](https://pubmed.ncbi.nlm.nih.gov/33745404/),
[DOI 10.1080/09540121.2021.1902932](https://doi.org/10.1080/09540121.2021.1902932).)*

---

## The agentic system

A LangGraph pipeline of thirteen nodes, checkpointed to SQLite so an interrupted run resumes where
it stopped. Two of the edges are conditional: curation can send the graph back for another search
round, and ranking can end it early when nothing survived.

```
              plain-language task            or  --charter charter.yaml
                       │
                       ▼
          ┌─────────────────────────┐
          │  charter        DEEP    │   shelf taxonomy · vocabularies · population · outcome
          └────────────┬────────────┘
                       │   ⏸  pause 1 — you read and edit the charter
                       ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │   search  MID  ─▶  dedupe  ─▶  rank  ─▶  screen  FAST  ─▶  curate  MID
  │      ▲                                                        │    │
  │      └────────  re-query round, for the shelves that came up short  │
  └────────────────────────────────┬───────────────────────────────────┘
                       │   ⏸  pause 2 — you approve the curated set
                       ▼
       fulltext  ─▶  extract  DEEP  ─▶  reconcile  ─▶  review
                       │
                       ▼
       emit_okf  ─▶  validate  ─▶  index_vectors
                       │                    │
                       ▼                    ▼
                  <bundle>/          <bundle>.chroma/
                       │                    │
                       └────────┬───────────┘
                                ▼
                  resources/ of a downstream
                  feature-construction agentic system
```

### Agents only for judgment

| Node | Who | What it decides |
|---|---|---|
| `charter` | **DEEP** model | Turns the task into shelves, vocabularies, population and outcome |
| `search` | **MID** model | Writes the PubMed query plan; the HTTP is code |
| `dedupe` | code | PMID / DOI / normalized title |
| `rank` | code | Recency, citations (iCite), and MMR for diversity |
| `screen` | **FAST** model | Include or exclude, and which shelf, one call per abstract |
| `curate` | **MID** model | Per shelf: what to keep, and **what is missing** — which drives the re-query |
| `fulltext` | code | BioC open-access check; the license is recorded verbatim, never inferred |
| `extract` | **DEEP** model | Predictor rows, nulls, vocabulary hints, caveats |
| `reconcile` | code | Numeric verification against the source text; unverifiable numbers are removed |
| `review` | human | Optional sign-off, written into `verified:` — which is where OKF's trust tier comes from |
| `emit_okf` | code | Markdown, indexes, catalog, descriptor |
| `validate` | code | The OKF contract, as a gate with an exit code |
| `index_vectors` | code | Chunks and embeds the **finished bundle**, never a second extraction pass |

That split is the design, not an implementation detail. HTTP, dedup, ranking, MMR, license logic,
file writing, validation, embedding and indexing are all deterministic code; a model is asked only
where a judgment is genuinely required. It keeps runs cheap, reproducible, and debuggable.

Three model roles are bound in config, so you choose the cost/quality tradeoff per role rather than
per run: **FAST** for screening (the highest-volume call by far), **MID** for query planning and
curation, **DEEP** for the charter and extraction. Any provider LiteLLM supports.

### Two pauses, on purpose

A run stops after the charter and again after curation. Both are the cheap moments: a bad shelf
taxonomy caught at the first pause costs nothing, and the same taxonomy caught after extraction has
already been paid for. `--yes` skips both; `--dry-run` prints the plan and its projected cost
having made zero LLM calls.

---

## The interface

Two renderers over the same event stream. **Nodes never print** — they emit typed events, and a
renderer subscribes; that is what lets the console, the TUI and `--json` all be exact views of one
run rather than three code paths that drift.

**Console** (default) — Rich progress per stage, live token and cost meters, the two pauses as
prompts, and a summary at the end.

**Full screen** (`--tui`, needs the `[tui]` extra) — the pipeline down the left with each node's
state, a scrolling log, and a live cost meter. The pauses become dialogs answered with `y` or `n`.
**`q` stops the run rather than killing it**: the work so far is checkpointed and the command prints
the `--resume <id>` that continues it. On a non-terminal it falls back to the console renderer with
a note rather than failing.

**`--json`** — one JSON object per event on stdout, for a wrapper or a CI job.

`inspect` reads a finished bundle back off disk and summarizes it. It works on a bundle whose run is
long gone, on another machine, with no API key:

```
                            shelves
┏━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ shelf ┃ title ┃ papers ┃ full text ┃ predictors ┃ permissive ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ alpha │ Alpha │      6 │         2 │          6 │          1 │
│ beta  │ Beta  │      6 │         3 │          6 │          1 │
│ delta │ Delta │      6 │         3 │          6 │          1 │
│ gamma │ Gamma │      6 │         3 │          6 │          1 │
├───────┼───────┼────────┼───────────┼────────────┼────────────┤
│ total │       │     24 │        11 │         24 │          4 │
└───────┴───────┴────────┴───────────┴────────────┴────────────┘
──────────────────────────────── corpus ─────────────────────────────────
  24 paper(s): 11 read from full text, 13 from the abstract only
  median reported sample size: 203
  24 predictor row(s); 0 paper(s) report a null or non-significant finding
  effect sizes: 11 verified, 13 row(s) reported none
  4 of 24 carry a license that permits redistribution
   study designs
design        papers
cohort study      24
──────────────────────────────── run ────────────────────────────────────
     run id  20260804-004750-c7aa
      built  2026-08-04T00:47:54Z
   duration  3s
       tool  okf-loremaster/0.1.0.dev0
stale after  2027-01-31
model calls  198
     tokens  111,940 (101,682 in / 10,258 out)
```

*(That is the synthetic corpus the test suite builds — `alpha`/`beta`/`gamma`/`delta` are its four
shelves — reproduced verbatim rather than dressed up as a real run.)*

The `effect sizes` line is the one to read: it separates magnitudes that numeric verification
confirmed from rows the paper reported without one, which is the difference between a corpus an
agent can quote with a number and one it can only paraphrase.

---

## Install

Requires Python 3.11 or 3.12.

```bash
conda create -n okf-loremaster python=3.11 -y
conda run -n okf-loremaster pip install -e ".[all,dev]"
```

| Extra | Adds | Cost |
|---|---|---|
| *(base)* | The whole OKF pipeline | — |
| `[vectors]` | Chroma + sentence-transformers | pulls torch; omit if you only want the bundle |
| `[tui]` | The full-screen interface | Textual |
| `[all]` | Both | |
| `[dev]` | pytest, mypy, ruff | |

The install is editable, which records this directory's absolute path. **If you move or rename the
folder, re-run `pip install -e .` from the new location** or imports stop resolving.

---

## Configure

```bash
okf-loremaster init          # writes .env from the template, then checks the environment
```

At minimum, set an LLM API key and the three model roles. Two more are worth setting:

- `OKF_LOREMASTER_NCBI_API_KEY` — free from NCBI, and raises the shared rate limit from 3 to 10
  requests/second.
- `HF_HOME` — a shared Hugging Face cache so the embedding model downloads once. **Keep it outside
  OneDrive, Dropbox or any sync folder**: the hub cache links `snapshots/` into `blobs/` with
  symlinks, which sync clients mangle.

Every variable is annotated in [.env.example](.env.example). Configuration failures are loud and
name the variable that is wrong.

---

## Commands

`okf-loremaster --help` lists all seven; `okf-loremaster <command> --help` lists its flags.
`loremaster` is installed as a shorter alias for the same tool.

### `init`

Write `.env` from the template and check the environment is usable. `--force` overwrites an
existing `.env`.

### `charter "<task>"`

Draft the charter alone — shelf taxonomy, vocabularies, query plan — without building anything.
Edit the result, then build from it.

| Flag | Default | |
|---|---|---|
| `-o, --out <path>` | `charter.yaml` | Where to write it |
| `--vocab <a,b,c>` | from the charter | Comma-separated coding vocabularies |
| `--target-papers <int>` | `180` | Target retained paper count |
| `--shelf-min <int>` / `--shelf-max <int>` | `8` / `40` | Papers per shelf |
| `-v, --verbose <int>` | `0` | Verbosity |

### `build ["<task>"]`

Build a bundle end to end. Pass a task, or `--charter` a drafted one.

| Flag | Default | |
|---|---|---|
| `--charter <path>` | — | Build from an existing `charter.yaml` (then the prompt is optional) |
| `-o, --out <path>` | from config | Bundle output path |
| `--pool-size <int>` | `800` | Candidate pool before screening |
| `--screen-budget <int>` | `400` | Maximum abstracts sent to the screener |
| `--target-papers <int>` | `180` | Target retained paper count |
| `--shelf-min <int>` / `--shelf-max <int>` | `8` / `40` | Papers per shelf |
| `--max-rounds <int>` | `2` | Search rounds including the first; `1` disables re-query |
| `--vocab <a,b,c>` | from the charter | Overrides the charter's vocabularies |
| `--index` | off | Also build the vector index |
| `--review` | off | Human sign-off before emit, written into `verified:` |
| `-y, --yes` | off | Skip both confirmation pauses |
| `--dry-run` | off | Plan and cost the run; makes **zero** LLM calls |
| `--resume <id>` | — | Resume a checkpointed run |
| `--tui` | off | Full-screen interface |
| `--json` | off | Machine-readable events on stdout |
| `-v, --verbose <int>` | `0` | Verbosity |

`--review` is refused in combination with `--yes`, `--dry-run` or `--json`: each of those means
nobody is going to look, and signing anyway would write an attestation naming a person who never saw
the bundle.

### `index <bundle>`

Build the vector index from a bundle that already exists. The store is written to
`<bundle>.chroma` and an existing collection there is replaced. This is how a bundle you edited by
hand gets an index that matches it.

### `validate <bundle>`

Check a bundle against the OKF contract and **exit non-zero if it fails**. Errors are contract
violations (a `domain` that does not match its folder, a missing required key, a section out of
order, a catalog that disagrees with the disk, an unresolvable link, a duplicate `id`); warnings are
quality signals (an untagged document, an empty shelf, an unmapped vocabulary key, a remote
embedding model a consumer will reject). Warnings never fail the gate.

This is the same code the graph runs, reached without a run — which is the only way to check a
bundle somebody else built, or one built six months ago.

### `export <bundle> -o <dest>`

Copy a bundle out. `--permissive-only` keeps just the documents whose licenses permit
redistribution.

The copy is a bundle in its own right — its own indexes, catalog and descriptor `id` — so it
validates and attaches on its own. Three details worth knowing:

- **Retained documents are copied byte for byte.** Re-rendering is where a verbatim quote stops
  being verbatim.
- **The vector index is not copied.** It embeds every document in the source, including the ones the
  filter just removed. Rebuild it with `okf-loremaster index <the copy>`.
- **A document is kept only if its `export_safe` flag and its recorded license agree.** Disagreement
  means the file was hand-edited; a redistribution decision takes the conservative side, and the
  file is named in a warning rather than dropped silently.

An emptied shelf keeps its directory and an index saying so — "no papers survived the filter" and
"this shelf does not exist" are different claims.

### `inspect <bundle>`

Summarize a bundle: shelf sizes, full-text coverage, study designs, median sample size, effect-size
verification, vocabulary hints, the run that built it, and its vector index if there is one. Reads
`_catalog.jsonl` as the spine — the file a downstream consumer actually reads — and falls back to
the documents when it is absent, saying so.

---

## Output contracts

Both outputs are **detected, not configured**: a conforming directory dropped into a downstream
system's `resources/` is the entire setup step.

### The OKF bundle

Eight rules a consumer may rely on, all of them enforced by `validate`:

1. **Required frontmatter is `title` + `domain` only.** Everything else is optional; a missing
   optional field degrades a citation, never the run. `id` falls back to the filename stem.
2. **`domain` must equal the containing folder name.** A mismatch is a validation error, not a
   silent re-shelve — it is almost always a copy-paste bug, and it hides a paper where nobody looks.
3. **`index.md` is reserved** at the root and in each shelf, and is regenerated. Never a document.
4. **`title`, `description` and `tags` are the search surface.** Downstream retrieval is fuzzy
   token-set matching over title + description + tags + journal, so a document titled "Study 3
   final" is effectively unfindable. `description` is in the haystack because it states a *finding*
   rather than a topic.
5. **Shelves come from the corpus, not from a list in the consumer's code.** They are read from the
   directory tree and the root index; a human-readable title lives in the shelf's `index.md`.
6. **A document is referenced three ways** — `id`/PMID, bare filename, or `domain/file.md` — and all
   three resolve. Agents cite inconsistently, and a lookup miss wastes a whole turn.
7. **Frontmatter is one key per line**: strictly quoted flat scalars (including bools and integers),
   string lists, and — for the three nested keys OKF v0.2 defines (`generated`, `verified`,
   `sources`) — YAML flow style on that one line. Flattening them to `generated_by`/`generated_at`
   forfeits conformance, and writing them across indented lines breaks a dependency-free line
   parser. Flow style on one line is valid YAML to a spec consumer and one opaque string to a line
   parser, which is the only shape that satisfies both.
8. **`resource_descriptor.yaml` is optional and authoritative when present.** Its `id` supplies the
   resource id, `domains: {slug: title}` supplies the shelf titles, and a consumer that ignores the
   rest — `built_on`, `stale_after`, `tool`, `charter_digest`, the `vectors:` block — still attaches
   cleanly. Unknown keys are ignored, never rejected.

The word for a folder is **shelf** in conversation and `domain` in frontmatter. The key is `domain`;
"shelf" never appears in a file. `_catalog.jsonl` sits outside a `*.md` walk by design and carries
one row per document, including `unmapped_vocab`, which is deliberately nowhere else.

`tests/test_afce_contract.py` re-implements a consumer from these rules — its own line parser, its
own resolver, its own haystack — and checks a finished bundle against it, rather than reading the
bundle back with the code that wrote it.

### The vector index

Optional and derived. It is built by **walking the finished bundle**, never by a second extraction
pass, so `index <bundle>` a year later produces the same store `build --index` did on the day. It
sits *beside* the bundle rather than inside it, so nothing that copies a bundle drags a binary store
along and nothing that reads one mistakes the store for a shelf.

Each paper contributes two levels of chunk: one **concept** chunk carrying the whole document except
the predictor table, and one **predictor** chunk per table row, with the population, outcome
definition and bottom line around it so a row retrieved on its own still means something.

Chunk metadata is `source`, `title`, `id`, `chunk_index`, `chunk_level`, `pmid`, `domain`,
`journal`, `published`, `study_design`, `n`, `timing`, `confidence`, `evidence_type`, `text_basis`
and `license`. A missing value is `""` and never null — Chroma rejects nulls — except `n`, which is
an integer where the paper reported one so a numeric filter works.

> **`timing`, `confidence` and `evidence_type` describe a predictor row, so a concept chunk carries
> `""` for all three.** A filter on any of them must either allow `""` or select
> `chunk_level == "predictor"`. One that does neither silently excludes every concept chunk — half
> the corpus — and looks like it simply found less. That is what `chunk_level` exists for.

The store's `resource_descriptor.yaml` declares the embedding model, its **resolved** revision, the
dimensions, and the distance metric (`cosine`). The metric is declared rather than left to default:
Chroma's own default is L2, and a consumer that guessed wrong would get results in a different order
and no error at all. Embedding models resolve through config and default to a local, revision-pinned
one — a downstream system that enforces a local-embeddings boundary will reject a remote one on
attach, and `validate` warns before it gets that far.

---

## Worked example

```bash
# 1. Draft the charter and read it. This is the cheap moment to fix the taxonomy.
okf-loremaster charter "predictors of 30-day readmission after heart failure hospitalization" \
    --vocab icd10,loinc,rxnorm -o hf-readmission.yaml

# 2. See what the run will do and roughly what it will cost. Zero LLM calls.
okf-loremaster build --charter hf-readmission.yaml --dry-run

# 3. Build it, with the vector index, full screen.
okf-loremaster build --charter hf-readmission.yaml --index --tui -o bundles/hf-readmission

# 4. Check and summarize.
okf-loremaster validate bundles/hf-readmission
okf-loremaster inspect  bundles/hf-readmission
```

Then hand it to the downstream system — the whole integration is a copy:

```bash
mkdir -p ../my-feature-agent/resources
cp -R bundles/hf-readmission        ../my-feature-agent/resources/okf
cp -R bundles/hf-readmission.chroma ../my-feature-agent/resources/rag
```

Its evidence agents pick both up by detection: the OKF shelf is browsed and read whole, the vector
store is queried for recall, and each becomes a separate reviewer rather than one agent's blend of
the two. Nothing is imported from this package and nothing is imported from that one.

To share the corpus outside your institution, filter it to what may be redistributed first:

```bash
okf-loremaster export bundles/hf-readmission -o bundles/hf-readmission-public --permissive-only
okf-loremaster index  bundles/hf-readmission-public       # the store is not copied — rebuild it
okf-loremaster validate bundles/hf-readmission-public
```

---

## Conduct and provenance

This tool uses NCBI's public APIs — E-utilities, BioC, PubTator and iCite — at their documented rate
limits, through **one shared limiter** because the limit is enforced per IP across all of them, and
**never scrapes PubMed or PMC web pages**. Set `OKF_LOREMASTER_NCBI_EMAIL` so NCBI can contact you
about your usage, as their access policy asks.

Each document records the license reported by its source, verbatim and never inferred. Most PubMed
records are abstracts under publisher copyright and are not redistributable; that is the normal case
rather than a failure, and `export --permissive-only` is how a shareable subset is produced.

Every bundle carries a `stale_after` date, the charter digest it was built from, the models that
wrote it, and — with `--review` — who signed it off. OKF v0.2 derives its trust tier from the
`verified` key specifically, so an unsigned bundle is *unverified* rather than merely unannotated.

---

## Status

Build steps 0–10 of 10 are complete. `build` runs end to end and writes a validated OKF bundle,
`--tui` drives it full screen, `index` derives the vector store, and `validate`, `export` and
`inspect` work on any conforming bundle without an API key. See
[Build_Progress.md](Build_Progress.md) for the datestamped log, including measurements and
rejected alternatives.

```bash
conda run -n okf-loremaster pytest        # the suite never reaches the network
conda run -n okf-loremaster mypy src/
conda run -n okf-loremaster ruff check src/ tests/
```

## License

Not yet chosen. Treat this as unlicensed and internal until one is set.
