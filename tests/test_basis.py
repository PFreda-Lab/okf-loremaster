"""`--basis`: what a run is willing to read, and what saying so costs.

The default is the whole point of the feature. A bundle built without the flag has to be
byte-for-byte the bundle it was before `--basis` existed — no extra requests, no extra
line in `index.md`, no extra key in the descriptor — because every restriction here is
a corpus the user chose to make smaller, and a reader who sees the policy line on every
bundle stops reading it on the one where it matters.

The other two are enforced in `rank`, before the screener, and each of them is a claim
the bundle then makes about itself:

- `abstract` never calls BioC at all. Not "calls it and ignores the answer" — the
  request is the expensive part, and a run that made it anyway would look identical from
  the outside while paying full price.
- `full-text` asks BioC whether each paper is genuinely in the open-access subset,
  because a PMC id is not that answer. Availability is read off the body: BioC says "not
  open access" with HTTP 200.

And the policy survives a resume. `--basis` is spent by the time a run is resumable, so
a *different* flag on `--resume` is ignored rather than half-applied to a corpus that was
already chosen under the old one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer
import yaml

from okf_loremaster.basis import TextBasisPolicy
from okf_loremaster.cli import app
from okf_loremaster.emitters.okf import _basis_policy
from okf_loremaster.llm.estimate import project_spend
from okf_loremaster.okf.layout import DESCRIPTOR_FILENAME, INDEX_FILENAME, LOG_FILENAME
from okf_loremaster.okf.reader import fact_list
from okf_loremaster.run import PastRun, RunOptions, _resumed_basis
from okf_loremaster.schemas import RunManifest, TextBasis

from fake_ncbi import has_full_text
from graph_runs import Run, charter_for, full_run

READ_FROM = "Read from"


def basis_run(
    settings_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: TextBasisPolicy,
) -> Any:
    return full_run(settings_factory, tmp_path, monkeypatch, basis=policy)


def facts(bundle: Path, filename: str, heading: str) -> dict[str, str]:
    """The `- **Label** — value` pairs under one `## ` heading of a root file.

    Scoped to the section rather than read off the whole file, so a test asserting that
    the policy line is absent cannot be satisfied by its absence from some other part of
    the document. `body_sections` splits on `# ` only — a root file's sections are one
    level down, and a `##` inside a document section deliberately belongs to it.
    """
    lines = (bundle / filename).read_text(encoding="utf-8").splitlines()
    start = lines.index(f"## {heading}") + 1
    end = next(
        (n for n in range(start, len(lines)) if lines[n].startswith("## ")), len(lines)
    )
    return fact_list("\n".join(lines[start:end]))


def bases(run: Run) -> set[TextBasis]:
    return {record.text_basis for record in run.records}


# --- the flag ---------------------------------------------------------------


def test_the_default_is_read_whatever_each_paper_offers() -> None:
    """One enum with a default, not two boolean flags.

    `--abstracts` and `--full-text` as separate switches have a fourth state — both — that
    means nothing, and the answer to "what happens with neither" has to be documented
    rather than read off the signature.
    """
    group = typer.main.get_command(app)
    params = {
        param.name: param for param in group.commands["build"].params  # type: ignore[attr-defined]
    }

    assert params["basis"].default is TextBasisPolicy.ANY
    assert set(params["basis"].type.choices) == {"any", "abstract", "full-text"}


# --- abstracts only ---------------------------------------------------------


async def test_abstract_only_never_asks_bioc_for_anything(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saving is the requests, not the parsing.

    Fetching full text and then declining to read it is the same run at the same price
    with a smaller bundle, which is the one outcome the flag must not produce.
    """
    run = await basis_run(settings_factory, tmp_path, monkeypatch, TextBasisPolicy.ABSTRACT)

    assert run.fake.bioc_requests == []
    assert bases(run) == {TextBasis.ABSTRACT}


async def test_abstract_only_keeps_a_bundle_that_still_validates(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrower corpus, not a broken one. Every root file and every document still
    has to satisfy the same validator the default bundle does."""
    from okf_loremaster.okf.validate import validate_bundle

    run = await basis_run(settings_factory, tmp_path, monkeypatch, TextBasisPolicy.ABSTRACT)
    report = validate_bundle(Path(run.state["bundle"]))

    assert report.errors == (), report.lines()
    assert run.state["validated"] is True


# --- open-access full text only ---------------------------------------------


async def test_full_text_only_keeps_exactly_the_papers_bioc_will_serve(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PMC id is not availability, and this is where that distinction is paid for.

    Every paper in the synthetic corpus carries a PMC id and only two in five are open
    access, so a run that filtered on the id alone would keep them all and then emit a
    bundle of abstracts labeled as a full-text one.
    """
    run = await basis_run(settings_factory, tmp_path, monkeypatch, TextBasisPolicy.FULL_TEXT)

    assert run.records
    assert bases(run) == {TextBasis.FULL_TEXT}
    assert all(has_full_text(record.pmid) for record in run.records)


async def test_a_corpus_narrowed_by_policy_says_so_rather_than_looking_thin(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sixty papers where two hundred were asked for reads as a thin literature.

    It is not one here — it is a filter the user asked for — and the only thing standing
    between those two readings is a warning that names the flag.
    """
    run = await basis_run(settings_factory, tmp_path, monkeypatch, TextBasisPolicy.FULL_TEXT)

    assert any(
        "--basis full-text kept" in note and "no open-access full text" in note
        for note in run.warnings
    ), run.warnings


async def test_a_uniform_basis_is_declared_to_stop_being_read_as_a_quality_spread(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`strength` reads what a paper was read from as one of its axes.

    Under a uniform basis that axis adds the same term to every paper, so it separates
    none of them — and a reader comparing two rows in the same bundle would otherwise
    have no way to know one of the four inputs had gone flat. The arithmetic is
    deliberately left alone so grades still mean the same thing across bundles.
    """
    run = await basis_run(settings_factory, tmp_path, monkeypatch, TextBasisPolicy.ABSTRACT)

    assert any(
        "gave every paper the same text basis" in note and "comparable across bundles" in note
        for note in run.warnings
    ), run.warnings


# --- what the bundle records ------------------------------------------------


async def test_a_restricted_run_tells_a_consumer_which_policy_produced_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-document `text_basis` cannot answer this and never could.

    "Every paper here is abstract-only" is two different bundles — one where that is all
    the literature offered, one where it is all the run asked for — and a consumer
    deciding how much to trust the corpus is asking which.
    """
    run = await basis_run(settings_factory, tmp_path, monkeypatch, TextBasisPolicy.ABSTRACT)
    bundle = Path(run.state["bundle"])

    assert facts(bundle, INDEX_FILENAME, "Run")[READ_FROM] == TextBasisPolicy.ABSTRACT.label
    assert facts(bundle, LOG_FILENAME, "Request")[READ_FROM] == TextBasisPolicy.ABSTRACT.label

    descriptor = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert descriptor["text_basis_policy"] == "abstract"


async def test_the_default_bundle_is_the_bundle_it_was_before_the_flag_existed(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default has to cost a reader nothing at all.

    A `Read from: full text where available` line on every bundle ever built is a line
    every reader learns to skip, which spends the one place the restricted runs have to
    be noticed.
    """
    run = await full_run(settings_factory, tmp_path, monkeypatch)
    bundle = Path(run.state["bundle"])

    assert READ_FROM not in facts(bundle, INDEX_FILENAME, "Run")
    assert READ_FROM not in facts(bundle, LOG_FILENAME, "Request")

    descriptor = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert "text_basis_policy" not in descriptor

    # And the corpus itself is mixed, which is what makes the assertions above a
    # statement about the policy rather than about a corpus that happens to be uniform.
    assert bases(run) == {TextBasis.FULL_TEXT, TextBasis.ABSTRACT}


@pytest.mark.parametrize("recorded", ["", "any", "not-a-policy"])
def test_a_manifest_naming_no_policy_renders_no_line(recorded: str) -> None:
    """Three ways of saying "nothing to report", including a bad one.

    Bundles built before this key existed carry the empty string, and a hand-edited
    manifest can carry anything. Neither is worth failing a read over — the honest
    rendering of an unrecognized policy is the same as the default's, which is silence.
    """
    assert _basis_policy(RunManifest(run_id="r", text_basis_policy=recorded)) == ""


# --- resuming ---------------------------------------------------------------


def past(basis: str) -> PastRun:
    return PastRun(
        run_id="20260817-abcd",
        started=None,
        prompt="a prompt",
        reached="screen",
        finished=False,
        basis=basis,
    )


def options(basis: TextBasisPolicy) -> RunOptions:
    return RunOptions(prompt="a prompt", basis=basis)


def test_a_fresh_run_takes_the_flag_it_was_given() -> None:
    policy, note = _resumed_basis(options(TextBasisPolicy.FULL_TEXT), None)

    assert policy is TextBasisPolicy.FULL_TEXT
    assert note == ""


def test_a_resume_takes_the_policy_its_corpus_was_chosen_under() -> None:
    """`--basis` is spent in `rank`, so by the time a run can be resumed it is spent.

    Honoring a new flag would restrict only the nodes that have yet to run, over a pool
    selected and screened under the old rule, and then write a manifest naming the new
    one — a bundle whose stated policy is not the policy that produced its contents.
    """
    policy, note = _resumed_basis(options(TextBasisPolicy.ABSTRACT), past("full-text"))

    assert policy is TextBasisPolicy.FULL_TEXT
    # Both policies named: which one was asked for and which one the run is actually on.
    assert "--basis abstract was ignored" in note
    assert "--basis full-text" in note


def test_a_matching_flag_on_a_resume_is_not_worth_a_warning() -> None:
    policy, note = _resumed_basis(options(TextBasisPolicy.ABSTRACT), past("abstract"))

    assert policy is TextBasisPolicy.ABSTRACT
    assert note == ""


@pytest.mark.parametrize("recorded", ["", "not-a-policy"])
def test_a_checkpoint_with_no_readable_basis_resumes_on_the_default(recorded: str) -> None:
    """Checkpoints predating the flag carry nothing, and that is exactly what they did.

    Failing a resume over a string in a sqlite column would strand every run built
    before this feature, for a value whose absence has one obvious meaning.
    """
    policy, note = _resumed_basis(options(TextBasisPolicy.ANY), past(recorded))

    assert policy is TextBasisPolicy.ANY
    assert note == ""


# --- what it does to the projection -----------------------------------------


def projection(settings_factory: Any, basis: TextBasisPolicy) -> Any:
    return project_spend(
        charter_for(),
        settings=settings_factory(model_balanced="m", model_fast="m", model_reasoning="m"),
        pool=[],
        screen_budget=100,
        target_papers=24,
        basis=basis,
    )


@pytest.mark.parametrize(
    ("basis", "phrase"),
    [
        (TextBasisPolicy.ABSTRACT, "every one read from its abstract"),
        (TextBasisPolicy.FULL_TEXT, "every one read from open-access full text"),
    ],
)
def test_a_restricted_run_prices_its_source_length_instead_of_guessing_it(
    settings_factory: Any, basis: TextBasisPolicy, phrase: str
) -> None:
    """The projection's largest unknown is removed, not adjusted.

    On the default, how much full text a run reaches is a rate applied to a PMC-id count.
    Under either restriction `rank` has already settled it, so the extraction line says
    what it knows — and drops the caveat about the thing it no longer has to guess.
    """
    estimate = projection(settings_factory, basis)
    extract = next(node for node in estimate.nodes if node.node == "extract")

    assert phrase in extract.basis
    assert not any("largest thing it cannot know" in note for note in estimate.notes)
    assert any(f"`--basis {basis.value}` fixes what each paper is read from" in note
               for note in estimate.notes)


def test_the_default_projection_still_says_what_it_is_guessing(settings_factory: Any) -> None:
    """The hedge belongs on the run that has something to hedge about, and only there."""
    estimate = projection(settings_factory, TextBasisPolicy.ANY)

    assert any("largest thing it cannot know" in note for note in estimate.notes)


def test_full_text_prices_higher_than_abstracts_over_the_same_corpus(
    settings_factory: Any,
) -> None:
    """Extraction is one call per paper and sets a run's price. Reading whole articles
    costs multiples of reading abstracts, and an estimate a human is shown before
    deciding whether to pay it has to move when the flag does."""
    abstracts = projection(settings_factory, TextBasisPolicy.ABSTRACT)
    full = projection(settings_factory, TextBasisPolicy.FULL_TEXT)

    def extract_tokens(estimate: Any) -> int:
        node = next(n for n in estimate.nodes if n.node == "extract")
        return int(node.prompt_tokens)

    assert extract_tokens(full) > extract_tokens(abstracts)
