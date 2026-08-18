"""The bundle a finished run leaves on disk, read back the way a consumer will read it.

The step 7 gate is `test_the_golden_bundle_passes_the_hard_gate`: one whole run over the
synthetic corpus, emitted, then validated from the files rather than from the objects
that wrote them. Everything else in this module is a property of those same files, so
the suite is asking about one bundle from several directions rather than building a new
one per assertion.

Nothing here reads a `ConceptRecord` to decide whether the bundle is right. The gap
between what the pipeline believes and what it wrote is exactly the gap the emitter can
fall into, and a test that closed it by looking at run state would never notice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from okf_loremaster.emitters.okf import _expansion_is_mechanical, search_markdown
from okf_loremaster.okf.frontmatter import load, parse, split
from okf_loremaster.okf.layout import (
    ABSTRACT_SECTION,
    BODY_SECTIONS,
    CATALOG_FILENAME,
    CHARTER_FILENAME,
    DESCRIPTOR_FILENAME,
    INDEX_FILENAME,
    LOG_FILENAME,
    PREDICTOR_INDEX_TYPE,
    PREDICTORS_FILENAME,
    REQUIRED_BODY_SECTIONS,
    SEARCH_FILENAME,
    SEARCH_STRATEGY_TYPE,
    UNVERIFIED_CELL,
)
from okf_loremaster.okf.reader import fact_list, markdown_table, read_bundle
from okf_loremaster.okf.validate import validate_bundle
from okf_loremaster.review import HUMAN_PREFIX, Signoff
from okf_loremaster.schemas import (
    ConceptRecord,
    ExecutedQuery,
    RunManifest,
    StrengthGrade,
    VerificationSummary,
)
from test_verification import fabricating

from fake_ncbi import TOPICS
from graph_runs import POOL_SIZE, TARGET, Run, charter_for, full_run, scripted_run

# What a flat frontmatter value may open with once the emitter has had its way: a quote
# for a scalar, a bracket or brace for the nested blocks OKF v0.2 requires.
QUOTED_OR_FLOW = ('"', "[", "{")


async def golden(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> tuple[Run, Path]:
    run = await full_run(settings_factory, tmp_path, monkeypatch, **overrides)
    return run, Path(run.state["bundle"])


# --- the gate ---------------------------------------------------------------


async def test_the_golden_bundle_passes_the_hard_gate(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 7 gate. A run that emitted a bundle its own validator rejects has
    shipped a contract violation to the only consumer that will ever notice."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    report = validate_bundle(bundle)
    assert report.errors == (), report.lines()
    assert report.ok
    assert report.documents == TARGET
    assert report.topics == len(TOPICS)

    # And the run reached the same verdict in-graph, over the same code path.
    assert run.state["validated"] is True
    assert run.state["validation_errors"] == []


async def test_the_bundle_has_every_file_a_consumer_is_promised(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for filename in (
        INDEX_FILENAME,
        CATALOG_FILENAME,
        DESCRIPTOR_FILENAME,
        LOG_FILENAME,
        CHARTER_FILENAME,
        PREDICTORS_FILENAME,
        SEARCH_FILENAME,
    ):
        assert (bundle / filename).is_file(), filename

    for slug in TOPICS:
        assert (bundle / slug / INDEX_FILENAME).is_file(), slug

    # `<pmid>_<Author>.md`, which is the shape AFCE's three-way resolver expects.
    documents = list(read_bundle(bundle).documents())
    assert len(documents) == TARGET
    for document in documents:
        assert document.filename == f"{document.pmid}_{document.filename.split('_', 1)[1]}"
        assert document.filename.endswith(".md")


# --- frontmatter ------------------------------------------------------------


async def test_every_flat_value_is_quoted_and_every_nested_one_is_flow_style(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The downstream parser is dependency-free and hands back strings. A bare `n: 200`
    is a number to one reader and a string to the other, which is the ambiguity the
    whole discipline exists to remove — so it is checked on the bytes, not the model."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for path in sorted(bundle.rglob("*.md")):
        block, _ = split(path.read_text(encoding="utf-8"))
        for line in block.splitlines():
            if not line.strip():
                continue
            _key, _, raw = line.partition(":")
            assert raw.strip().startswith(QUOTED_OR_FLOW), f"{path.name}: {line}"


async def test_both_parsers_see_the_same_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One key per line for a line-parser, valid YAML for a spec consumer. A block that
    means two different things depending on who opens it is the failure flow style
    prevents, and only reading it twice can prove it did not happen."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for path in sorted(bundle.rglob("*.md")):
        block, _ = split(path.read_text(encoding="utf-8"))
        strict = parse(block)
        loaded = yaml.safe_load(block)
        assert strict == loaded, path.name
        # And the nested blocks really did survive as structures rather than as text.
        if "generated" in strict:
            assert isinstance(strict["generated"], dict)
            assert set(strict["generated"]) == {"by", "at"}


async def test_the_key_is_domain_it_equals_the_folder_and_topic_never_appears(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Topic" is the human word. A file carrying both keys is filed by whichever one
    the reader happens to prefer, which is a bundle that reorganizes itself per tool."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)
    parsed = read_bundle(bundle)

    for document in parsed.documents():
        assert document.domain == document.path.parent.name
        assert "topic" not in document.fields
    for topic in parsed.topics:
        assert topic.index is not None
        assert topic.index.domain == topic.slug
        assert "topic" not in topic.index.fields

    # The root sits in no domain folder, so it must not claim one.
    assert parsed.index is not None
    assert "domain" not in parsed.index.fields


async def test_the_required_pair_is_present_on_every_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for document in read_bundle(bundle).documents():
        assert document.title
        assert document.domain
        # Optional, but the search surface is title + tags + journal and an untagged
        # document is findable by its title alone.
        assert document.tags


# --- the body ---------------------------------------------------------------


async def test_the_required_sections_are_always_present_in_order(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five sections every document has, two more it may have, and one order for all of
    them.

    `# Abstract` and `# Interactions` are omitted when there is nothing to put in them —
    a paper PubMed served no abstract for, and the common case of a study reporting no
    interaction between its own predictors. The five that are required are required
    because an absent one and an empty one are different claims about the paper.
    """
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for document in read_bundle(bundle).documents():
        headings = [name for name, _ in document.sections()]
        assert set(headings) <= set(BODY_SECTIONS), document.path.name
        assert set(REQUIRED_BODY_SECTIONS) <= set(headings), document.path.name
        # Present or absent, never reordered: the order is what makes the file scannable
        # in the same shape every time, and a reader who learns it should not have to
        # re-learn it per document.
        assert headings == [name for name in BODY_SECTIONS if name in headings], (
            document.path.name
        )
        for _name, text in document.sections():
            assert text.strip(), document.path.name


async def test_a_paper_with_no_null_finding_says_so_rather_than_omitting_the_section(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant, checked where it is actually consumed. An absent section and an
    absent finding are different claims, and a downstream agent cannot tell them apart
    once the section is gone."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    empties = 0
    for document in read_bundle(bundle).documents():
        section = document.section("Null or non-significant findings")
        assert section is not None
        assert section.strip()
        # The sentinel is rendered as prose, never as a row reading "none reported".
        if "None reported" in section:
            empties += 1
            assert "|" not in section
    assert empties, "no paper exercised the sentinel, so nothing was proved"


async def test_the_quotes_are_reproduced_verbatim_beside_the_row_they_belong_to(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Numbered against the table's `#` column. A quote list that did not key back to a
    row would be a paragraph of sentences with nothing to attach them to."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)
    by_pmid = {record.pmid: record for record in run.records}

    checked = 0
    for document in read_bundle(bundle).documents():
        record = by_pmid[document.pmid]
        section = document.section("Predictors reported") or ""
        for index, row in enumerate(record.extraction.predictors, start=1):
            if not row.quote:
                continue
            assert f"{index}. {row.quote}" in section, document.path.name
            checked += 1
    assert checked


async def test_an_effect_verification_removed_is_marked_rather_than_printed(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The number the extractor invented must not reappear in the emitted table.

    `downgraded()` keeps `effect_raw` so the paper trail survives, which means the
    fabricated string is still on the row when the emitter reaches it. Rendering it
    would undo step 6 in the one artifact anybody reads.
    """
    run, bundle = await golden(
        settings_factory, tmp_path, monkeypatch, scripted=scripted_run(extract=fabricating)
    )
    summary: VerificationSummary | None = run.state["verification"]
    assert summary is not None and summary.effects_dropped == TARGET

    for document in read_bundle(bundle).documents():
        section = document.section("Predictors reported") or ""
        assert "4.44" not in section, document.path.name
        assert UNVERIFIED_CELL in section, document.path.name

    # And the bundle is still a valid bundle: a dropped number is a downgrade, not a
    # structural failure.
    assert validate_bundle(bundle).ok


# --- the abstract -----------------------------------------------------------


async def test_the_abstract_is_written_unless_it_is_turned_off(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On by default, and off only when asked. The flag is the whole feature, so the
    thing worth checking is that the default did not quietly become the other one."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    documents = list(read_bundle(bundle).documents())
    assert documents
    assert all(document.section(ABSTRACT_SECTION) for document in documents)


async def test_no_abstract_leaves_the_section_out_of_every_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every document, not most of them, and still a valid bundle: `# Abstract` is
    outside `REQUIRED_BODY_SECTIONS` precisely so a corpus may be written without it."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch, abstracts=False)

    documents = list(read_bundle(bundle).documents())
    assert documents
    for document in documents:
        headings = [name for name, _ in document.sections()]
        assert ABSTRACT_SECTION not in headings, document.path.name
        assert set(REQUIRED_BODY_SECTIONS) <= set(headings), document.path.name
        assert headings == [name for name in BODY_SECTIONS if name in headings], (
            document.path.name
        )

    report = validate_bundle(bundle)
    assert report.errors == (), report.lines()


async def test_no_abstract_removes_the_section_and_disturbs_nothing_else(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim the flag makes: the same papers, read the same way, minus one section.

    Two runs off the same fixtures and the same scripted model, compared document for
    document. A flag that also changed which papers were kept, or what a table said about
    them, would be a different corpus wearing the same name — and the only way to know is
    to build both and diff them.
    """
    first, second = tmp_path / "with", tmp_path / "without"
    first.mkdir()
    second.mkdir()
    _, kept = await golden(settings_factory, first, monkeypatch)
    _, dropped = await golden(settings_factory, second, monkeypatch, abstracts=False)

    with_abstract = {d.pmid: d for d in read_bundle(kept).documents()}
    without = {d.pmid: d for d in read_bundle(dropped).documents()}
    assert set(with_abstract) == set(without)

    for pmid, document in with_abstract.items():
        other = without[pmid]
        assert document.filename == other.filename
        assert document.domain == other.domain
        # Every section but the abstract, byte for byte, in the same order.
        assert [
            (name, text) for name, text in document.sections() if name != ABSTRACT_SECTION
        ] == list(other.sections()), pmid
        # And the one that went was not empty to begin with, or this proves nothing.
        assert (document.section(ABSTRACT_SECTION) or "").strip()


async def test_a_bundle_with_no_abstracts_says_so_rather_than_looking_unlucky(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing `# Abstract` already means "PubMed had none". A whole corpus of them has
    to be able to say which of the two happened, and the documents cannot — so the log,
    the root index and the descriptor do, and only when the answer is the unusual one."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch, abstracts=False)

    manifest = run.state["manifest"]
    assert manifest is not None and manifest.abstracts is False

    for filename in (LOG_FILENAME, INDEX_FILENAME):
        text = (bundle / filename).read_text(encoding="utf-8")
        assert "--no-abstract" in text, filename
    descriptor = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert descriptor["abstracts"] is False


async def test_a_default_bundle_says_nothing_about_abstracts_at_all(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the rule. A line announcing that the normal thing happened, on
    every bundle ever built, teaches readers to skip the place the unusual one is said."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    manifest = run.state["manifest"]
    assert manifest is not None and manifest.abstracts is True

    for filename in (LOG_FILENAME, INDEX_FILENAME):
        assert "--no-abstract" not in (bundle / filename).read_text(encoding="utf-8"), filename
    descriptor = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert "abstracts" not in descriptor


# --- evidence strength ------------------------------------------------------


async def test_every_document_says_how_strong_its_study_is(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two flat keys, not one nested block. Nested would be legal flow style, but rule 7
    reserves that for the three structures OKF v0.2 actually nests — anything else comes
    back to a line parser as one opaque string it has to re-parse."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for document in read_bundle(bundle).documents():
        grade = document.fields.get("strength")
        assert grade in {g.value for g in StrengthGrade if g is not StrengthGrade.UNGRADED}, (
            document.filename
        )
        assert 0.0 <= float(document.fields["strength_score"]) <= 1.0, document.filename


async def test_the_row_a_strength_belongs_to_is_the_row_it_is_printed_on(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pairing is positional, which is the kind of thing that renders perfectly while
    being entirely wrong. Checked against the records, cell by cell."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)
    by_pmid = {record.pmid: record for record in run.records}

    checked = 0
    for document in read_bundle(bundle).documents():
        record = by_pmid[document.pmid]
        assert record.strength is not None
        table = markdown_table(document.section("Predictors reported") or "")
        assert len(table) == len(record.strength.rows), document.filename
        for cell, scored in zip(table, record.strength.rows, strict=True):
            assert cell["Strength"] == f"{scored.grade.value} {scored.score:.2f}"
            checked += 1
    assert checked


async def test_strength_and_confidence_are_two_columns_because_they_are_two_questions(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A well-read row from a weak study is `high` and `limited`, and a reader shown only
    one of the two draws the wrong conclusion from either."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for document in read_bundle(bundle).documents():
        table = markdown_table(document.section("Predictors reported") or "")
        for row in table:
            assert row["Confidence"] and row["Strength"], document.filename


async def test_the_bottom_line_says_what_the_score_had_nothing_to_go_on(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A score whose gaps are invisible is one nobody can argue with. The synthetic
    corpus prints no covariate list on a paper with no numbers, so at least one document
    has to admit an unmeasured component rather than quietly averaging over it."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    facts = [
        fact_list(document.section("Bottom line") or "")
        for document in read_bundle(bundle).documents()
    ]
    stated = [entry["Evidence strength"] for entry in facts if "Evidence strength" in entry]

    assert len(stated) == TARGET
    assert any("nothing to score on" in line for line in stated)


async def test_the_topic_index_lets_a_reader_choose_a_paper_by_strength(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The browse table is where the choice of which file to open is actually made."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for topic in read_bundle(bundle).topics:
        assert topic.index is not None
        rows = markdown_table(topic.index.body)
        assert rows, topic.slug
        for row in rows:
            assert row["strength"], f"{topic.slug} {row['pmid']}"


# --- the predictor index ----------------------------------------------------


async def test_the_predictor_index_is_a_third_kind_of_file_and_not_a_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It cuts across every topic, so it sits at the root beside the folders rather than
    in one of them — and a `domain` key there would make it a document filed in a folder
    that does not exist, which is the rule a consumer rejects the whole bundle over."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    fields, body = load((bundle / PREDICTORS_FILENAME).read_text(encoding="utf-8"))
    assert fields["type"] == PREDICTOR_INDEX_TYPE
    assert fields["title"] and fields["description"]
    assert "domain" not in fields
    assert body.strip()

    # And the reader agrees: it is not one of the documents, and no topic gained one.
    parsed = read_bundle(bundle)
    assert all(document.path.name != PREDICTORS_FILENAME for document in parsed.documents())
    assert len(list(parsed.documents())) == TARGET


async def test_every_line_of_the_predictor_index_is_an_address_that_resolves(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the whole file rests on. An index a reader can use *instead of* the
    corpus is one that will be, and then the quotes and provenance stop being opened —
    so every row has to name a file that exists and a row number that is really in it."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    _fields, body = load((bundle / PREDICTORS_FILENAME).read_text(encoding="utf-8"))
    rows = markdown_table(body)
    assert rows, "the golden corpus reports one predictor per paper, so it must recur"

    for entry in rows:
        target = entry["paper"].split("](", 1)[1].rstrip(")")
        assert (bundle / target).is_file(), target
        # The `#` column of that paper's own table, which is the second half of the
        # address: without it a reader lands on a file and not on a finding.
        document = next(
            d for d in read_bundle(bundle).documents() if d.path == bundle / target
        )
        numbers = {
            r["#"] for r in markdown_table(document.section("Predictors reported") or "")
        }
        assert entry["row"] in numbers, f"{target} has no row {entry['row']}"


async def test_the_predictor_index_prints_the_papers_own_words_not_the_cluster_label(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heading is a lexical guess about a field the reader knows better. What each
    paper actually wrote has to stay on the page, or the guess cannot be disputed."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    _fields, body = load((bundle / PREDICTORS_FILENAME).read_text(encoding="utf-8"))
    for entry in markdown_table(body):
        assert entry["as measured"]
        assert entry["direction"]
        assert entry["strength"]


async def test_the_descriptor_and_the_root_index_both_point_at_the_predictor_index(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 4 searches over documents, so a file that is not one is invisible unless
    something names it. Both places, because a consumer reads one or the other."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    payload = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert payload["predictors"] == PREDICTORS_FILENAME
    assert PREDICTORS_FILENAME in (bundle / INDEX_FILENAME).read_text(encoding="utf-8")


# --- the search strategy ----------------------------------------------------


async def test_the_search_strategy_is_a_root_file_and_not_a_document(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It describes how the whole corpus was found, so it belongs beside the topic
    folders rather than inside one — and the `domain` trap is the same one
    `predictors.md` sits next to."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    fields, body = load((bundle / SEARCH_FILENAME).read_text(encoding="utf-8"))
    assert fields["type"] == SEARCH_STRATEGY_TYPE
    assert fields["title"] and fields["description"]
    assert "domain" not in fields
    assert body.strip()

    parsed = read_bundle(bundle)
    assert all(document.path.name != SEARCH_FILENAME for document in parsed.documents())
    assert len(list(parsed.documents())) == TARGET


async def test_every_query_reaches_the_search_strategy_with_what_pubmed_made_of_it(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the file exists for. A term is reproducible only alongside what
    PubMed made of it: a field tag it does not know is rewritten rather than rejected,
    so a term printed on its own cannot be told apart from one that quietly asked
    something far broader. Every expansion is *checked*; only the ones that changed the
    question are printed."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    text = (bundle / SEARCH_FILENAME).read_text(encoding="utf-8")
    executed = run.state["executed"]
    assert executed, "the golden run has to have searched for this to mean anything"

    mechanical = 0
    for query in executed:
        assert query.term in text, query.term
        if _expansion_is_mechanical(query):
            mechanical += 1
        else:
            assert query.translation in text, query.translation
        if query.rationale:
            assert query.rationale in text, query.rationale

    if mechanical:
        assert "with each field tag written out in full" in text
    # Every query accounts for its expansion one way or the other. Collapsing a block is
    # allowed to shorten the answer, never to leave the question unanswered. Counted below
    # the queries heading, since the key above it names the same field.
    queries_section = text.split("## The queries", 1)[1]
    assert queries_section.count("**PubMed ran**") == sum(1 for q in executed if q.translation)


async def test_the_search_strategy_names_the_two_things_that_decide_a_replay(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap and the order it applies to. Without both, a query with more hits than
    the cap looks exactly like one taken whole, and the difference is the difference
    between a search that repeats and one that only mostly does."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    text = (bundle / SEARCH_FILENAME).read_text(encoding="utf-8")
    assert "relevance" in text
    assert str(POOL_SIZE) in text  # `per_query_retmax`, which the golden run leaves at 200


async def test_the_descriptor_and_the_root_index_both_point_at_the_search_strategy(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same reason as the predictor index: a file that is not a document is invisible to
    a consumer walking documents unless something at the root names it."""
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    payload = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert payload["search"] == SEARCH_FILENAME
    assert SEARCH_FILENAME in (bundle / INDEX_FILENAME).read_text(encoding="utf-8")


def _rendered(*executed: ExecutedQuery, retmax: int = 200) -> str:
    """`search.md` for a run that did nothing but these searches."""
    return search_markdown(
        charter_for(),
        RunManifest(run_id="test", queries=list(executed), retmax=retmax, sort="relevance"),
    )


def test_a_query_taken_whole_and_one_cut_off_by_the_cap_do_not_read_alike() -> None:
    """A reader comparing two runs needs to know which queries could have drifted. The
    uncapped ones cannot; saying so is most of the value of printing either."""
    text = _rendered(
        ExecutedQuery(term="whole[tiab]", count=12, retrieved=12),
        ExecutedQuery(term="cut[tiab]", count=900, retrieved=200),
    )
    assert "all 12 were retrieved" in text
    assert "The first 200 were retrieved" in text
    assert "the other 700 were never seen by this run" in text


def test_a_query_that_matched_nothing_says_so_rather_than_printing_a_zero() -> None:
    """PubMed reports an empty search as a successful one, so a bare `0` in a table is
    the one result a reader is most likely to skim past."""
    text = _rendered(ExecutedQuery(term="nothing[tiab]", count=0, retrieved=0))
    assert "no papers matched" in text
    assert "reports an empty search as a successful one" in text


def test_a_suspect_query_carries_its_verdict_and_pubmeds_own_report() -> None:
    """Two sources saying the same thing, kept separate on purpose: one is a conclusion
    drawn from comparing the term with its expansion, the other is a list the service
    returned. A reader deciding whether to trust the query wants both."""
    text = _rendered(
        ExecutedQuery(
            term="x[nosuchfield]",
            translation='"x"[All Fields]',
            count=9_000,
            retrieved=200,
            suspect=True,
            note="PubMed rewrote the field tag",
            fields_not_found=["nosuchfield"],
        )
    )
    assert "**Suspect.** PubMed rewrote the field tag" in text
    assert "`nosuchfield` as a field it does not have" in text


def test_an_expansion_that_only_spells_out_its_own_tags_is_not_printed_twice() -> None:
    """A term written in explicit field tags comes back as itself with `[tiab]` written
    `[Title/Abstract]`. Printing both doubles the file for no information, and a page of
    near-duplicate blocks is one a reader stops checking."""
    text = _rendered(
        ExecutedQuery(
            term='("a"[tiab] OR "b"[tiab]) AND eng[la]',
            translation='"a"[Title/Abstract] OR "b"[Title/Abstract] AND "english"[Language]',
            count=5,
            retrieved=5,
        )
    )
    assert "with each field tag written out in full" in text
    assert "[Title/Abstract]" not in text


def test_an_expansion_that_reached_for_a_field_the_term_never_asked_for_is_printed() -> None:
    """The case the block exists for, and the one the collapse must not swallow. An
    untagged word picks up `[MeSH Terms]` and an unknown tag becomes `[All Fields]`;
    either means PubMed answered a broader question than the one that was asked."""
    text = _rendered(
        ExecutedQuery(
            term="delirium AND eng[la]",
            translation='("delirium"[MeSH Terms] OR "delirium"[All Fields]) AND (english[Filter])',
            count=90_000,
            retrieved=200,
        )
    )
    assert '("delirium"[MeSH Terms] OR "delirium"[All Fields])' in text
    assert "with each field tag written out in full" not in text


def test_the_round_is_named_only_when_there_was_more_than_one() -> None:
    """A single-round run has no rounds to distinguish, and a `(round 1)` on every
    heading of one is noise that reads as though something were missing."""
    one = _rendered(ExecutedQuery(term="a[tiab]", count=1, retrieved=1, search_round=1))
    assert "round 1" not in one

    two = _rendered(
        ExecutedQuery(term="a[tiab]", count=1, retrieved=1, search_round=1),
        ExecutedQuery(term="b[tiab]", count=1, retrieved=1, search_round=2),
    )
    assert "(round 1)" in two
    assert "(round 2)" in two


def test_a_query_with_no_topic_is_said_to_be_for_the_whole_task() -> None:
    """The planner is allowed to write a query that fills no single topic, and a blank
    where a heading should be reads as a bug rather than as a deliberate breadth."""
    text = _rendered(ExecutedQuery(term="broad[tiab]", count=5, retrieved=5, topic=""))
    assert "Across the whole task" in text


# --- the catalog ------------------------------------------------------------


async def test_the_catalog_names_every_document_and_nothing_else(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    rows = [
        json.loads(line)
        for line in (bundle / CATALOG_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == TARGET

    for row in rows:
        assert (bundle / row["file"]).is_file()
        assert row["domain"] == Path(row["file"]).parent.name
        for key in (
            "pmid",
            "title",
            "domain",
            "description",
            "design",
            "n",
            "tags",
            "strength",
            "strength_score",
        ):
            assert key in row, key


# --- sign-off ---------------------------------------------------------------


class StubReviewer:
    """A sign-off surface with nobody behind it. Records what it was shown."""

    def __init__(self, signoff: Signoff) -> None:
        self.signoff = signoff
        self.shown: list[ConceptRecord] = []

    async def sign_off(
        self,
        records: Any,
        *,
        topics: dict[str, list[str]],
        verification: VerificationSummary | None,
        warnings: Any,
    ) -> Signoff:
        self.shown = list(records)
        return self.signoff


def attach(monkeypatch: pytest.MonkeyPatch, signoff: Signoff) -> StubReviewer:
    """Stand in for the console reviewer `--review` would have built."""
    stub = StubReviewer(signoff)

    def build(signer: str, console: Any = None, **_: Any) -> StubReviewer:
        return stub

    monkeypatch.setattr("okf_loremaster.ui.review.ConsoleReviewer", build)
    return stub


async def test_without_review_no_document_claims_to_have_been_verified(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default, and the honest one. `unverified` is the tier a machine extraction
    has earned, and the way OKF spells it is by the block not being there."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    assert run.state.get("verified_by", "") == ""
    for document in read_bundle(bundle).documents():
        assert "verified" not in document.fields
    # Nothing was asked, so nothing is reported as a shortfall.
    assert not [note for note in run.warnings if "sign-off" in note]


async def test_review_signs_every_document_with_a_named_human(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step 7 gate for `--review`: `verified: [{by: "human:<id>", ...}]`, on disk."""
    stub = attach(monkeypatch, Signoff.granted("human:tester"))
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch, review=True)

    assert len(stub.shown) == TARGET
    assert run.state["verified_by"] == "human:tester"

    for document in read_bundle(bundle).documents():
        entries = document.fields["verified"]
        assert isinstance(entries, list) and len(entries) == 1
        assert entries[0]["by"] == "human:tester"
        assert entries[0]["by"].startswith(HUMAN_PREFIX)
        assert entries[0]["at"].endswith("Z")

    # And it is still one key on one line, nested value and all.
    sample = next(iter(read_bundle(bundle).documents())).path
    block, _ = split(sample.read_text(encoding="utf-8"))
    verified = [line for line in block.splitlines() if line.startswith("verified:")]
    assert len(verified) == 1
    assert verified[0].endswith("}]")

    assert validate_bundle(bundle).ok


async def test_a_declined_sign_off_still_emits_the_bundle_unverified(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining is not a failure. Refusing to emit would punish the reviewer for
    reading carefully, and the files are simply written at the tier they earned."""
    attach(monkeypatch, Signoff.declined("the delta topic is thin"))
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch, review=True)

    assert run.state.get("verified_by", "") == ""
    documents = list(read_bundle(bundle).documents())
    assert len(documents) == TARGET
    for document in documents:
        assert "verified" not in document.fields

    declined = [note for note in run.warnings if "sign-off was not given" in note]
    assert len(declined) == 1
    assert "the delta topic is thin" in declined[0]
    assert validate_bundle(bundle).ok


# --- the manifest -----------------------------------------------------------


async def test_the_bundle_can_say_what_it_was_built_from_without_the_tool(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)
    manifest = run.state["manifest"]
    assert manifest is not None

    descriptor = yaml.safe_load((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))
    assert descriptor["kind"] == "okf"
    assert descriptor["id"] == run.state["run_id"]
    assert descriptor["documents"] == TARGET
    assert sorted(descriptor["domains"]) == sorted(TOPICS)
    assert descriptor["charter_digest"] == manifest.charter_digest
    # Advisory, not an expiry — but it has to be there, and it has to be later.
    assert manifest.stale_after is not None
    assert descriptor["stale_after"] == manifest.stale_after.isoformat()

    counts = manifest.counts
    assert counts.emitted == TARGET
    assert counts.unique >= counts.curated >= counts.emitted

    # The charter the descriptor points at is in the bundle, so the pointer resolves.
    assert (bundle / descriptor["charter"]).is_file()


async def test_the_log_never_reports_a_dollar_amount_for_an_unpriced_model(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fake/*` is in nobody's price map, and LiteLLM answers that with 0.0 rather than
    an error. A log saying `$0.00` would read as good news about a run that spent."""
    run, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    log = (bundle / LOG_FILENAME).read_text(encoding="utf-8")
    assert "cost unavailable" in log
    assert "$0.00" not in log
    manifest = run.state["manifest"]
    assert manifest is not None and manifest.cost.calls > 0


async def test_the_log_and_the_root_index_are_readable_documents_in_their_own_right(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = await golden(settings_factory, tmp_path, monkeypatch)

    for filename in (INDEX_FILENAME, LOG_FILENAME):
        fields, body = load((bundle / filename).read_text(encoding="utf-8"))
        assert fields["title"]
        assert fields["type"]
        assert body.strip()
