"""Scoring, MMR and the per-topic quota.

Ranking decides what the screener sees, which is the largest cost in a run, so the
properties worth pinning are the ones a refactor could quietly break: that the order is
total, that MMR trades relevance for spread rather than discarding relevance, and that
the quota reaches topics pure rank would never get to.
"""

from __future__ import annotations

from okf_loremaster.ranking import (
    UNASSIGNED,
    mmr_order,
    quota_select,
    relevance,
    score_all,
    selection_diff,
    similarity,
    tokens,
    topic_affinity,
)
from okf_loremaster.schemas import Candidate

NOW = 2026


def make(
    pmid: str,
    *,
    title: str = "a study of something",
    year: int | None = 2022,
    found_by: list[str] | None = None,
    best_rank: int = 0,
    citation_count: int = 10,
    rcr: float | None = 1.0,
    # Defaults to True because every other default here describes a paper iCite answered
    # about. A helper that handed out metrics and then said they were never measured
    # would be testing the outage path in every test that did not mean to.
    metrics_known: bool = True,
    publication_types: list[str] | None = None,
    mesh: list[str] | None = None,
) -> Candidate:
    return Candidate(
        pmid=pmid,
        title=title,
        abstract="an abstract",
        journal="J Test",
        year=year,
        found_by=found_by if found_by is not None else ["q1"],
        best_rank=best_rank,
        citation_count=citation_count,
        rcr=rcr,
        metrics_known=metrics_known,
        publication_types=publication_types or ["Journal Article"],
        mesh_terms=mesh or [],
    )


# --- relevance --------------------------------------------------------------


def test_agreement_across_queries_raises_the_score() -> None:
    one = make("1", found_by=["q1"])
    three = make("2", found_by=["q1", "q2", "q3"])
    assert relevance(three, now_year=NOW)[0] > relevance(one, now_year=NOW)[0]


def test_components_are_reported_and_bounded() -> None:
    _, parts = relevance(make("1"), now_year=NOW)
    assert set(parts) == {"agreement", "position", "recency", "citation", "abstract", "article"}
    assert all(0.0 <= value <= 1.0 for value in parts.values())


def test_a_comment_scores_below_a_journal_article() -> None:
    article = make("1", publication_types=["Journal Article"])
    comment = make("2", publication_types=["Comment"])
    assert relevance(comment, now_year=NOW)[0] < relevance(article, now_year=NOW)[0]


def test_a_missing_year_is_neither_rewarded_nor_buried() -> None:
    unknown = relevance(make("1", year=None), now_year=NOW)[1]["recency"]
    assert 0.0 < unknown < 1.0


# --- the citation signal, and its absence ------------------------------------
#
# These exist because adding `metrics_known` to `Candidate` broke nothing in this file.
# Every test above passed with the field defaulting to False — that is, with the citation
# component reading "unmeasured" for every paper — which showed that nothing here had
# ever asserted that the component reached a score at all.


def test_a_well_cited_paper_outranks_a_barely_cited_one() -> None:
    strong = make("1", rcr=4.0)
    weak = make("2", rcr=0.1)
    assert relevance(strong, now_year=NOW)[0] > relevance(weak, now_year=NOW)[0]


def test_citations_are_used_when_icite_gave_no_rcr() -> None:
    """RCR is field-normalized and better, but a raw count is still a signal."""
    many = make("1", rcr=None, citation_count=500)
    few = make("2", rcr=None, citation_count=1)
    assert relevance(many, now_year=NOW)[0] > relevance(few, now_year=NOW)[0]


def test_an_unanswered_paper_is_not_read_as_an_uncited_one() -> None:
    """`citation_count = 0` is equally "cited by nobody" and "never asked", and only
    `metrics_known` tells those apart. Read as the former, a service outage becomes a
    corpus of worthless papers."""
    unknown = relevance(make("1", metrics_known=False, citation_count=0, rcr=None), now_year=NOW)
    uncited = relevance(make("2", metrics_known=True, citation_count=0, rcr=None), now_year=NOW)
    assert unknown[1]["citation"] > uncited[1]["citation"]


def test_an_icite_outage_is_a_no_op_rather_than_a_second_recency_term() -> None:
    """The regression this was written for. With iCite down, every paper's citation count
    defaulted to zero, so the component scored 0.0 for the whole corpus except for papers
    too new to have been cited, which scored the neutral value instead — turning an
    unavailable signal into a bonus for being recent. Rolling-dated reference chapters
    floated into the top twenty of the pool on it.

    An unavailable signal has to be a constant across the corpus, because a constant
    added to every score changes no ordering at all.
    """
    down = [
        make("1", metrics_known=False, citation_count=0, rcr=None, year=2026),
        make("2", metrics_known=False, citation_count=0, rcr=None, year=2015),
        make("3", metrics_known=False, citation_count=0, rcr=None, year=None),
    ]
    components = {relevance(c, now_year=NOW)[1]["citation"] for c in down}
    assert len(components) == 1, "the citation component moved without any citation data"


# --- the E-utilities fallback ------------------------------------------------


def test_a_cited_by_count_restores_the_signal_an_outage_removed() -> None:
    """The point of the fallback, stated as an ordering rather than a field.

    With iCite down and nothing in its place, the previous test's constant is the best
    that can be done — correct, and worth nothing to a ranking. A raw count is not
    field-normalized and runs low, and it still separates a paper a hundred people cited
    from one nobody did.
    """
    much = Candidate(pmid="1", title="t", year=2015).with_citation_count(120)
    little = Candidate(pmid="2", title="t", year=2015).with_citation_count(0)

    assert (
        relevance(much, now_year=NOW)[1]["citation"]
        > (relevance(little, now_year=NOW)[1]["citation"])
    )


def test_a_count_without_an_rcr_does_not_pretend_to_be_field_normalized() -> None:
    """`rcr` stays None, so `_citation` takes the log-scaled branch and says so.

    Deriving a ratio from a raw count would be inventing the field baseline that makes
    RCR worth preferring in the first place.
    """
    candidate = Candidate(pmid="1", title="t", year=2015).with_citation_count(42)

    assert candidate.rcr is None
    assert candidate.citation_count == 42
    assert candidate.metrics_known, "measured, so ranking must not treat it as unknown"
    # And iCite's own classification is untouched: PubMed's publication types already
    # carry the part of that signal `_article` depends on.
    assert candidate.is_research_article


def test_a_paper_too_new_to_be_cited_is_still_not_scored_at_the_floor() -> None:
    """The guard that makes a fallback of raw counts safe.

    elink reports zero for a paper published last month exactly as it does for one
    nobody ever cited, and the two mean opposite things. `_TOO_NEW_YEARS` already
    separates them, which is why the fallback needs no rule of its own.
    """
    fresh = Candidate(pmid="1", title="t", year=NOW).with_citation_count(0)
    old = Candidate(pmid="2", title="t", year=2010).with_citation_count(0)

    assert (
        relevance(fresh, now_year=NOW)[1]["citation"]
        > (relevance(old, now_year=NOW)[1]["citation"])
    )


def test_order_is_total_so_two_runs_agree() -> None:
    # Identical in every scored respect, so only the tie-break separates them.
    twins = [make("200"), make("100"), make("300")]
    assert [item.pmid for item in score_all(twins, now_year=NOW)] == ["100", "200", "300"]


def test_position_is_assigned_after_sorting() -> None:
    scored = score_all([make("1", found_by=["a"]), make("2", found_by=["a", "b"])], now_year=NOW)
    assert [item.position for item in scored] == [0, 1]
    assert scored[0].pmid == "2"


# --- similarity -------------------------------------------------------------


def test_similarity_ignores_the_abstract() -> None:
    left = make("1", title="alpha beta")
    right = make("2", title="alpha beta")
    right.abstract = "a completely unrelated body of text about other things entirely"
    assert similarity(tokens(left), tokens(right)) == 1.0


def test_mesh_terms_count_toward_similarity() -> None:
    bare = make("1", title="alpha")
    tagged = make("2", title="alpha", mesh=["Cohort Studies"])
    assert tokens(tagged) > tokens(bare)
    assert 0.0 < similarity(tokens(bare), tokens(tagged)) < 1.0


# --- MMR --------------------------------------------------------------------


def test_mmr_keeps_the_top_ranked_item() -> None:
    scored = score_all(
        [make(str(i), title=f"topic {i}", found_by=["a"] * (5 - min(i, 4))) for i in range(5)],
        now_year=NOW,
    )
    assert mmr_order(scored, limit=3)[0].pmid == scored[0].pmid


def test_mmr_prefers_a_different_paper_to_a_near_duplicate() -> None:
    # Three papers: the top one, its clone, and a weaker but unrelated one.
    scored = score_all(
        [
            make("1", title="alpha beta gamma", found_by=["a", "b", "c"]),
            make("2", title="alpha beta gamma", found_by=["a", "b"]),
            make("3", title="delta epsilon zeta", found_by=["a"]),
        ],
        now_year=NOW,
    )
    assert [item.pmid for item in mmr_order(scored, limit=2)] == ["1", "3"]
    # Pure relevance would have taken the clone.
    assert [item.pmid for item in scored[:2]] == ["1", "2"]


def test_mmr_at_lambda_one_is_pure_relevance() -> None:
    scored = score_all([make(str(i), title=f"t{i}") for i in range(6)], now_year=NOW)
    assert [i.pmid for i in mmr_order(scored, limit=4, lambda_=1.0)] == [i.pmid for i in scored[:4]]


# --- quota ------------------------------------------------------------------


def test_affinity_is_blank_when_no_query_targeted_a_topic() -> None:
    assert topic_affinity(make("1", found_by=["q1"]), {}) == ""


def test_the_unassigned_group_is_named_in_the_comparison() -> None:
    scored = score_all([make("1"), make("2")], now_year=NOW)
    diff = selection_diff(scored, scored[:1], query_topic={})
    assert diff.pure_by_topic == {UNASSIGNED: 2}
    assert diff.diversified_by_topic == {UNASSIGNED: 1}


def test_affinity_follows_the_charter_order_not_the_search_order() -> None:
    # Found by the second topic's query first, but topic-a comes first in the mapping.
    query_topic = {"qa": "topic-a", "qb": "topic-b"}
    assert topic_affinity(make("1", found_by=["qb", "qa"]), query_topic) == "topic-a"


def test_quota_reaches_topics_pure_rank_never_would() -> None:
    query_topic = {"qa": "topic-a", "qb": "topic-b"}
    # Every topic-a paper outranks every topic-b paper: found by two queries, not one.
    corpus = [make(f"1{i:02d}", found_by=["qa", "qb"], best_rank=i) for i in range(10)]
    corpus += [make(f"2{i:02d}", found_by=["qb"], best_rank=20 + i) for i in range(10)]
    scored = score_all(corpus, now_year=NOW)

    pure = scored[:6]
    pool = quota_select(scored, query_topic=query_topic, pool_size=6)
    diff = selection_diff(pure, pool, query_topic=query_topic)

    assert diff.pure_by_topic.get("topic-b", 0) == 0
    assert diff.diversified_by_topic.get("topic-b", 0) > 0
    assert diff.changed > 0
    assert "topic-b" in diff.topics_helped


def test_quota_never_exceeds_the_pool_size() -> None:
    query_topic = {f"q{i}": f"topic-{i}" for i in range(4)}
    corpus = [
        make(f"{i}{n:02d}", found_by=[f"q{i}"], best_rank=n) for i in range(4) for n in range(20)
    ]
    scored = score_all(corpus, now_year=NOW)
    assert len(quota_select(scored, query_topic=query_topic, pool_size=17)) == 17


def test_a_pool_larger_than_the_corpus_keeps_everything() -> None:
    scored = score_all([make(str(i)) for i in range(5)], now_year=NOW)
    assert len(quota_select(scored, query_topic={}, pool_size=50)) == 5


def test_selection_diff_reports_no_change_when_there_is_none() -> None:
    scored = score_all([make(str(i)) for i in range(5)], now_year=NOW)
    diff = selection_diff(scored, scored, query_topic={})
    assert diff.changed == 0
    assert diff.added == [] and diff.dropped == []
