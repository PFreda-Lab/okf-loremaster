"""The domain-agnosticism invariant, checked mechanically.

`src/` is biomedical by construction — PubMed, MeSH, `[tiab]`, BioC, PubTator and a
biomedical embedding model all belong there. What must never appear is anything *below*
that level: a named disease, drug, drug class, lab, specialty, cohort, registry, topic,
or a fixed list of vocabulary keys. Those are decided by the charter at runtime, which
is the whole reason the same code serves any cohort.

This scan is easy to "fix" into breaking the search node, so the infrastructure terms
that must **not** be flagged are asserted explicitly below. If a future edit widens the
blocklist until `mesh` or `pubmed` trips it, `test_infrastructure_terms_are_not_flagged`
fails first and says so.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from okf_loremaster.schemas import Charter, CodedAs, VocabularyHint

SRC = Path(__file__).resolve().parents[1] / "src"

# Grouped by the kind of mistake, because the fix differs: a disease name means the
# charter should have supplied it, while a topic slug means someone copied the bundle
# this tool replaces.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "disease or condition": (
        "hiv",
        "aids",
        "diabetes",
        "diabetic",
        "sepsis",
        "septic shock",
        "cancer",
        "carcinoma",
        "asthma",
        "copd",
        "stroke",
        "myocardial infarction",
        "heart failure",
        "covid",
        "sars-cov-2",
        "influenza",
        "tuberculosis",
        "malaria",
        "hepatitis",
        "hypertension",
        "alzheimer",
        "parkinson",
        "obesity",
        "pneumonia",
        "delirium",
        "preeclampsia",
    ),
    "drug or drug class": (
        "metformin",
        "aspirin",
        "statin",
        "statins",
        "insulin",
        "warfarin",
        "heparin",
        "opioid",
        "opioids",
        "antiretroviral",
        "tenofovir",
        "ssri",
        "beta-blocker",
        "ace inhibitor",
        "chemotherapy",
        "anticoagulant",
    ),
    "lab or biomarker": (
        "cd4",
        "hba1c",
        "creatinine",
        "troponin",
        "hemoglobin",
        "bilirubin",
        "viral load",
        "egfr",
        "ldl",
        "hdl",
    ),
    "specialty": (
        "cardiology",
        "oncology",
        "nephrology",
        "hepatology",
        "endocrinology",
        "neurology",
        "psychiatry",
        "pediatrics",
        "obstetrics",
        "infectious disease",
    ),
    "cohort or registry": (
        "nhanes",
        "mimic-iv",
        "uk biobank",
        "ukbiobank",
        "framingham",
        "seer",
    ),
    # The topics of the hand-built bundle this tool replaces. A slug from that bundle
    # in `src/` means a taxonomy got hardcoded instead of derived.
    "topic slug": (
        "art-pharmacology-adherence",
        "comorbidities-coinfections",
        "immunology-virology",
        "labs-biomarkers",
        "mental-health-substance-use",
        "social-determinants",
    ),
}

# Biomedical infrastructure. In scope, and in `src/` on purpose. Listed here so that a
# future widening of the blocklist fails this file rather than the search node.
INFRASTRUCTURE = (
    "pubmed",
    "pmc",
    "pubmedbert",
    "mesh",
    "tiab",
    "majr",
    "eutils",
    "esearch",
    "efetch",
    "bioc",
    "pubtator",
    "icite",
    "entrez",
    "icd10",
    "icd-10",
    "atc",
    "loinc",
    "snomed",
    "rxnorm",
    "cpt",
    "doi",
    "orcid",
    "chroma",
    "biomedical",
    "clinical",
    "cohort",
    "outcome",
    "predictor",
    "biomarker",
)

_MATCHERS = {
    term: re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
    for terms in FORBIDDEN.values()
    for term in terms
}


def flagged_terms(text: str) -> set[str]:
    """Blocklisted terms appearing in `text`, matched whole rather than as substrings.

    Whole-token matching is what keeps `pubmedbert` from tripping on nothing and
    `keystroke` from tripping on `stroke`.
    """
    lowered = text.lower()
    return {term for term, matcher in _MATCHERS.items() if matcher.search(lowered)}


def source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


# --- the scan ---------------------------------------------------------------


def test_the_scan_actually_has_files_to_scan() -> None:
    """A glob that silently finds nothing would make every check below vacuous."""
    files = source_files()
    assert len(files) > 10
    assert any(path.name == "config.py" for path in files)


@pytest.mark.parametrize("path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_clinical_specific_appears_in_source(path: Path) -> None:
    offenders: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for term in sorted(flagged_terms(line)):
            category = next(k for k, v in FORBIDDEN.items() if term in v)
            offenders.append(f"{path.name}:{number} {term!r} ({category})")

    assert not offenders, (
        "src/ must not name a clinical specific — it is derived from the charter at "
        "runtime, and a constant here would be wrong for the next project:\n"
        + "\n".join(offenders)
    )


# --- the guardrails on the scan itself --------------------------------------


def test_infrastructure_terms_are_not_flagged() -> None:
    """The negative cases. Blocking these would break the search node, not protect it.

    Kept explicit so that widening the blocklist fails here — loudly, and with the
    reason attached — rather than in a node that quietly stops finding papers.
    """
    prose = (
        "Search PubMed via E-utilities esearch/efetch, refine with MeSH and [tiab] and "
        "[majr] terms, fetch full text from PMC through BioC, annotate with PubTator, "
        "score with iCite, and embed with a PubMedBERT model. Vocabulary keys such as "
        "icd10, atc, loinc, snomed, rxnorm and cpt are charter-supplied examples, not "
        "constants. A clinical cohort study reports a biomedical predictor and an "
        "outcome; a biomarker is one kind of predictor."
    )
    assert flagged_terms(prose) == set()

    for term in INFRASTRUCTURE:
        assert flagged_terms(term) == set(), f"{term!r} is infrastructure and must pass"


def test_the_scan_would_actually_catch_something() -> None:
    """A positive control. Without it, a broken matcher reads as a clean codebase."""
    assert "hiv" in flagged_terms('TOPICS = ["hiv-labs"]  # HIV cohort')
    assert "metformin" in flagged_terms("if drug == 'Metformin':")
    assert "labs-biomarkers" in flagged_terms('domain = "labs-biomarkers"')
    assert "cd4" in flagged_terms("the CD4 count")


def test_whole_token_matching_does_not_fire_on_substrings() -> None:
    assert flagged_terms("keystroke handling") == set()
    assert flagged_terms("pubmedbert embeddings") == set()
    assert flagged_terms("the raids on the cache") == set()


# --- the invariant behind the scan ------------------------------------------


def test_the_charter_decides_no_taxonomy_in_advance() -> None:
    """A baked-in default would work on the project it was written for and no other."""
    assert Charter(prompt="anything").topic_taxonomy == []


def test_any_coding_system_a_paper_used_can_be_recorded() -> None:
    """There is no allowlist to be absent from.

    The package holds no list of approved standards, so a hint records whatever the
    paper printed — including a registry-specific system nobody anticipated.
    """
    hint = VocabularyHint(
        concept="whatever the paper called it",
        codes=[CodedAs(system="some-local-registry-v3", code="X-1")],
    )
    assert hint.codes[0].system == "somelocalregistryv3"
