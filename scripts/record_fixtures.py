#!/usr/bin/env python
"""Record HTTP fixtures for the offline test suite.

The only script in the repo that is expected to touch the network. Run it when an API
changes shape or a new case needs covering; the test suite itself replays what this
produces and never makes a live call.

    python scripts/record_fixtures.py --email you@example.org

The corpus topic is deliberately unrelated to the downstream consumer's domain, so that
a fixture can never quietly become the reason something looks domain-agnostic when it
is not.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from okf_loremaster.clients import build_clients  # noqa: E402
from okf_loremaster.clients.cassette import CassetteMode, CassetteTransport  # noqa: E402
from okf_loremaster.config import Settings  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
CASSETTE = FIXTURES / "ncbi.jsonl"

# One topic, deliberately not the downstream project's.
TERM = "postoperative respiratory failure[tiab] AND risk factors[tiab]"
# A field tag PubMed does not know. It answers with a silent rewrite to [All Fields]
# rather than an error, which is the behavior the search node has to defend against.
BAD_FIELD_TERM = "postoperative respiratory failure[nosuchfield]"

# Chosen for the awkward cases, not for their content:
#   9500320  - retracted (carries a RetractionIn pointer)
#   20301425 - PubmedBookArticle, a sibling element type, not a PubmedArticle
#   33745404 - journal article with no PMC id at all
PMIDS = ["9500320", "20301425", "33745404"]
# Carries a long cited-reference list, each entry with its own ArticleIdList. Its real
# PMC id is PMC6340782; an unscoped id lookup returns a reference's instead.
PMID_WITH_REFS = "30035690"
# In the open-access subset, so BioC returns real full text and a license.
PMCID_PRESENT = "PMC13424880"
# Not in the subset. Returns HTTP 200 with an `[Error]` body — the case that matters.
PMCID_MISSING = "PMC99999999"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default=None, help="NCBI contact address")
    parser.add_argument(
        "--append",
        action="store_true",
        help="add to the existing cassette instead of replacing it",
    )
    args = parser.parse_args()

    settings = Settings()
    email = args.email or settings.ncbi_email
    if not email:
        parser.error("no contact address: pass --email or set OKF_LOREMASTER_NCBI_EMAIL")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    if CASSETTE.exists() and not args.append:
        CASSETTE.unlink()

    # The cache is disabled on purpose: a cache hit records nothing.
    recording = settings.model_copy(update={"ncbi_email": email, "http_cache_enabled": False})
    transport = CassetteTransport(CASSETTE, CassetteMode.RECORD)
    clients = build_clients(recording, transport=transport)

    try:
        search = await clients.eutils.esearch(TERM, retmax=10)
        print(f"esearch      {search.count:>7} hits, {len(search.ids)} ids")
        print(f"             translation: {search.query_translation[:80]}...")

        bad = await clients.eutils.esearch(BAD_FIELD_TERM, retmax=5)
        print(f"esearch/bad  {bad.count:>7} hits for an unknown field tag")
        print(f"             fields_not_found={bad.fields_not_found!r}  <- note it is empty")

        # Fixed id lists, never search-derived: a cassette is keyed by request, so a
        # drifting id set would silently invalidate the fixtures on every re-record.
        records = await clients.eutils.efetch(PMIDS)
        records += await clients.eutils.efetch([PMID_WITH_REFS])
        print(f"efetch       {len(records)} record(s)")
        for record in records:
            flags = [record.source_type]
            if record.is_retracted:
                flags.append("RETRACTED")
            if record.pmcid:
                flags.append(record.pmcid)
            if not record.has_abstract:
                flags.append("no-abstract")
            print(
                f"             {record.pmid:>9}  {record.year}  "
                f"{record.title[:40]:<40} {' '.join(flags)}"
            )

        present = await clients.bioc.fetch(PMCID_PRESENT)
        if present is not None:
            print(
                f"bioc         {present.pmcid} license={present.license!r} "
                f"{len(present.sections)} sections, {present.word_count} words"
            )
        missing = await clients.bioc.fetch(PMCID_MISSING)
        print(f"bioc/missing {missing!r}  <- 200 OK with an [Error] body")

        annotated = await clients.pubtator.annotate(PMIDS)
        for pmid, doc in annotated.items():
            print(
                f"pubtator     {pmid:>9}  {len(doc.annotations):>3} annotations  "
                f"{doc.concept_types}"
            )

        metrics = await clients.icite.metrics([*PMIDS, PMID_WITH_REFS])
        print(f"icite        {len(metrics)} record(s)")
        for pmid, m in list(metrics.items())[:4]:
            print(
                f"             {pmid:>9}  cites={m.citation_count:<5} "
                f"rcr={m.relative_citation_ratio}  clinical={m.is_clinical}"
            )
    finally:
        await clients.aclose()

    print(f"\nwrote {transport.interactions} interaction(s) to {CASSETTE}")
    print(f"      {CASSETTE.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
