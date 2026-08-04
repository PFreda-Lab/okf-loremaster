"""Every typed object that moves between graph nodes.

Nothing in this package names a disease, a specialty, a drug, a lab, a registry, a
shelf, or a vocabulary key. Shelf slugs and vocabulary keys are strings decided by the
charter at runtime, which is what makes the same code serve any cohort.
`tests/test_domain_agnostic.py` scans `src/` and fails if that stops being true.
"""

from __future__ import annotations

from okf_loremaster.schemas.candidates import (
    Candidate,
    ExecutedQuery,
    PlannedQuery,
    QueryPlan,
    ScoredCandidate,
)
from okf_loremaster.schemas.charter import (
    DEFAULT_SHELF_MAX,
    DEFAULT_SHELF_MIN,
    DEFAULT_TARGET_PAPERS,
    MAX_SHELVES,
    Charter,
    Shelf,
)
from okf_loremaster.schemas.common import (
    Confidence,
    Direction,
    EvidenceType,
    Model,
    Slug,
    TextBasis,
    filename_token,
    is_export_safe,
    slugify,
)
from okf_loremaster.schemas.concept import (
    NONE_REPORTED,
    ConceptRecord,
    Extraction,
    NullFinding,
    PredictorRow,
    SourceRef,
    Verification,
    partition_vocabulary,
)
from okf_loremaster.schemas.evidence import (
    PaperText,
    VerificationSummary,
)
from okf_loremaster.schemas.limits import (
    MAX_BODY_WORDS,
    MAX_BOTTOM_LINE_SENTENCES,
    MAX_CAVEAT_SENTENCES,
    MAX_DESCRIPTION_CHARS,
    MAX_PREDICTOR_ROWS,
    MAX_SOURCE_CHARS,
    MAX_TAGS,
)
from okf_loremaster.schemas.manifest import (
    DEFAULT_FRESHNESS_DAYS,
    BundleCounts,
    CostSummary,
    RunManifest,
    ShelfSummary,
)
from okf_loremaster.schemas.parse import (
    SchemaError,
    parse_model,
    parse_model_with,
    response_format_for,
)
from okf_loremaster.schemas.screening import (
    CurationDecision,
    CurationResult,
    ScreenVerdict,
    ShelfCuration,
    ShelfGap,
)

__all__ = [
    "DEFAULT_FRESHNESS_DAYS",
    "DEFAULT_SHELF_MAX",
    "DEFAULT_SHELF_MIN",
    "DEFAULT_TARGET_PAPERS",
    "MAX_BODY_WORDS",
    "MAX_BOTTOM_LINE_SENTENCES",
    "MAX_CAVEAT_SENTENCES",
    "MAX_DESCRIPTION_CHARS",
    "MAX_PREDICTOR_ROWS",
    "MAX_SHELVES",
    "MAX_SOURCE_CHARS",
    "MAX_TAGS",
    "NONE_REPORTED",
    "BundleCounts",
    "Candidate",
    "Charter",
    "ConceptRecord",
    "Confidence",
    "CostSummary",
    "CurationDecision",
    "CurationResult",
    "Direction",
    "EvidenceType",
    "ExecutedQuery",
    "Extraction",
    "Model",
    "NullFinding",
    "PaperText",
    "PlannedQuery",
    "PredictorRow",
    "QueryPlan",
    "RunManifest",
    "SchemaError",
    "ScoredCandidate",
    "ScreenVerdict",
    "Shelf",
    "ShelfCuration",
    "ShelfGap",
    "ShelfSummary",
    "Slug",
    "SourceRef",
    "TextBasis",
    "Verification",
    "VerificationSummary",
    "filename_token",
    "is_export_safe",
    "parse_model",
    "parse_model_with",
    "partition_vocabulary",
    "response_format_for",
    "slugify",
]
