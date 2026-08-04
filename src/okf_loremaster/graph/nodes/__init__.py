"""One module per node. Every node is `async (state, deps) -> state update`.

Nodes never print and never raise for a recoverable problem: they emit events, add to
`warnings`, and return. The renderers decide what a person sees.
"""

from __future__ import annotations

from okf_loremaster.graph.nodes.charter import charter_node
from okf_loremaster.graph.nodes.curate import curate_node
from okf_loremaster.graph.nodes.dedupe import dedupe_node
from okf_loremaster.graph.nodes.emit_okf import emit_okf_node, manifest_for
from okf_loremaster.graph.nodes.extract import extract_node
from okf_loremaster.graph.nodes.fulltext import fulltext_node
from okf_loremaster.graph.nodes.index_vectors import index_vectors_node
from okf_loremaster.graph.nodes.rank import rank_node
from okf_loremaster.graph.nodes.reconcile import reconcile_node
from okf_loremaster.graph.nodes.review import review_node
from okf_loremaster.graph.nodes.screen import screen_node
from okf_loremaster.graph.nodes.search import pending_gap_plan, search_node
from okf_loremaster.graph.nodes.validate import validate_node

__all__ = [
    "charter_node",
    "curate_node",
    "dedupe_node",
    "emit_okf_node",
    "extract_node",
    "fulltext_node",
    "index_vectors_node",
    "manifest_for",
    "pending_gap_plan",
    "rank_node",
    "reconcile_node",
    "review_node",
    "screen_node",
    "search_node",
    "validate_node",
]
