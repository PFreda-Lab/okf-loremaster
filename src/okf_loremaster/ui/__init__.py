"""Event renderers. Nodes emit; these subscribe and display."""

from __future__ import annotations

from okf_loremaster.ui.plain import PlainRenderer, rich_enabled

__all__ = ["PlainRenderer", "rich_enabled"]
