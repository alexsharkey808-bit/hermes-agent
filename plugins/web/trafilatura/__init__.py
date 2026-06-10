"""Trafilatura page-reader plugin — bundled, free extract backend.

Registers a keyless, in-process ``web_extract`` provider backed by the
``trafilatura`` + ``httpx`` packages. No API key required; ``is_available()``
gates on the packages being importable.
"""

from __future__ import annotations

from plugins.web.trafilatura.provider import TrafilaturaWebSearchProvider


def register(ctx) -> None:
    """Register the trafilatura extract provider with the plugin context."""
    ctx.register_web_search_provider(TrafilaturaWebSearchProvider())
