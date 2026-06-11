#!/usr/bin/env python3
"""web_research — gated parallel fan-out with refs-on-disk (off by default).

Searches once, fetches the top-k result pages CONCURRENTLY, writes each page body to a
per-run scratch dir under ``HERMES_HOME/research/.scratch/<run-id>/``, and returns only
compact refs (url + filename + byte count) to the model. The model then reads bodies
back on demand with ``search_files`` / ``read_file`` — so a wide fan-out costs a few
hundred ref-bytes of context instead of k full pages inline.

Off by default (``web.research_fanout_enabled``), gated by a ``check_fn``. Any failure
(search empty, every fetch failed, an egress/cap violation, a parse error) falls back
HARD to today's serial ``web_search`` + ``web_extract`` behavior, logged with the cause.
Enabling the flag is a downstream (measured) decision; Plan 1 ships this dormant.

Security boundary (the tool is fixed Python — the model authors no code): the egress
allow-list is built from THIS run's own search-result hostnames (we only ever fetch URLs
that search surfaced, never model-supplied), enforced per-run with ``Caps`` (call + byte
budget) from ``retrieval.search_primitives``. The SSRF + website-policy checks inside
``web_extract_tool`` remain the host-safety backstop.
"""

import asyncio
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from hermes_constants import get_hermes_home
from retrieval.search_primitives import Caps, EgressDenied
from tools.registry import registry
from tools.web_research_summarizer import summarize_for_query
from tools.web_tools import _load_web_config, web_extract_tool, web_search_tool

logger = logging.getLogger(__name__)

_SCRATCH_TTL_SECONDS = 24 * 3600


def _research_enabled() -> bool:
    """check_fn: expose the tool only when web.research_fanout_enabled is truthy."""
    return bool(_load_web_config().get("research_fanout_enabled", False))


class _Egress:
    """Per-run egress boundary — a hostname allow-list + a ``Caps`` (call/byte) budget.

    ``allow`` is built from the run's own search-result hostnames, so it is never
    empty-denies-all and never wide-open. State is per-instance (one ``_Egress`` per run).
    """

    def __init__(self, allow, caps: Caps):
        self._allow = set(allow)
        self._caps = caps
        self._calls = 0
        self._bytes = 0

    def check(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        if not any(host == a or host.endswith("." + a) for a in self._allow):
            raise EgressDenied(f"host not allowed: {host}")
        self._calls += 1
        if self._calls > self._caps.max_calls:
            raise EgressDenied("rate cap exceeded")

    def account(self, nbytes: int) -> None:
        self._bytes += nbytes
        if self._bytes > self._caps.max_bytes:
            raise EgressDenied("byte cap exceeded")


def _search_hits(query: str, k: int) -> list:
    """Run the sync ``web_search`` and return its result items (``data.web``)."""
    raw = web_search_tool(query, limit=k)
    return json.loads(raw).get("data", {}).get("web", []) or []


async def _fetch(url: str, egress: _Egress, scratch_dir: Path, query: str):
    """Fetch one URL, summarize it for the query, write the extract, return a ref.

    Runs the raw page through the cheap auxiliary side-model (``summarize_for_query``)
    with the search query, so the returned ref carries a query-relevant ``extract`` INLINE
    (the model answers directly instead of a multi-turn read_file loop). The extract — not the
    raw body — is what lands on disk.

    Returns ``None`` (skip + log) when the page yields an error/empty body. Raises
    ``EgressDenied`` when the boundary rejects the host or a per-run cap is exceeded.
    """
    egress.check(url)  # EgressDenied → propagates to the caller's fallback
    out = await web_extract_tool([url], use_llm_processing=False)
    try:
        result = json.loads(out)["results"][0]
    except (KeyError, IndexError, ValueError):
        logger.info("web_research: no extract result for %s", url)
        return None
    content = result.get("content") or ""
    if result.get("error") or not content:
        logger.info("web_research: skip %s (error=%r empty=%s)", url, result.get("error"), not content)
        return None
    nbytes = len(content.encode("utf-8"))
    egress.account(nbytes)  # egress budget = RAW fetched bytes (unchanged)
    title = result.get("title") or ""
    extract = await summarize_for_query(content, query, url=url, title=title)
    # summarize_for_query returns None (page < min_length, or no aux model), a real
    # extract, a truncated-raw+banner string (timeout — keep it), or a bracket placeholder
    # ("[Content too large…]" >2M chars / "[Failed to process…]" all-chunks-failed). Fall
    # back to a CAPPED raw body whenever there's no real extract or only a placeholder —
    # never let a bare placeholder masquerade as a query-relevant extract, never dump a
    # full page inline.
    if (not extract) or extract.startswith("[Content too large") or extract.startswith("[Failed to process"):
        extract = content[:5000]
    fname = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".txt"
    (scratch_dir / fname).write_text(extract, encoding="utf-8")
    return {"url": url, "file": fname, "bytes": len(extract.encode("utf-8")), "extract": extract}


async def _fanout(query: str, k: int, scratch_dir: Path) -> list:
    """Search once, fetch the hits concurrently, return the list of refs (no bodies).

    A per-URL error/empty page is skipped. An ``EgressDenied`` (host outside the run's
    search set, or a cap) aborts the fan-out and propagates so the handler falls back.
    """
    hits = _search_hits(query, k)
    urls = [h["url"] for h in hits if h.get("url")]
    allow = {urlparse(u).hostname for u in urls if urlparse(u).hostname}
    egress = _Egress(allow, Caps())
    results = await asyncio.gather(*[_fetch(u, egress, scratch_dir, query) for u in urls])
    return [r for r in results if r]


def _sweep_old_scratch() -> None:
    """Best-effort: drop scratch dirs older than 24h (full retention is a Plan-3 item)."""
    base = get_hermes_home() / "research" / ".scratch"
    if not base.exists():
        return
    cutoff = time.time() - _SCRATCH_TTL_SECONDS
    for d in base.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


async def _serial_fallback(query: str, k: int) -> str:
    """Today's behavior: search, then extract the top-k URLs inline (bodies returned)."""
    hits = _search_hits(query, k)
    urls = [h["url"] for h in hits if h.get("url")][:k]
    if urls:
        extract_out = await web_extract_tool(urls, use_llm_processing=False)
        results = json.loads(extract_out).get("results", [])
    else:
        results = []
    return json.dumps({"success": True, "fallback": True, "data": {"results": results}})


async def web_research_tool(query: str, k: int = 5) -> str:
    """Parallel web research with refs-on-disk; hard serial fallback on any failure."""
    if not _research_enabled():
        # Not normally reachable (check_fn gates exposure); degrade safely if dispatched.
        return await _serial_fallback(query, k)
    _sweep_old_scratch()
    run_id = uuid4().hex
    scratch_dir = get_hermes_home() / "research" / ".scratch" / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)
    try:
        refs = await _fanout(query, k, scratch_dir)
    except EgressDenied as e:
        logger.warning(
            "web_research: %s: %s — falling back to serial web_search/web_extract",
            type(e).__name__, e,
        )
        return await _serial_fallback(query, k)
    except Exception as e:
        logger.info(
            "web_research: fan-out failed (%s: %s) — falling back to serial web_search/web_extract",
            type(e).__name__, e,
        )
        return await _serial_fallback(query, k)
    if not refs:
        logger.info("web_research: 0 usable fetches — falling back to serial web_search/web_extract")
        return await _serial_fallback(query, k)
    return json.dumps({
        "success": True,
        "data": {"refs": refs, "scratch_dir": str(scratch_dir)},
        "hint": (
            "Query-relevant extracts are INLINE in each ref's 'extract' field — answer directly from them. "
            "The same extract text is saved at scratch_dir (NOT the full raw page). If an extract is "
            "insufficient, re-run web_research with a more specific query, or use web_search/web_extract directly."
        ),
    })


WEB_RESEARCH_SCHEMA = {
    "name": "web_research",
    "description": (
        "Research the web at breadth: run one search, fetch the top results CONCURRENTLY, "
        "summarize each page for your query via a fast side-model, and return a query-relevant "
        "EXTRACT inline per result — so you answer directly from the extracts in ~2 turns instead "
        "of reading pages back one by one. The same extract text is also saved to a scratch dir. "
        "Falls back to a normal inline search+extract if the fan-out cannot complete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to research."},
            "limit": {
                "type": "integer",
                "description": "How many top results to fetch concurrently (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

# NOTE: ``requires_env`` and ``max_result_size_chars`` are intentionally OMITTED (unlike
# web_extract): the ``check_fn`` flag is the gate, the configured search backend (ddgs) is
# keyless so there is no env requirement to surface, and refs are tiny so a result-size cap
# is moot. See reports/web-research-fanout.01.md for the rationale.
registry.register(
    name="web_research",
    toolset="web",
    schema=WEB_RESEARCH_SCHEMA,
    handler=lambda args, **kw: web_research_tool(
        args.get("query", ""), k=int(args.get("limit", 5) or 5)
    ),
    check_fn=_research_enabled,
    is_async=True,
    emoji="🔎",
)
