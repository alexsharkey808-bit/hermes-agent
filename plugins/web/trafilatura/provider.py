"""Free in-process page reader — ``trafilatura`` + ``httpx``.

An EXTRACT-only :class:`~agent.web_search_provider.WebSearchProvider` that
replaces the paid Firecrawl backend for the common "read this page" case.
Each URL is fetched with ``httpx`` (own per-URL timeout; redirects are
followed *manually* so every hop is re-validated against the SSRF +
website-policy guards) and the main article text is extracted with
``trafilatura``. No API key, no external service.

Response shape matches the extract contract documented in
:mod:`agent.web_search_provider` — a list of
``{url, title, content, raw_content, metadata}`` dicts, with ``error`` /
``blocked_by_policy`` set on a per-URL failure instead.

Note on guards: the shared ``web_extract_tool`` SSRF check
(``async_is_safe_url``) only validates the *input* URLs before dispatch.
Because this provider fetches directly from the Hermes host, a redirect
could otherwise reach an internal address — so we re-run ``is_safe_url``
*and* ``check_website_access`` on every redirect hop, mirroring Firecrawl's
post-redirect policy re-check but extended to SSRF.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Per-URL hard timeout (mirrors Firecrawl's 60s). Redirect + content caps
# keep the in-process reader bounded in time, memory, and tokens.
_TIMEOUT_SECONDS = 60.0
_MAX_REDIRECTS = 4
_MAX_CONTENT_CHARS = 4000  # clip extracted text (token-bounded)
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_USER_AGENT = (
    "Mozilla/5.0 (compatible; HermesBot/1.0; "
    "+https://github.com/NousResearch/hermes)"
)


class TrafilaturaWebSearchProvider(WebSearchProvider):
    """Keyless, in-process page-content extractor (trafilatura + httpx)."""

    @property
    def name(self) -> str:
        return "trafilatura"

    @property
    def display_name(self) -> str:
        return "Trafilatura (free reader)"

    def is_available(self) -> bool:
        """True when ``trafilatura`` and ``httpx`` import. No network I/O."""
        try:
            import httpx  # noqa: F401
            import trafilatura  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch + extract each URL. Sync — the dispatcher runs us in a thread."""
        # Cold-start install of the opt-in extra (matches firecrawl/exa).
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("search.trafilatura", prompt=False)
        except Exception as exc:  # noqa: BLE001 — lazy_deps surfaces install hints
            logger.debug("trafilatura lazy ensure: %s", exc)

        import httpx

        try:
            import trafilatura
        except ImportError:
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": "trafilatura is not installed — run `pip install trafilatura`",
                }
                for u in urls
            ]

        from tools.url_safety import is_safe_url
        from tools.website_policy import check_website_access

        results: List[Dict[str, Any]] = []
        with httpx.Client(
            timeout=_TIMEOUT_SECONDS,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            for url in urls:
                results.append(
                    self._extract_one(
                        url, client, trafilatura, is_safe_url, check_website_access
                    )
                )
        return results

    def _extract_one(
        self, url, client, trafilatura, is_safe_url, check_website_access
    ) -> Dict[str, Any]:
        current = url
        html = ""
        final_url = url
        try:
            for _hop in range(_MAX_REDIRECTS + 1):
                # Re-validate EVERY hop (SSRF + website policy). The shared
                # upstream SSRF check only covers the input URL.
                if not is_safe_url(current):
                    return {
                        "url": current,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": "Blocked: URL targets a private or internal network address",
                    }
                blocked = check_website_access(current)
                if blocked:
                    return {
                        "url": current,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": blocked["message"],
                        "blocked_by_policy": {
                            "host": blocked["host"],
                            "rule": blocked["rule"],
                            "source": blocked["source"],
                        },
                    }
                resp = client.get(current)
                if resp.status_code in _REDIRECT_CODES and "location" in resp.headers:
                    current = str(resp.url.join(resp.headers["location"]))
                    continue
                resp.raise_for_status()
                html = resp.text
                final_url = str(resp.url)
                break
            else:
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"Too many redirects (>{_MAX_REDIRECTS})",
                }
        except Exception as exc:  # noqa: BLE001 — surface fetch errors per-URL
            logger.debug("trafilatura fetch failed for %s: %s", url, exc)
            return {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": f"Fetch failed: {exc}",
            }

        content = ""
        title = ""
        metadata: Dict[str, Any] = {}
        try:
            content = (
                trafilatura.extract(html, url=final_url, include_comments=False)
                or ""
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("trafilatura.extract failed for %s: %s", final_url, exc)
        try:
            meta = trafilatura.extract_metadata(html, default_url=final_url)
            if meta is not None:
                title = getattr(meta, "title", "") or ""
                metadata = {
                    k: getattr(meta, k)
                    for k in ("title", "author", "date", "sitename", "description")
                    if getattr(meta, k, None)
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("trafilatura metadata failed for %s: %s", final_url, exc)

        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS] + "\n…[truncated]"

        if not content:
            return {
                "url": final_url,
                "title": title,
                "content": "",
                "raw_content": "",
                "error": "No extractable content (page may be JS-rendered or empty)",
                "metadata": metadata,
            }
        return {
            "url": final_url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": metadata,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Trafilatura (free reader)",
            "badge": "free · no key · extract only",
            "tag": (
                "In-process page reader (trafilatura + httpx) — no API key, "
                "replaces Firecrawl for reading article text"
            ),
            "env_vars": [],
        }
