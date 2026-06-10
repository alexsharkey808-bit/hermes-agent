"""Tests for the trafilatura (free in-process page reader) extract provider.

Covers, with NO live network:
- capability flags (extract-only)
- happy-path extract → {url, title, content, raw_content, metadata}
- SSRF block on an internal input URL (per-hop is_safe_url gate)
- redirect-to-internal block (per-hop re-validation — the key guard, since the
  shared upstream SSRF check only covers the input URL)
- website-policy block via check_website_access()
- content clipping (token bound)
"""

from __future__ import annotations

import httpx
import pytest

from plugins.web.trafilatura.provider import (
    _MAX_CONTENT_CHARS,
    TrafilaturaWebSearchProvider,
)


class _FakeResp:
    def __init__(self, *, text="", status_code=200, headers=None, url="https://example.com/article"):
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.url = httpx.URL(url)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class _FakeClient:
    """Stand-in for httpx.Client used as a context manager."""

    def __init__(self, resp_for):
        # resp_for: a _FakeResp, or a callable(url) -> _FakeResp
        self._resp_for = resp_for

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def get(self, url):
        return self._resp_for(url) if callable(self._resp_for) else self._resp_for


@pytest.fixture(autouse=True)
def _no_lazy_install(monkeypatch):
    # trafilatura is installed in the venv; make the cold-start ensure a no-op
    # so the test never touches pip / network.
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None)


def _patch_guards(monkeypatch, *, safe=True, blocked=None):
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: safe)
    monkeypatch.setattr("tools.website_policy.check_website_access", lambda u: blocked)


def _patch_client(monkeypatch, resp_for):
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(resp_for))


def test_capability_flags():
    p = TrafilaturaWebSearchProvider()
    assert p.name == "trafilatura"
    assert p.supports_extract() is True
    assert p.supports_search() is False


def test_extract_happy_path(monkeypatch):
    _patch_guards(monkeypatch, safe=True, blocked=None)
    monkeypatch.setattr("trafilatura.extract", lambda html, **k: "Clean article text.")

    class _Meta:
        title = "My Title"
        author = "An Author"
        date = "2026-06-10"
        sitename = "Example"
        description = "A description"

    monkeypatch.setattr("trafilatura.extract_metadata", lambda html, **k: _Meta())
    _patch_client(monkeypatch, _FakeResp(text="<html>body</html>"))

    out = TrafilaturaWebSearchProvider().extract(["https://example.com/article"])
    assert len(out) == 1
    r = out[0]
    assert r["title"] == "My Title"
    assert r["content"] == "Clean article text."
    assert r["raw_content"] == "Clean article text."
    assert r["metadata"]["author"] == "An Author"
    assert "error" not in r


def test_extract_ssrf_block_on_input(monkeypatch):
    _patch_guards(monkeypatch, safe=False, blocked=None)
    _patch_client(monkeypatch, _FakeResp())  # never reached
    out = TrafilaturaWebSearchProvider().extract(["http://169.254.169.254/"])
    assert "private or internal" in out[0]["error"]


def test_extract_redirect_to_internal_blocked(monkeypatch):
    # External input is safe; the 302 target is an internal address. The
    # per-hop re-validation must block it (manual redirect following).
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda u: "169.254" not in u)
    monkeypatch.setattr("tools.website_policy.check_website_access", lambda u: None)
    redirect = _FakeResp(
        status_code=302,
        headers={"location": "http://169.254.169.254/latest/meta-data/"},
        url="https://example.com/start",
    )
    _patch_client(monkeypatch, lambda url: redirect)
    out = TrafilaturaWebSearchProvider().extract(["https://example.com/start"])
    assert "private or internal" in out[0]["error"]


def test_extract_website_policy_block(monkeypatch):
    blocked = {"message": "blocked by rule", "host": "evil.com", "rule": "deny", "source": "config"}
    _patch_guards(monkeypatch, safe=True, blocked=blocked)
    _patch_client(monkeypatch, _FakeResp(url="https://evil.com/x"))
    out = TrafilaturaWebSearchProvider().extract(["https://evil.com/x"])
    assert out[0]["error"] == "blocked by rule"
    assert out[0]["blocked_by_policy"]["host"] == "evil.com"


def test_extract_content_is_clipped(monkeypatch):
    _patch_guards(monkeypatch, safe=True, blocked=None)
    monkeypatch.setattr("trafilatura.extract", lambda html, **k: "x" * (_MAX_CONTENT_CHARS + 5000))
    monkeypatch.setattr("trafilatura.extract_metadata", lambda html, **k: None)
    _patch_client(monkeypatch, _FakeResp(text="<html/>"))
    out = TrafilaturaWebSearchProvider().extract(["https://example.com/long"])
    content = out[0]["content"]
    assert len(content) <= _MAX_CONTENT_CHARS + 32
    assert content.endswith("[truncated]")
