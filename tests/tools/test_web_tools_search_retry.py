"""Tests for the opt-in web_search retry/backoff (web.search_retries).

No live network — uses a flaky stub provider and monkeypatched config.
"""

from __future__ import annotations

import pytest

import tools.web_tools as wt


class _FlakyProvider:
    name = "flaky"

    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def search(self, query, limit):
        self.calls += 1
        if self.calls <= self.fail_times:
            return {"success": False, "error": "rate limited"}
        return {"success": True, "data": {"web": [{"title": "ok", "url": "u", "description": "", "position": 1}]}}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # don't actually back off during tests
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)


def _set_retries(monkeypatch, n):
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_retries": n})


def test_get_search_retries_reads_config(monkeypatch):
    _set_retries(monkeypatch, 3)
    assert wt._get_search_retries() == 3


def test_get_search_retries_clamped(monkeypatch):
    _set_retries(monkeypatch, 99)
    assert wt._get_search_retries() == 5


def test_get_search_retries_default_off(monkeypatch):
    monkeypatch.setattr(wt, "_load_web_config", lambda: {})
    assert wt._get_search_retries() == 0


def test_retry_recovers_on_transient_failure(monkeypatch):
    _set_retries(monkeypatch, 2)
    p = _FlakyProvider(fail_times=2)
    out = wt._search_with_retry(p, "q", 5)
    assert p.calls == 3
    assert out["success"] is True


def test_gate_off_is_single_call_explicit_error(monkeypatch):
    _set_retries(monkeypatch, 0)
    p = _FlakyProvider(fail_times=2)
    out = wt._search_with_retry(p, "q", 5)
    assert p.calls == 1
    assert out["success"] is False


def test_retries_exhausted_returns_failure(monkeypatch):
    _set_retries(monkeypatch, 2)
    p = _FlakyProvider(fail_times=5)
    out = wt._search_with_retry(p, "q", 5)
    assert p.calls == 3
    assert out["success"] is False


def test_empty_results_trigger_retry(monkeypatch):
    # success=True but zero web hits also counts as "needs retry"
    _set_retries(monkeypatch, 1)

    class _EmptyThenFull:
        name = "empty"

        def __init__(self):
            self.calls = 0

        def search(self, query, limit):
            self.calls += 1
            if self.calls == 1:
                return {"success": True, "data": {"web": []}}
            return {"success": True, "data": {"web": [{"title": "x", "url": "u", "description": "", "position": 1}]}}

    p = _EmptyThenFull()
    out = wt._search_with_retry(p, "q", 5)
    assert p.calls == 2
    assert out["data"]["web"]
