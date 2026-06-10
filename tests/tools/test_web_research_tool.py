"""Tests for the gated web_research tool (parallel fan-out + refs-on-disk).

No network: web_search_tool / web_extract_tool are faked. HERMES_HOME is the per-test
temp dir (the _isolate_hermes_home autouse fixture), so scratch dirs land in temp.
"""

import asyncio
import json
import logging

import pytest

from retrieval.search_primitives import Caps, EgressDenied
from tools.registry import invalidate_check_fn_cache, registry
import tools.web_research_tool as wr


def _fake_search(query, limit=5):
    return json.dumps({
        "success": True,
        "data": {"web": [
            {"title": f"r{i}", "url": f"https://ok.test/{i}", "description": "d", "position": i}
            for i in range(limit)
        ]},
    })


async def _fake_extract_ok(urls, use_llm_processing=False, **kw):
    url = urls[0]
    return json.dumps({"results": [{"url": url, "title": "t", "content": f"BODY-{url}"}]})


async def _fake_extract_all_fail(urls, use_llm_processing=False, **kw):
    url = urls[0]
    return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": "boom"}]})


def _set_flag(monkeypatch, enabled):
    monkeypatch.setattr(wr, "_load_web_config", lambda: {"research_fanout_enabled": enabled})
    invalidate_check_fn_cache()


# --------------------------------------------------------------------------
# Task 3 — async fanout + adapter
# --------------------------------------------------------------------------

def test_async_fanout_writes_bodies_and_returns_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)
    scratch = tmp_path / "s"
    scratch.mkdir()

    refs = asyncio.run(wr._fanout("q", 3, scratch))

    # k bodies land on disk
    files = sorted(scratch.glob("*.txt"))
    assert len(files) == 3
    # refs carry no inline body text — only url / file / bytes
    assert len(refs) == 3
    for ref in refs:
        assert set(ref.keys()) == {"url", "file", "bytes"}
        assert "BODY" not in json.dumps(ref)
        assert (scratch / ref["file"]).exists()
        assert ref["bytes"] > 0

    # the egress boundary rejects a host NOT in the run's search-result set
    eg = wr._Egress({"ok.test"}, Caps())
    eg.check("https://ok.test/anything")  # in-set: allowed
    with pytest.raises(EgressDenied):
        eg.check("https://evil.test/x")


# --------------------------------------------------------------------------
# Task 4 — tool gate / run-id scratch / serial fallback
# --------------------------------------------------------------------------

def test_flag_off_hides_tool(monkeypatch):
    _set_flag(monkeypatch, False)
    assert registry.get_definitions({"web_research"}) == []
    _set_flag(monkeypatch, True)
    defs = registry.get_definitions({"web_research"})
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "web_research"


def test_returns_refs_when_enabled(monkeypatch):
    from hermes_constants import get_hermes_home

    _set_flag(monkeypatch, True)
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)

    out1 = json.loads(asyncio.run(wr.web_research_tool("q", k=3)))
    out2 = json.loads(asyncio.run(wr.web_research_tool("q", k=3)))

    assert out1["success"] is True
    refs = out1["data"]["refs"]
    assert len(refs) == 3
    assert all(set(r) == {"url", "file", "bytes"} for r in refs)
    assert "BODY" not in json.dumps(refs)  # no inline body in the returned payload

    base = get_hermes_home() / "research" / ".scratch"
    d1, d2 = out1["data"]["scratch_dir"], out2["data"]["scratch_dir"]
    assert d1 != d2  # unique per call — concurrent runs cannot collide
    assert str(base) in d1 and str(base) in d2
    # bodies actually exist under the unique run dir
    assert sorted(__import__("pathlib").Path(d1).glob("*.txt"))


def test_all_fetches_fail_falls_back(monkeypatch, caplog):
    _set_flag(monkeypatch, True)
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_all_fail)

    with caplog.at_level(logging.INFO, logger="tools.web_research_tool"):
        out = json.loads(asyncio.run(wr.web_research_tool("q", k=3)))

    # degraded serial result, no exception escaped (we got a JSON string)
    assert out.get("fallback") is True
    assert "falling back to serial" in caplog.text.lower()


def test_egress_denied_is_logged_distinctly(monkeypatch, caplog):
    _set_flag(monkeypatch, True)
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)

    async def _boom(query, k, scratch_dir):
        raise EgressDenied("host not allowed: evil.test")

    monkeypatch.setattr(wr, "_fanout", _boom)

    with caplog.at_level(logging.WARNING, logger="tools.web_research_tool"):
        out = json.loads(asyncio.run(wr.web_research_tool("q", k=3)))

    assert out.get("fallback") is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # distinct: the WARNING names both the exception TYPE and the offending host
    assert any(
        "EgressDenied" in r.getMessage() and "evil.test" in r.getMessage()
        for r in warnings
    )
