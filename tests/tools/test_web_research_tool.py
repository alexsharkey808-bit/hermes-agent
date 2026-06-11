"""Tests for the gated web_research tool (parallel fan-out + query-aware inline extracts).

No network: web_search_tool / web_extract_tool / summarize_for_query are faked. HERMES_HOME
is the per-test temp dir (the _isolate_hermes_home autouse fixture), so scratch dirs land in temp.

web_research summarizes each fetched page through the cheap aux model with the search query
(via the fork-owned ``tools.web_research_summarizer.summarize_for_query`` overlay) and returns the
query-relevant extract INLINE in each ref's "extract" field (the extract — not the raw page — is
what lands on disk). The mocks patch the names where web_research_tool LOOKS THEM UP
(`wr.summarize_for_query` / `wr.web_extract_tool`), not where they're defined.
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock

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
# async fanout + inline query-aware extracts
# --------------------------------------------------------------------------

def test_async_fanout_writes_extracts_and_returns_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)
    # short mocked bodies (<5000 chars) → summarize_for_query returns None (no network);
    # mock it explicitly for determinism → _fetch falls back to the (capped) raw body.
    monkeypatch.setattr(wr, "summarize_for_query", AsyncMock(return_value=None))
    scratch = tmp_path / "s"
    scratch.mkdir()

    refs = asyncio.run(wr._fanout("q", 3, scratch))

    # k extracts land on disk
    files = sorted(scratch.glob("*.txt"))
    assert len(files) == 3
    # refs now carry an inline extract alongside url / file / bytes
    assert len(refs) == 3
    for ref in refs:
        assert set(ref.keys()) == {"url", "file", "bytes", "extract"}
        assert ref["extract"]  # inline extract present (None-summary → raw body fallback)
        assert (scratch / ref["file"]).exists()
        assert (scratch / ref["file"]).read_text(encoding="utf-8") == ref["extract"]
        assert ref["bytes"] == len(ref["extract"].encode("utf-8"))

    # the egress boundary rejects a host NOT in the run's search-result set
    eg = wr._Egress({"ok.test"}, Caps())
    eg.check("https://ok.test/anything")  # in-set: allowed
    with pytest.raises(EgressDenied):
        eg.check("https://evil.test/x")


def test_query_and_title_forwarded_to_summarizer(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)  # title "t"
    summ = AsyncMock(return_value="EXTRACT")
    monkeypatch.setattr(wr, "summarize_for_query", summ)
    scratch = tmp_path / "s"
    scratch.mkdir()

    refs = asyncio.run(wr._fanout("my query", 1, scratch))

    assert len(refs) == 1
    summ.assert_awaited()
    args, kwargs = summ.call_args
    # _fetch calls summarize_for_query(content, query, url=url, title=title) — query is POSITIONAL.
    assert args[1] == "my query"                      # the search query is forwarded (positional)
    assert kwargs.get("title") == "t"                 # title comes from the web_extract result


def test_refs_carry_inline_extract(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)
    monkeypatch.setattr(wr, "summarize_for_query", AsyncMock(return_value="EXTRACT"))
    scratch = tmp_path / "s"
    scratch.mkdir()

    refs = asyncio.run(wr._fanout("q", 2, scratch))

    assert len(refs) == 2
    for ref in refs:
        assert ref["extract"] == "EXTRACT"
        assert ref["bytes"] == len("EXTRACT".encode("utf-8"))
        assert (scratch / ref["file"]).read_text(encoding="utf-8") == "EXTRACT"


def test_none_extract_falls_back_to_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)  # raw content "BODY-<url>"
    monkeypatch.setattr(wr, "summarize_for_query", AsyncMock(return_value=None))
    scratch = tmp_path / "s"
    scratch.mkdir()

    refs = asyncio.run(wr._fanout("q", 1, scratch))

    ref = refs[0]
    assert "extract" in ref
    assert ref["extract"].startswith("BODY-")  # raw mocked body (capped)
    assert (scratch / ref["file"]).read_text(encoding="utf-8") == ref["extract"]


def test_placeholder_extract_falls_back_to_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_ok)
    # the >2M-char placeholder must NOT be stored as a "query-relevant extract"
    placeholder = "[Content too large to process: 3.0MB. Try a more focused source URL.]"
    monkeypatch.setattr(wr, "summarize_for_query", AsyncMock(return_value=placeholder))
    scratch = tmp_path / "s"
    scratch.mkdir()

    refs = asyncio.run(wr._fanout("q", 1, scratch))

    ref = refs[0]
    assert not ref["extract"].startswith("[Content too large")
    assert ref["extract"].startswith("BODY-")  # fell back to the raw body


def test_skips_error_and_empty_results(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "web_search_tool", _fake_search)
    monkeypatch.setattr(wr, "web_extract_tool", _fake_extract_all_fail)
    # summarize_for_query must NOT be reached for error/empty pages (skipped earlier)
    summ = AsyncMock(return_value="EXTRACT")
    monkeypatch.setattr(wr, "summarize_for_query", summ)
    scratch = tmp_path / "s"
    scratch.mkdir()

    refs = asyncio.run(wr._fanout("q", 3, scratch))

    assert refs == []                 # every error/empty result is skipped → one ref per usable URL
    summ.assert_not_awaited()


# --------------------------------------------------------------------------
# tool gate / run-id scratch / serial fallback
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
    monkeypatch.setattr(wr, "summarize_for_query", AsyncMock(return_value="EXTRACT"))

    out1 = json.loads(asyncio.run(wr.web_research_tool("q", k=3)))
    out2 = json.loads(asyncio.run(wr.web_research_tool("q", k=3)))

    assert out1["success"] is True
    refs = out1["data"]["refs"]
    assert len(refs) == 3
    assert all(set(r) == {"url", "file", "bytes", "extract"} for r in refs)
    assert all(r["extract"] == "EXTRACT" for r in refs)  # query-relevant extract inline

    base = get_hermes_home() / "research" / ".scratch"
    d1, d2 = out1["data"]["scratch_dir"], out2["data"]["scratch_dir"]
    assert d1 != d2  # unique per call — concurrent runs cannot collide
    assert str(base) in d1 and str(base) in d2
    # extracts actually exist under the unique run dir
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
