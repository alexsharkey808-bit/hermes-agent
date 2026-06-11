"""Fidelity tests for the fork-owned query-aware summarizer overlay.

``tools/web_research_summarizer.summarize_for_query`` reproduces the former
``process_content_with_llm(..., query=query)`` behaviour. Since the web_research tool tests MOCK
this function, this file is the only place its INTERNAL control flow is exercised. No network:
the aux seam (``_resolve_web_extract_auxiliary`` / ``async_call_llm`` / ``extract_content_or_reasoning``)
is mocked where the module looks it up.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import tools.web_research_summarizer as s


def _mock_aux(monkeypatch, extract_return):
    """Wire the upstream seam so the summarizer runs offline; return the async_call_llm mock."""
    monkeypatch.setattr(s, "_resolve_web_extract_auxiliary", lambda model=None: (object(), "test-model", {}))
    call = AsyncMock(return_value="RESP")
    monkeypatch.setattr(s, "async_call_llm", call)
    monkeypatch.setattr(s, "extract_content_or_reasoning", MagicMock(return_value=extract_return))
    return call


def _user_prompts(call):
    """Return the user-message contents from every async_call_llm invocation."""
    out = []
    for c in call.call_args_list:
        for m in c.kwargs["messages"]:
            if m["role"] == "user":
                out.append(m["content"])
    return out


def test_query_in_user_prompt_and_returns_extract(monkeypatch):
    call = _mock_aux(monkeypatch, "EXTRACTED")
    content = "x" * 6000  # ≥ min_length (5000), < CHUNK_THRESHOLD → single pass

    result = asyncio.run(s.summarize_for_query(content, "my query", url="https://u", title="T"))

    assert result == "EXTRACTED"
    call.assert_awaited_once()
    user = _user_prompts(call)[0]
    assert "Research Query: my query" in user          # the query reaches the prompt
    assert "Extract all important information relevant to the query 'my query'." in user
    assert "Title: T" in user and "Source: https://u" in user  # context_str assembled


def test_short_content_returns_none(monkeypatch):
    call = _mock_aux(monkeypatch, "EXTRACTED")

    result = asyncio.run(s.summarize_for_query("tiny", "q"))

    assert result is None                 # below min_length → None (caller keeps raw page)
    call.assert_not_awaited()             # no LLM call for short content


def test_too_large_returns_placeholder(monkeypatch):
    call = _mock_aux(monkeypatch, "EXTRACTED")
    content = "x" * (s.MAX_CONTENT_SIZE + 1)  # > 2M chars

    result = asyncio.run(s.summarize_for_query(content, "q"))

    assert result.startswith("[Content too large to process:")
    call.assert_not_awaited()


def test_chunked_path_synthesizes_with_query(monkeypatch):
    call = _mock_aux(monkeypatch, "CHUNKSUM")
    content = "x" * (s.CHUNK_THRESHOLD + 50_000)  # > 500k, < 2M → chunk + synthesize

    result = asyncio.run(s.summarize_for_query(content, "my query"))

    assert result == "CHUNKSUM"           # synthesis output (capped — short here)
    # multiple chunk calls + one synthesis call
    assert call.await_count >= 2
    prompts = _user_prompts(call)
    # a chunk prompt used the query + the SECTION framing
    assert any("SECTION CONTENT:" in p and "Research Query: my query" in p for p in prompts)
    # the synthesis prompt referenced the query + the SECTION EXTRACTS framing
    assert any("SECTION EXTRACTS:" in p and "query: my query" in p for p in prompts)


def test_output_capped_at_max_output_size(monkeypatch):
    _mock_aux(monkeypatch, "y" * 6000)    # summarizer returns more than the cap
    content = "x" * 6000                   # single-pass

    result = asyncio.run(s.summarize_for_query(content, "q"))

    banner = "\n\n[... summary truncated for context management ...]"
    assert result.endswith(banner)
    assert result[: s.MAX_OUTPUT_SIZE] == "y" * s.MAX_OUTPUT_SIZE
    assert len(result) == s.MAX_OUTPUT_SIZE + len(banner)
