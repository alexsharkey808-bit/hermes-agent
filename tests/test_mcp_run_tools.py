"""Offline, mocked tests for the run-driving MCP tools in mcp_serve.py.

No live network: the gateway HTTP layer (``_gateway_request``) and the SSE
stream (``urllib.request.urlopen``) are mocked. Tests target the module-level
``_run_*_impl`` functions that hold the real logic (the ``@mcp.tool()`` wrappers
delegate to them verbatim).
"""

import json
from unittest import mock

import mcp_serve


class _SSE:
    """Fake ``urlopen`` return value: a context manager yielding byte lines.

    Records how many lines were actually consumed so a test can prove that
    ``run_await`` breaks early on an approval event instead of draining.
    """

    def __init__(self, lines):
        self._lines = [l.encode() if isinstance(l, str) else l for l in lines]
        self.yielded = 0

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def __iter__(self):
        for line in self._lines:
            self.yielded += 1
            yield line


def _patch_stream(sse):
    """Patch the gateway base/headers/urlopen used by run_await's SSE read."""
    return (
        mock.patch.object(mcp_serve, "_gateway_base", return_value="http://x"),
        mock.patch.object(mcp_serve, "_gateway_headers", return_value={}),
        mock.patch("mcp_serve.urllib.request.urlopen", return_value=sse),
    )


# --------------------------------------------------------------------------
# run_start
# --------------------------------------------------------------------------

def test_run_start_posts_analysis_instructions():
    captured = {}

    def fake_req(method, path, body=None, timeout=30):
        captured.update(method=method, path=path, body=body)
        return {"run_id": "run_1", "session_id": "sess_1", "status": "running"}

    with mock.patch.object(mcp_serve, "_gateway_request", fake_req):
        out = json.loads(mcp_serve._run_start_impl("hello", act=False))

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/runs"
    assert captured["body"]["input"] == "hello"
    assert captured["body"]["instructions"] == mcp_serve._GATEWAY_ANALYSIS_INSTRUCTIONS
    assert "session_id" not in captured["body"]
    assert out == {"run_id": "run_1", "session_id": "sess_1", "status": "running"}


def test_run_start_act_sets_act_instructions_and_session():
    captured = {}

    def fake_req(method, path, body=None, timeout=30):
        captured.update(body=body)
        return {"run_id": "r", "session_id": "s", "status": "running"}

    with mock.patch.object(mcp_serve, "_gateway_request", fake_req):
        mcp_serve._run_start_impl("do it", act=True, session_id="sess_9")

    assert captured["body"]["instructions"] == mcp_serve._GATEWAY_ACT_INSTRUCTIONS
    assert captured["body"]["session_id"] == "sess_9"


# --------------------------------------------------------------------------
# run_await
# --------------------------------------------------------------------------

def test_run_await_drains_then_one_get():
    calls = []

    def fake_req(method, path, body=None, timeout=30):
        calls.append((method, path))
        return {"status": "completed", "output": "done", "usage": {"input_tokens": 5}}

    sse = _SSE([
        'data: {"event": "tool.started", "tool": "search"}\n',
        'data: {"event": "tool.completed", "tool": "search"}\n',
        ": stream closed\n",
    ])
    p_base, p_head, p_open = _patch_stream(sse)
    with mock.patch.object(mcp_serve, "_gateway_request", fake_req), p_base, p_head, p_open:
        out = json.loads(mcp_serve._run_await_impl("run_1"))

    assert calls == [("GET", "/v1/runs/run_1")]  # exactly ONE status fetch
    assert out["status"] == "completed"
    assert out["output"] == "done"
    assert out["usage"] == {"input_tokens": 5}
    assert "approval" not in out


def test_run_await_breaks_on_approval_request():
    calls = []

    def fake_req(method, path, body=None, timeout=30):
        calls.append((method, path))
        return {"status": "waiting_for_approval", "output": "", "usage": {}}

    sse = _SSE([
        'data: {"event": "tool.started", "tool": "terminal"}\n',
        'data: {"event": "approval.request", "approval_id": "appr_7", "summary": "run rm -rf"}\n',
        'data: {"event": "must.not.be.read"}\n',
    ])
    p_base, p_head, p_open = _patch_stream(sse)
    with mock.patch.object(mcp_serve, "_gateway_request", fake_req), p_base, p_head, p_open:
        out = json.loads(mcp_serve._run_await_impl("run_2"))

    assert calls == [("GET", "/v1/runs/run_2")]  # one GET, no tight-polling
    assert sse.yielded == 2  # broke at the approval line; 3rd line never read
    assert out["status"] == "waiting_for_approval"
    assert out["approval"]["approval_id"] == "appr_7"
    assert "run rm -rf" in out["approval"]["summary"]


def test_run_await_approval_loop_resolves_to_terminal():
    """waiting_for_approval -> run_approve('once') -> re-await reaches terminal."""
    # First await: pauses for approval.
    sse1 = _SSE([
        'data: {"event": "approval.request", "approval_id": "a1", "summary": "ok?"}\n',
    ])
    p_base, p_head, p_open = _patch_stream(sse1)
    with mock.patch.object(
        mcp_serve, "_gateway_request",
        return_value={"status": "waiting_for_approval", "output": "", "usage": {}},
    ), p_base, p_head, p_open:
        first = json.loads(mcp_serve._run_await_impl("run_x"))
    assert first["status"] == "waiting_for_approval"

    # Resolve it.
    appr = {}

    def fake_req(method, path, body=None, timeout=30):
        appr.update(method=method, path=path, body=body)
        return {"resolved": 1}

    with mock.patch.object(mcp_serve, "_gateway_request", fake_req):
        res = json.loads(mcp_serve._run_approve_impl("run_x", "once"))
    assert appr["method"] == "POST"
    assert appr["path"] == "/v1/runs/run_x/approval"
    assert appr["body"] == {"choice": "once"}
    assert res == {"resolved": 1}

    # Re-await now reaches a terminal state.
    sse2 = _SSE([": stream closed\n"])
    p_base, p_head, p_open = _patch_stream(sse2)
    with mock.patch.object(
        mcp_serve, "_gateway_request",
        return_value={"status": "completed", "output": "fin", "usage": {"input_tokens": 3}},
    ), p_base, p_head, p_open:
        final = json.loads(mcp_serve._run_await_impl("run_x"))
    assert final["status"] == "completed"
    assert final["output"] == "fin"


# --------------------------------------------------------------------------
# run_approve
# --------------------------------------------------------------------------

def test_run_approve_rejects_invalid_choice_without_calling_gateway():
    called = {"n": 0}

    def fake_req(*a, **k):
        called["n"] += 1
        return {}

    with mock.patch.object(mcp_serve, "_gateway_request", fake_req):
        out = json.loads(mcp_serve._run_approve_impl("run_3", "yolo"))

    assert "error" in out
    assert called["n"] == 0  # never hit the gateway on a bad choice


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------

def test_run_await_failed_run():
    sse = _SSE([": stream closed\n"])
    p_base, p_head, p_open = _patch_stream(sse)
    with mock.patch.object(
        mcp_serve, "_gateway_request",
        return_value={"status": "failed", "output": "", "usage": {}, "error": "boom"},
    ), p_base, p_head, p_open:
        out = json.loads(mcp_serve._run_await_impl("run_4"))

    assert out["status"] == "failed"
    assert "approval" not in out


def test_run_await_stream_error_still_fetches_status():
    """Stream closes/raises with no completion event -> single GET still runs."""
    boom = mock.MagicMock()
    boom.__enter__ = mock.MagicMock(side_effect=OSError("stream closed"))
    boom.__exit__ = mock.MagicMock(return_value=False)
    with mock.patch.object(
        mcp_serve, "_gateway_request",
        return_value={"status": "completed", "output": "ok", "usage": {"input_tokens": 1}},
    ), mock.patch.object(mcp_serve, "_gateway_base", return_value="http://x"), \
            mock.patch.object(mcp_serve, "_gateway_headers", return_value={}), \
            mock.patch("mcp_serve.urllib.request.urlopen", return_value=boom):
        out = json.loads(mcp_serve._run_await_impl("run_5"))

    assert out["status"] == "completed"


def test_run_await_partial_sse_frame_does_not_crash():
    sse = _SSE([
        'data: {"event": "tool.started"',  # split/partial JSON — must be skipped
        'data: {"event": "tool.completed", "tool": "x"}\n',
        ": stream closed\n",
    ])
    p_base, p_head, p_open = _patch_stream(sse)
    with mock.patch.object(
        mcp_serve, "_gateway_request",
        return_value={"status": "completed", "output": "", "usage": {}},
    ), p_base, p_head, p_open:
        out = json.loads(mcp_serve._run_await_impl("run_6"))

    assert out["status"] == "completed"  # malformed frame skipped, no crash


def test_run_status_single_get():
    calls = []

    def fake_req(method, path, body=None, timeout=30):
        calls.append((method, path))
        return {"run_id": "run_7", "status": "running", "usage": {}}

    with mock.patch.object(mcp_serve, "_gateway_request", fake_req):
        out = json.loads(mcp_serve._run_status_impl("run_7"))

    assert calls == [("GET", "/v1/runs/run_7")]
    assert out["status"] == "running"
