"""Mocked-lifecycle tests for lossless run-telemetry capture in api_server.

Exercises the per-run buffer (filled at the tool callback) and the terminal
flush (`_flush_run_telemetry`) without a live gateway. telemetry_store.record_run
is mocked so nothing is written to disk.
"""

from unittest import mock

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _adapter():
    a = APIServerAdapter(PlatformConfig(enabled=True))
    a._telemetry_cfg_cache = {"enabled": True, "store_path": "telemetry.db"}
    return a


def test_callback_appends_completed_tool_to_buffer():
    a = _adapter()
    a._run_tool_buffers["r3"] = []
    cb = a._make_run_event_callback("r3", None)  # loop unused (no _run_streams entry)
    cb("tool.started", tool_name="search")
    cb("tool.completed", tool_name="search", duration=0.5, is_error=True)
    buf = a._run_tool_buffers["r3"]
    assert len(buf) == 1  # one entry per COMPLETED call (matches cw_hermes SSE count)
    assert buf[0]["tool"] == "search"
    assert buf[0]["duration"] == 0.5
    assert buf[0]["is_error"] is True
    assert "started_at" in buf[0]


def test_callback_noop_when_no_buffer():
    a = _adapter()
    cb = a._make_run_event_callback("r4", None)  # no buffer created (disabled)
    cb("tool.completed", tool_name="x", duration=0.1)  # must not raise
    assert "r4" not in a._run_tool_buffers


def test_flush_persists_tool_count_and_drops_buffer():
    a = _adapter()
    a._run_statuses["r1"] = {
        "status": "completed", "session_id": "s1",
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "created_at": 1.0, "updated_at": 2.0,
    }
    a._run_tool_buffers["r1"] = [{"tool": "a"}, {"tool": "b"}, {"tool": "c"}]
    with mock.patch("gateway.telemetry_store.record_run") as rec:
        a._flush_run_telemetry("r1")
    assert rec.call_count == 1
    run_row, tool_rows = rec.call_args.args[0], rec.call_args.args[1]
    assert run_row["tool_count"] == 3
    assert run_row["status"] == "completed"
    assert run_row["input_tokens"] == 10
    assert len(tool_rows) == 3
    assert "r1" not in a._run_tool_buffers  # buffer dropped, no leak


def test_flush_noop_when_telemetry_disabled():
    a = _adapter()
    a._run_statuses["r2"] = {"status": "completed"}
    # No buffer for r2 => telemetry was disabled for this run.
    with mock.patch("gateway.telemetry_store.record_run") as rec:
        a._flush_run_telemetry("r2")
    assert rec.call_count == 0


def test_flush_on_failed_run_defaults_tokens_and_drops_buffer():
    a = _adapter()
    a._run_statuses["r5"] = {"status": "failed", "session_id": "s5"}  # no usage
    a._run_tool_buffers["r5"] = [{"tool": "a"}]
    with mock.patch("gateway.telemetry_store.record_run") as rec:
        a._flush_run_telemetry("r5")
    assert rec.call_count == 1
    run_row = rec.call_args.args[0]
    assert run_row["status"] == "failed"
    assert run_row["input_tokens"] is None  # record_run coerces missing usage -> 0
    assert run_row["tool_count"] == 1
    assert "r5" not in a._run_tool_buffers


def test_flush_on_cancelled_drops_buffer():
    a = _adapter()
    a._run_statuses["r6"] = {"status": "cancelled"}
    a._run_tool_buffers["r6"] = [{"tool": "a"}, {"tool": "b"}]
    with mock.patch("gateway.telemetry_store.record_run") as rec:
        a._flush_run_telemetry("r6")
    assert rec.call_count == 1
    assert rec.call_args.args[0]["status"] == "cancelled"
    assert "r6" not in a._run_tool_buffers
