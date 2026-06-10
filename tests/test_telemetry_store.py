"""Tests for the lossless run-telemetry SQLite store (gateway/telemetry_store.py).

No network; uses a tmp db via the ``store_path`` argument.
"""

import sqlite3

from gateway import telemetry_store


def _db(tmp_path):
    return str(tmp_path / "telemetry.db")


def test_record_run_with_three_tool_events(tmp_path):
    db = _db(tmp_path)
    telemetry_store.record_run(
        {
            "run_id": "run_a", "session_id": "s1", "status": "completed",
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "tool_count": 3, "started_at": 1.0, "ended_at": 2.0,
        },
        [
            {"tool": "search", "duration": 0.5, "is_error": False, "started_at": 1.1},
            {"tool": "read_file", "duration": 0.2, "is_error": False, "started_at": 1.3},
            {"tool": "terminal", "duration": 1.0, "is_error": True, "started_at": 1.5},
        ],
        store_path=db,
    )
    conn = sqlite3.connect(db)
    try:
        tc = conn.execute("SELECT tool_count FROM runs WHERE run_id='run_a'").fetchone()[0]
        n = conn.execute("SELECT COUNT(*) FROM tool_events WHERE run_id='run_a'").fetchone()[0]
        err = conn.execute(
            "SELECT is_error FROM tool_events WHERE run_id='run_a' AND tool='terminal'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert tc == 3
    assert n == 3
    assert err == 1  # bool stored as integer


def test_record_run_without_usage_defaults_tokens_to_zero(tmp_path):
    db = _db(tmp_path)
    # A failed run carries no usage — token fields must persist as 0, not crash.
    telemetry_store.record_run(
        {"run_id": "run_b", "session_id": "s2", "status": "failed", "tool_count": 0},
        [],
        store_path=db,
    )
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT input_tokens, output_tokens, total_tokens, status "
            "FROM runs WHERE run_id='run_b'"
        ).fetchone()
    finally:
        conn.close()
    assert row == (0, 0, 0, "failed")


def test_record_run_is_idempotent_on_run_id(tmp_path):
    db = _db(tmp_path)
    row = {"run_id": "run_c", "status": "completed", "tool_count": 1}
    telemetry_store.record_run(row, [{"tool": "x"}], store_path=db)
    telemetry_store.record_run(row, [{"tool": "x"}], store_path=db)  # re-record
    conn = sqlite3.connect(db)
    try:
        runs = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id='run_c'").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM tool_events WHERE run_id='run_c'").fetchone()[0]
    finally:
        conn.close()
    assert runs == 1  # no duplicate run row
    assert events == 1  # tool events replaced, not appended


def test_schema_has_exactly_the_scalar_columns(tmp_path):
    db = _db(tmp_path)
    telemetry_store.record_run({"run_id": "run_d", "status": "completed"}, [], store_path=db)
    conn = sqlite3.connect(db)
    try:
        runs_cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
        te_cols = [r[1] for r in conn.execute("PRAGMA table_info(tool_events)")]
    finally:
        conn.close()
    assert runs_cols == [
        "run_id", "session_id", "status", "input_tokens", "output_tokens",
        "total_tokens", "tool_count", "started_at", "ended_at",
    ]
    assert te_cols == ["run_id", "seq", "tool", "duration", "is_error", "started_at"]
    # no preview/args/output columns leaked in
    assert "preview" not in te_cols and "args" not in te_cols and "output" not in te_cols
