"""Durable, lossless run-telemetry store (SQLite).

Captures one row per gateway run plus its tool events, written at the run's
terminal status chokepoint from a per-run in-memory buffer that is filled at the
tool callback (UPSTREAM of the ephemeral SSE pop). This is a complete record —
it does not depend on a client draining the SSE stream in time.

Scalar columns only: tool name, duration, error flag, timestamps, token counts.
Never persist tool ``preview`` / ``args`` / output — those can carry secrets/PII.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, session_id TEXT, status TEXT,
  input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
  tool_count INTEGER, started_at REAL, ended_at REAL);
CREATE TABLE IF NOT EXISTS tool_events (
  run_id TEXT, seq INTEGER, tool TEXT, duration REAL, is_error INTEGER, started_at REAL,
  PRIMARY KEY (run_id, seq));
"""


def _hermes_home() -> Path:
    """HERMES_HOME, honoring the active profile (falls back to ~/.hermes)."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        import os
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def resolve_store_path(store_path: Optional[str] = None) -> Path:
    """Resolve the telemetry db path.

    A relative ``store_path`` (the config default ``"telemetry.db"``) resolves
    against HERMES_HOME; an absolute path is used as-is; ``None`` → the default.
    """
    if store_path:
        p = Path(store_path)
        return p if p.is_absolute() else _hermes_home() / store_path
    return _hermes_home() / "telemetry.db"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(conn, db_label="gateway.telemetry.db")
    except Exception:
        # WAL is an optimization; a store that can't set it still works.
        pass
    conn.executescript(_SCHEMA)
    return conn


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_run(
    run_row: Dict[str, Any],
    tool_event_rows: List[Dict[str, Any]],
    store_path: Optional[str] = None,
) -> None:
    """Persist one run row + its tool events in a single transaction.

    Idempotent on ``run_id`` (``INSERT OR REPLACE`` for the run; tool events for
    the run are replaced wholesale). Token fields default to 0 when ``usage`` was
    absent (failed/cancelled runs still persist with their ``tool_count``).
    """
    run_id = run_row.get("run_id")
    if not run_id:
        return
    rows = tool_event_rows or []
    conn = _connect(resolve_store_path(store_path))
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, session_id, status, input_tokens, output_tokens, "
                " total_tokens, tool_count, started_at, ended_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    run_row.get("session_id", ""),
                    run_row.get("status", ""),
                    _int(run_row.get("input_tokens")),
                    _int(run_row.get("output_tokens")),
                    _int(run_row.get("total_tokens")),
                    _int(run_row.get("tool_count")),
                    run_row.get("started_at"),
                    run_row.get("ended_at"),
                ),
            )
            conn.execute("DELETE FROM tool_events WHERE run_id = ?", (run_id,))
            conn.executemany(
                "INSERT OR REPLACE INTO tool_events "
                "(run_id, seq, tool, duration, is_error, started_at) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (
                        run_id,
                        seq,
                        ev.get("tool"),
                        ev.get("duration"),
                        1 if ev.get("is_error") else 0,
                        ev.get("started_at"),
                    )
                    for seq, ev in enumerate(rows)
                ],
            )
    finally:
        conn.close()
