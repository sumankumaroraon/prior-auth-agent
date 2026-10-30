"""Execution log and case store (SQLite).

* events : one row per node run, tool call, LLM call and human action, with timing
* cases  : one row per request, with route, SLA due time, status and final decision

Member IDs are stored as a salted SHA-256 hash, so the log can link duplicate
submissions without holding the raw identifier (PHI minimization).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

SALT = "pa-agent-demo"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  request_id TEXT PRIMARY KEY, thread_id TEXT, received_at TEXT, channel TEXT,
  member_hash TEXT, item_key TEXT, item_name TEXT, request_type TEXT, urgency TEXT,
  route TEXT, status TEXT, sla_due_at TEXT, processed_at TEXT, decided_at TEXT,
  decided_by TEXT, decision TEXT, reasons TEXT, flags TEXT, processing_ms REAL, mode TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, request_id TEXT, node TEXT,
  event_type TEXT, payload TEXT, duration_ms REAL
);
CREATE INDEX IF NOT EXISTS ix_events_req ON events(request_id);
CREATE INDEX IF NOT EXISTS ix_cases_member ON cases(member_hash, item_key);
"""


def member_hash(member_id: str | None) -> str | None:
    if not member_id:
        return None
    return hashlib.sha256(f"{SALT}:{member_id.strip().upper()}".encode()).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            yield con
            con.commit()
        finally:
            con.close()

    # ---------------------------------------------------------------- events
    def log(self, request_id: str, node: str, event_type: str, payload: Any = None, duration_ms: float | None = None) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO events(ts, request_id, node, event_type, payload, duration_ms) VALUES (?,?,?,?,?,?)",
                      (now_iso(), request_id, node, event_type, json.dumps(payload, default=str), duration_ms))

    def events(self, request_id: str | None = None) -> list[dict]:
        with self._conn() as c:
            q = "SELECT * FROM events" + (" WHERE request_id=?" if request_id else "") + " ORDER BY id"
            rows = c.execute(q, (request_id,) if request_id else ()).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- cases
    def upsert_case(self, request_id: str, **fields: Any) -> None:
        for k in ("reasons", "flags"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO cases(request_id) VALUES (?)", (request_id,))
            if fields:
                sets = ", ".join(f"{k}=?" for k in fields)
                c.execute(f"UPDATE cases SET {sets} WHERE request_id=?", (*fields.values(), request_id))

    def case(self, request_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM cases WHERE request_id=?", (request_id,)).fetchone()
        return dict(r) if r else None

    def cases(self, status: str | None = None) -> list[dict]:
        with self._conn() as c:
            q = "SELECT * FROM cases" + (" WHERE status=?" if status else "") + " ORDER BY received_at"
            rows = c.execute(q, (status,) if status else ()).fetchall()
        return [dict(r) for r in rows]

    def find_duplicate(self, request_id: str, member_id: str, item_key: str, received_at: str, window_days: int = 30) -> str | None:
        mh = member_hash(member_id)
        start = (datetime.fromisoformat(received_at) - timedelta(days=window_days)).isoformat()
        with self._conn() as c:
            r = c.execute(
                "SELECT request_id FROM cases WHERE member_hash=? AND item_key=? AND request_id<>? "
                "AND received_at<=? AND received_at>=? ORDER BY received_at LIMIT 1",
                (mh, item_key, request_id, received_at, start)).fetchone()
        return r["request_id"] if r else None

    def reset(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM events")
            c.execute("DELETE FROM cases")
