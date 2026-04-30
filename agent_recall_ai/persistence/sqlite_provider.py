"""
SQLiteProvider — local development persistence.

Zero config: creates a single .db file at the specified path.
Uses WAL mode for concurrent read access.
Identical interface to RedisProvider for drop-in substitution.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.state import SessionStatus, TaskState


class SQLiteProvider:
    """
    Persist agent checkpoints to a local SQLite database.

    Recommended for: local development, CI, single-machine deployments.
    For distributed / production use, see RedisProvider.

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to .agent-recall-ai/sessions.db
    """

    def __init__(self, db_path: str = ".agent-recall-ai/sessions.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    session_id      TEXT PRIMARY KEY,
                    status          TEXT NOT NULL DEFAULT 'active',
                    framework       TEXT NOT NULL DEFAULT 'custom',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    checkpoint_seq  INTEGER NOT NULL DEFAULT 0,
                    cost_usd        REAL NOT NULL DEFAULT 0.0,
                    total_tokens    INTEGER NOT NULL DEFAULT 0,
                    goal_summary    TEXT NOT NULL DEFAULT '',
                    decision_count  INTEGER NOT NULL DEFAULT 0,
                    state_json      TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_status ON checkpoints(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_updated ON checkpoints(updated_at DESC)"
            )
            # Decision log table for fast queries without deserialising full state
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL REFERENCES checkpoints(session_id) ON DELETE CASCADE,
                    decision_summary TEXT NOT NULL,
                    is_anchor       INTEGER NOT NULL DEFAULT 0,
                    timestamp       TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dl_session ON decision_log(session_id)"
            )

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def save(self, state: TaskState) -> None:
        """Upsert a TaskState. Increments checkpoint_seq atomically."""
        state.checkpoint_seq += 1
        state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        goal_summary = state.goals[0][:120] if state.goals else ""
        state_json = state.model_dump_json()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                    (session_id, status, framework, created_at, updated_at,
                     checkpoint_seq, cost_usd, total_tokens, goal_summary,
                     decision_count, state_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    checkpoint_seq=excluded.checkpoint_seq,
                    cost_usd=excluded.cost_usd,
                    total_tokens=excluded.total_tokens,
                    goal_summary=excluded.goal_summary,
                    decision_count=excluded.decision_count,
                    state_json=excluded.state_json
                """,
                (
                    state.session_id,
                    state.status.value,
                    state.metadata.get("framework", "custom"),
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    state.checkpoint_seq,
                    state.cost_usd,
                    state.token_usage.total,
                    goal_summary,
                    len(state.decisions),
                    state_json,
                ),
            )
            # Sync decision log (delete old, insert new)
            conn.execute("DELETE FROM decision_log WHERE session_id = ?", (state.session_id,))
            from ..core.semantic_pruner import _is_decision_anchor
            for d in state.decisions:
                anchor_text = d.summary + " " + d.reasoning
                is_anchor = 1 if _is_decision_anchor(anchor_text) else 0
                conn.execute(
                    "INSERT INTO decision_log (session_id, decision_summary, is_anchor, timestamp) "
                    "VALUES (?,?,?,?)",
                    (state.session_id, d.summary[:200], is_anchor, d.timestamp.isoformat()),
                )

    def load(self, session_id: str) -> Optional[TaskState]:
        """Load a TaskState by session_id. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM checkpoints WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            return TaskState.model_validate_json(row["state_json"])

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        framework: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        query = """
            SELECT session_id, status, framework, created_at, updated_at,
                   checkpoint_seq, cost_usd, total_tokens, goal_summary, decision_count
            FROM checkpoints
        """
        conditions: list[str] = []
        params: list = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        if framework is not None:
            conditions.append("framework = ?")
            params.append(framework)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def search_decisions(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over the decision log."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT dl.session_id, dl.decision_summary, dl.timestamp, dl.is_anchor,
                       cp.goal_summary
                FROM decision_log dl
                JOIN checkpoints cp ON dl.session_id = cp.session_id
                WHERE dl.decision_summary LIKE ?
                ORDER BY dl.timestamp DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM checkpoints WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount > 0

    def exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM checkpoints WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row is not None

    def count(self, status: Optional[SessionStatus] = None) -> int:
        with self._connect() as conn:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE status = ?", (status.value,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
            return row[0] if row else 0
