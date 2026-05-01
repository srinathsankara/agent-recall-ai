"""
SQLite + JSON disk store for agent checkpoints.

Each session is stored as a row in SQLite (for fast querying) with the full
TaskState serialized as JSON. This makes it easy to list/search/delete sessions
while keeping the full state intact.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..core.state import SessionStatus, TaskState


class DiskStore:
    """Persist checkpoints to a local SQLite database."""

    def __init__(self, base_dir: str = ".agent-recall-ai") -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "sessions.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    checkpoint_seq INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    goal_summary TEXT NOT NULL DEFAULT '',
                    state_json TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)"
            )

    def save(self, state: TaskState) -> None:
        """Upsert a TaskState. Increments checkpoint_seq on every save."""
        state.checkpoint_seq += 1
        state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        goal_summary = state.goals[0][:120] if state.goals else ""
        state_json = state.model_dump_json()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (session_id, status, created_at, updated_at, checkpoint_seq,
                     cost_usd, total_tokens, goal_summary, state_json)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    checkpoint_seq=excluded.checkpoint_seq,
                    cost_usd=excluded.cost_usd,
                    total_tokens=excluded.total_tokens,
                    goal_summary=excluded.goal_summary,
                    state_json=excluded.state_json
                """,
                (
                    state.session_id,
                    state.status.value,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    state.checkpoint_seq,
                    state.cost_usd,
                    state.token_usage.total,
                    goal_summary,
                    state_json,
                ),
            )

    def load(self, session_id: str) -> TaskState | None:
        """Load a TaskState by session_id. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            return TaskState.model_validate_json(row["state_json"])

    def list_sessions(
        self,
        status: SessionStatus | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List sessions with lightweight metadata (no full state_json)."""
        query = """
            SELECT session_id, status, created_at, updated_at,
                   checkpoint_seq, cost_usd, total_tokens, goal_summary
            FROM sessions
        """
        params: list = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def delete(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount > 0

    def exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row is not None

    def search_decisions(self, query: str, limit: int = 20) -> list[dict]:
        """
        Search decisions across all sessions by substring match.

        Deserializes each session's state_json to scan decisions — use
        SQLiteProvider for a dedicated decision_log table with indexed search.
        Returns list of dicts with: session_id, decision_summary, timestamp, goal_summary.
        """
        query_lower = query.lower()
        results: list[dict] = []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, goal_summary, state_json FROM sessions ORDER BY updated_at DESC"
            ).fetchall()

        for row in rows:
            try:
                state = TaskState.model_validate_json(row["state_json"])
            except Exception:
                continue
            for d in state.decisions:
                if query_lower in d.summary.lower() or query_lower in d.reasoning.lower():
                    results.append({
                        "session_id": row["session_id"],
                        "decision_summary": d.summary[:200],
                        "timestamp": d.timestamp.isoformat(),
                        "goal_summary": row["goal_summary"],
                    })
                    if len(results) >= limit:
                        return results

        return results

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            return row[0] if row else 0
