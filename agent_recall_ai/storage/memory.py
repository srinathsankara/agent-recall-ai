"""
In-memory store for testing and ephemeral use.
Same interface as DiskStore but no persistence.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Optional

from ..core.state import SessionStatus, TaskState


class MemoryStore:
    """Non-persistent in-memory checkpoint store. Primarily for tests."""

    def __init__(self) -> None:
        self._sessions: dict[str, TaskState] = {}

    def save(self, state: TaskState) -> None:
        state.checkpoint_seq += 1
        state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self._sessions[state.session_id] = copy.deepcopy(state)

    def load(self, session_id: str) -> Optional[TaskState]:
        state = self._sessions.get(session_id)
        return copy.deepcopy(state) if state else None

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> list[dict]:
        sessions = list(self._sessions.values())
        if status is not None:
            sessions = [s for s in sessions if s.status == status]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "checkpoint_seq": s.checkpoint_seq,
                "cost_usd": s.cost_usd,
                "total_tokens": s.token_usage.total,
                "goal_summary": s.goals[0][:120] if s.goals else "",
            }
            for s in sessions[:limit]
        ]

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def count(self) -> int:
        return len(self._sessions)

    def clear(self) -> None:
        self._sessions.clear()
