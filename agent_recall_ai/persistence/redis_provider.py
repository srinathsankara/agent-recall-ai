"""
RedisProvider — production state hydration and distribution.

Designed for:
- Distributed agent deployments (multiple workers sharing state)
- Fast context hydration (checkpoint loads in <5ms from Redis)
- TTL-based expiry of old sessions
- Pub/sub notifications when checkpoints are updated (enables real-time dashboards)

Key design decisions:
- State stored as Redis Hash: HSET {prefix}:{session_id} state <json>
- Session index stored as Sorted Set: ZADD {prefix}:index <timestamp> <session_id>
- Decision log stored as Redis List: RPUSH {prefix}:{session_id}:decisions <summary>
- TTL defaults to 7 days for active sessions, 1 day for completed

Requires: pip install 'agent-recall-ai[redis]' (adds redis>=5.0)

Falls back gracefully with a clear error if redis is not installed.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from ..core.state import SessionStatus, TaskState

logger = logging.getLogger(__name__)

try:
    import redis as _redis_module
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# TTL constants (seconds)
_TTL_ACTIVE = 7 * 24 * 3600       # 7 days for active sessions
_TTL_COMPLETED = 24 * 3600         # 1 day for completed/failed
_TTL_INDEX = 30 * 24 * 3600        # 30 days for the session index


class RedisProvider:
    """
    Persist and hydrate agent checkpoints via Redis.

    Recommended for: production, distributed agents, multi-machine deployments.
    For local development, see SQLiteProvider.

    Args:
        url: Redis connection URL. Default: redis://localhost:6379/0
        prefix: Key prefix to namespace all keys. Default: "agentcp"
        default_ttl: Seconds before a key expires. Default: 7 days.
        enable_pubsub: Publish to a channel when a checkpoint is saved.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "agentcp",
        default_ttl: int = _TTL_ACTIVE,
        enable_pubsub: bool = False,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required: pip install 'agent-recall-ai[redis]' "
                "(adds redis>=5.0)"
            )
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._enable_pubsub = enable_pubsub
        self._client: Any = _redis_module.from_url(url, decode_responses=True)

    # ── Key helpers ───────────────────────────────────────────────────────────

    def _state_key(self, session_id: str) -> str:
        return f"{self._prefix}:state:{session_id}"

    def _meta_key(self, session_id: str) -> str:
        return f"{self._prefix}:meta:{session_id}"

    def _decisions_key(self, session_id: str) -> str:
        return f"{self._prefix}:decisions:{session_id}"

    def _index_key(self) -> str:
        return f"{self._prefix}:index"

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save(self, state: TaskState) -> None:
        """Upsert a TaskState. Increments checkpoint_seq and sets TTL."""
        state.checkpoint_seq += 1
        state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        ttl = _TTL_ACTIVE if state.status == SessionStatus.ACTIVE else _TTL_COMPLETED
        state_json = state.model_dump_json()
        score = time.time()

        pipe = self._client.pipeline(transaction=True)

        # Store full state
        pipe.set(self._state_key(state.session_id), state_json, ex=ttl)

        # Store lightweight metadata for list operations
        meta = {
            "session_id": state.session_id,
            "status": state.status.value,
            "checkpoint_seq": str(state.checkpoint_seq),
            "cost_usd": str(state.cost_usd),
            "total_tokens": str(state.token_usage.total),
            "goal_summary": state.goals[0][:120] if state.goals else "",
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "framework": state.metadata.get("framework", "custom"),
        }
        pipe.hset(self._meta_key(state.session_id), mapping=meta)
        pipe.expire(self._meta_key(state.session_id), ttl)

        # Update sorted index (score = timestamp for time-ordered listing)
        pipe.zadd(self._index_key(), {state.session_id: score})
        pipe.expire(self._index_key(), _TTL_INDEX)

        # Update decision log (replace with current)
        decisions_key = self._decisions_key(state.session_id)
        pipe.delete(decisions_key)
        for d in state.decisions[-50:]:    # cap at 50 most recent
            pipe.rpush(decisions_key, json.dumps({"summary": d.summary, "is_anchor": True}))
        if state.decisions:
            pipe.expire(decisions_key, ttl)

        pipe.execute()

        if self._enable_pubsub:
            try:
                self._client.publish(
                    f"{self._prefix}:events",
                    json.dumps({"event": "checkpoint_saved", "session_id": state.session_id}),
                )
            except Exception:
                pass

    def load(self, session_id: str) -> Optional[TaskState]:
        """Load a TaskState by session_id. Returns None if not found or expired."""
        raw = self._client.get(self._state_key(session_id))
        if raw is None:
            return None
        try:
            return TaskState.model_validate_json(raw)
        except Exception as exc:
            logger.error("Failed to deserialise checkpoint %s: %s", session_id, exc)
            return None

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List sessions ordered by most recently updated."""
        # Get session IDs from the sorted index (descending by timestamp)
        session_ids = self._client.zrevrange(self._index_key(), 0, limit * 2)

        results: list[dict] = []
        pipe = self._client.pipeline(transaction=False)
        for sid in session_ids:
            pipe.hgetall(self._meta_key(sid))
        meta_list = pipe.execute()

        for meta in meta_list:
            if not meta:
                continue
            if status is not None and meta.get("status") != status.value:
                continue
            results.append({
                "session_id": meta.get("session_id", ""),
                "status": meta.get("status", "unknown"),
                "framework": meta.get("framework", "custom"),
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
                "checkpoint_seq": int(meta.get("checkpoint_seq", 0)),
                "cost_usd": float(meta.get("cost_usd", 0)),
                "total_tokens": int(meta.get("total_tokens", 0)),
                "goal_summary": meta.get("goal_summary", ""),
            })
            if len(results) >= limit:
                break

        return results

    def get_decision_log(self, session_id: str) -> list[dict]:
        """Fast retrieval of the decision log without loading the full state."""
        raw_decisions = self._client.lrange(self._decisions_key(session_id), 0, -1)
        result: list[dict] = []
        for raw in raw_decisions:
            try:
                result.append(json.loads(raw))
            except Exception:
                result.append({"summary": raw, "is_anchor": False})
        return result

    def delete(self, session_id: str) -> bool:
        pipe = self._client.pipeline(transaction=True)
        pipe.delete(self._state_key(session_id))
        pipe.delete(self._meta_key(session_id))
        pipe.delete(self._decisions_key(session_id))
        pipe.zrem(self._index_key(), session_id)
        results = pipe.execute()
        return bool(results[0])

    def exists(self, session_id: str) -> bool:
        return bool(self._client.exists(self._state_key(session_id)))

    def count(self, status: Optional[SessionStatus] = None) -> int:
        if status is None:
            return int(self._client.zcard(self._index_key()))
        # For filtered count we need to scan (expensive — use SQLite for analytics)
        sessions = self.list_sessions(status=status, limit=10_000)
        return len(sessions)

    def ping(self) -> bool:
        """Check if the Redis connection is alive."""
        try:
            return bool(self._client.ping())
        except Exception:
            return False
