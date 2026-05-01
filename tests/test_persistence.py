"""Tests for SQLiteProvider, MemoryStore, and RedisProvider persistence."""
from __future__ import annotations

import os

import pytest

from agent_recall_ai.core.state import SessionStatus, TaskState
from agent_recall_ai.persistence.sqlite_provider import SQLiteProvider
from agent_recall_ai.storage.memory import MemoryStore

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return SQLiteProvider(db_path=db_path)


@pytest.fixture
def memory_store():
    store = MemoryStore()
    yield store
    store.clear()


def make_state(session_id: str = "test-session") -> TaskState:
    state = TaskState(session_id=session_id)
    state.goals = ["Refactor auth module"]
    state.constraints = ["No API changes"]
    state.add_decision("Use PyJWT", reasoning="Well maintained")
    state.add_file("auth/tokens.py")
    state.token_usage.add(prompt=5000, completion=1000)
    state.cost_usd = 0.02
    return state


# ── SQLiteProvider Tests ──────────────────────────────────────────────────────

class TestSQLiteProvider:
    def test_save_and_load(self, tmp_db):
        state = make_state("session-1")
        tmp_db.save(state)
        loaded = tmp_db.load("session-1")
        assert loaded is not None
        assert loaded.session_id == "session-1"
        assert loaded.goals == ["Refactor auth module"]
        assert len(loaded.decisions) == 1

    def test_load_nonexistent_returns_none(self, tmp_db):
        result = tmp_db.load("does-not-exist")
        assert result is None

    def test_exists(self, tmp_db):
        assert not tmp_db.exists("s1")
        tmp_db.save(make_state("s1"))
        assert tmp_db.exists("s1")

    def test_checkpoint_seq_increments(self, tmp_db):
        state = make_state("s1")
        assert state.checkpoint_seq == 0
        tmp_db.save(state)
        assert state.checkpoint_seq == 1
        tmp_db.save(state)
        assert state.checkpoint_seq == 2

    def test_upsert_overwrites(self, tmp_db):
        state = make_state("s1")
        tmp_db.save(state)
        state.goals.append("Extra goal")
        tmp_db.save(state)
        loaded = tmp_db.load("s1")
        assert "Extra goal" in loaded.goals

    def test_delete(self, tmp_db):
        tmp_db.save(make_state("s1"))
        assert tmp_db.exists("s1")
        assert tmp_db.delete("s1")
        assert not tmp_db.exists("s1")

    def test_delete_nonexistent_returns_false(self, tmp_db):
        assert not tmp_db.delete("ghost")

    def test_list_sessions(self, tmp_db):
        tmp_db.save(make_state("alpha"))
        tmp_db.save(make_state("beta"))
        sessions = tmp_db.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "alpha" in ids
        assert "beta" in ids

    def test_list_sessions_filter_by_status(self, tmp_db):
        s1 = make_state("active-1")
        s2 = make_state("done-1")
        s2.status = SessionStatus.COMPLETED
        tmp_db.save(s1)
        tmp_db.save(s2)
        active = tmp_db.list_sessions(status=SessionStatus.ACTIVE)
        completed = tmp_db.list_sessions(status=SessionStatus.COMPLETED)
        assert all(s["status"] == "active" for s in active)
        assert all(s["status"] == "completed" for s in completed)

    def test_count(self, tmp_db):
        assert tmp_db.count() == 0
        tmp_db.save(make_state("a"))
        tmp_db.save(make_state("b"))
        assert tmp_db.count() == 2

    def test_search_decisions(self, tmp_db):
        state = make_state("s1")
        state.add_decision("Chosen PostgreSQL for ACID compliance")
        tmp_db.save(state)
        results = tmp_db.search_decisions("PostgreSQL")
        assert len(results) >= 1
        assert any("PostgreSQL" in r["decision_summary"] for r in results)

    def test_full_state_roundtrip(self, tmp_db):
        """All fields survive serialisation → deserialisation."""
        state = make_state("full-test")
        state.context_summary = "Working on auth refactor in FastAPI project"
        state.next_steps = ["Update tests", "Update docs"]
        state.open_questions = ["Should we migrate to OAuth2?"]
        state.context_utilization = 0.42
        tmp_db.save(state)
        loaded = tmp_db.load("full-test")
        assert loaded.context_summary == state.context_summary
        assert loaded.next_steps == state.next_steps
        assert loaded.open_questions == state.open_questions
        assert abs(loaded.context_utilization - 0.42) < 0.001


# ── MemoryStore Tests ─────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_save_and_load(self, memory_store):
        state = make_state("m1")
        memory_store.save(state)
        loaded = memory_store.load("m1")
        assert loaded is not None
        assert loaded.session_id == "m1"

    def test_load_returns_deep_copy(self, memory_store):
        state = make_state("m1")
        memory_store.save(state)
        loaded = memory_store.load("m1")
        loaded.goals.append("extra")
        loaded2 = memory_store.load("m1")
        assert "extra" not in loaded2.goals

    def test_delete(self, memory_store):
        memory_store.save(make_state("m1"))
        assert memory_store.delete("m1")
        assert not memory_store.exists("m1")

    def test_count(self, memory_store):
        assert memory_store.count() == 0
        memory_store.save(make_state("a"))
        assert memory_store.count() == 1

    def test_clear(self, memory_store):
        memory_store.save(make_state("a"))
        memory_store.save(make_state("b"))
        memory_store.clear()
        assert memory_store.count() == 0


# ── RedisProvider Tests ───────────────────────────────────────────────────────

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Skip entire class if redis package is not installed or server unreachable
def _redis_available() -> bool:
    try:
        import redis as _r
        client = _r.from_url(_REDIS_URL, socket_connect_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture
def redis_provider():
    """RedisProvider scoped to a unique test prefix so tests don't collide."""
    import time

    from agent_recall_ai.persistence.redis_provider import RedisProvider
    prefix = f"test_{int(time.time() * 1000)}"
    provider = RedisProvider(url=_REDIS_URL, prefix=prefix)
    yield provider
    # Cleanup: remove all keys for this prefix
    try:
        keys = provider._client.keys(f"{prefix}:*")
        if keys:
            provider._client.delete(*keys)
    except Exception:
        pass


@pytest.mark.skipif(not _redis_available(), reason="Redis not reachable")
class TestRedisProvider:
    def test_ping(self, redis_provider):
        assert redis_provider.ping() is True

    def test_save_and_load(self, redis_provider):
        state = make_state("redis-session-1")
        redis_provider.save(state)
        loaded = redis_provider.load("redis-session-1")
        assert loaded is not None
        assert loaded.session_id == "redis-session-1"
        assert loaded.goals == ["Refactor auth module"]
        assert loaded.constraints == ["No API changes"]

    def test_load_nonexistent_returns_none(self, redis_provider):
        assert redis_provider.load("does-not-exist") is None

    def test_exists(self, redis_provider):
        assert redis_provider.exists("redis-ex-1") is False
        redis_provider.save(make_state("redis-ex-1"))
        assert redis_provider.exists("redis-ex-1") is True

    def test_checkpoint_seq_increments(self, redis_provider):
        state = make_state("redis-seq-1")
        assert state.checkpoint_seq == 0
        redis_provider.save(state)
        assert state.checkpoint_seq == 1
        redis_provider.save(state)
        assert state.checkpoint_seq == 2

    def test_delete(self, redis_provider):
        redis_provider.save(make_state("redis-del-1"))
        assert redis_provider.exists("redis-del-1") is True
        result = redis_provider.delete("redis-del-1")
        assert result is True
        assert redis_provider.exists("redis-del-1") is False

    def test_delete_nonexistent_returns_false(self, redis_provider):
        assert redis_provider.delete("never-existed") is False

    def test_count(self, redis_provider):
        assert redis_provider.count() == 0
        redis_provider.save(make_state("redis-cnt-1"))
        redis_provider.save(make_state("redis-cnt-2"))
        assert redis_provider.count() == 2

    def test_list_sessions(self, redis_provider):
        redis_provider.save(make_state("redis-list-1"))
        redis_provider.save(make_state("redis-list-2"))
        sessions = redis_provider.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "redis-list-1" in ids
        assert "redis-list-2" in ids

    def test_list_sessions_filter_by_status(self, redis_provider):
        active = make_state("redis-active-1")
        active.status = SessionStatus.ACTIVE
        completed = make_state("redis-done-1")
        completed.status = SessionStatus.COMPLETED
        redis_provider.save(active)
        redis_provider.save(completed)

        active_only = redis_provider.list_sessions(status=SessionStatus.ACTIVE)
        ids = [s["session_id"] for s in active_only]
        assert "redis-active-1" in ids
        assert "redis-done-1" not in ids

    def test_decision_log(self, redis_provider):
        state = make_state("redis-decisions-1")
        redis_provider.save(state)
        log = redis_provider.get_decision_log("redis-decisions-1")
        assert len(log) == 1
        assert log[0]["summary"] == "Use PyJWT"

    def test_full_state_roundtrip(self, redis_provider):
        state = make_state("redis-roundtrip-1")
        state.next_steps = ["Deploy to staging"]
        state.open_questions = ["Which cloud provider?"]
        redis_provider.save(state)
        loaded = redis_provider.load("redis-roundtrip-1")
        assert loaded.next_steps == ["Deploy to staging"]
        assert loaded.open_questions == ["Which cloud provider?"]
        assert len(loaded.decisions) == 1
        assert loaded.token_usage.prompt == 5000

    def test_upsert_overwrites(self, redis_provider):
        state = make_state("redis-upsert-1")
        redis_provider.save(state)
        state.goals.append("Second goal")
        redis_provider.save(state)
        loaded = redis_provider.load("redis-upsert-1")
        assert "Second goal" in loaded.goals
        assert loaded.checkpoint_seq == 2
