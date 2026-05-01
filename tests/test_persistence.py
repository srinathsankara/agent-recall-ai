"""Tests for SQLiteProvider and MemoryStore persistence."""
from __future__ import annotations

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
