"""Tests for the Checkpoint context manager and public API."""
from __future__ import annotations

import pytest

from agent_recall_ai import Checkpoint, resume
from agent_recall_ai.core.state import SessionStatus
from agent_recall_ai.monitors.cost_monitor import CostBudgetExceeded, CostMonitor
from agent_recall_ai.storage.memory import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


class TestCheckpointBasicAPI:
    def test_context_manager_saves_on_exit(self, store):
        with Checkpoint("task-1", store=store) as cp:
            cp.set_goal("Do something")
        assert store.exists("task-1")

    def test_completed_status_on_clean_exit(self, store):
        with Checkpoint("task-2", store=store) as cp:
            cp.set_goal("Deploy service")
        state = store.load("task-2")
        assert state.status == SessionStatus.COMPLETED

    def test_failed_status_on_exception(self, store):
        with pytest.raises(ValueError):
            with Checkpoint("task-err", store=store):
                raise ValueError("something went wrong")
        state = store.load("task-err")
        assert state.status == SessionStatus.FAILED

    def test_set_goal(self, store):
        with Checkpoint("task-3", store=store) as cp:
            cp.set_goal("Refactor auth module")
            cp.set_goal("Add JWT support")
        state = store.load("task-3")
        assert "Refactor auth module" in state.goals
        assert "Add JWT support" in state.goals

    def test_duplicate_goals_not_added(self, store):
        with Checkpoint("task-4", store=store) as cp:
            cp.set_goal("Same goal")
            cp.set_goal("Same goal")
        state = store.load("task-4")
        assert state.goals.count("Same goal") == 1

    def test_add_constraint(self, store):
        with Checkpoint("task-5", store=store) as cp:
            cp.add_constraint("Do not change public API")
        state = store.load("task-5")
        assert "Do not change public API" in state.constraints

    def test_record_decision(self, store):
        with Checkpoint("task-6", store=store) as cp:
            cp.record_decision(
                "Use PyJWT",
                reasoning="More maintained than python-jose",
                alternatives_rejected=["python-jose", "authlib"],
            )
        state = store.load("task-6")
        assert len(state.decisions) == 1
        d = state.decisions[0]
        assert d.summary == "Use PyJWT"
        assert "python-jose" in d.alternatives_rejected

    def test_record_file_modified(self, store):
        with Checkpoint("task-7", store=store) as cp:
            cp.record_file_modified("auth/tokens.py", action="modified")
            cp.record_file_modified("auth/middleware.py", action="created")
        state = store.load("task-7")
        paths = [f.path for f in state.files_modified]
        assert "auth/tokens.py" in paths
        assert "auth/middleware.py" in paths

    def test_record_tokens_updates_cost(self, store):
        with Checkpoint("task-8", store=store, model="gpt-4o-mini") as cp:
            cost = cp.record_tokens(prompt=10_000, completion=1_000)
            assert cost > 0
        state = store.load("task-8")
        assert state.cost_usd > 0
        assert state.token_usage.prompt == 10_000

    def test_record_tool_call(self, store):
        with Checkpoint("task-9", store=store) as cp:
            cp.record_tool_call("bash", input_summary="ls -la", output_summary="total 42")
        state = store.load("task-9")
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].tool_name == "bash"

    def test_next_step_and_open_question(self, store):
        with Checkpoint("task-10", store=store) as cp:
            cp.add_next_step("Update tests")
            cp.add_open_question("Should we migrate to OAuth2?")
        state = store.load("task-10")
        assert "Update tests" in state.next_steps
        assert "Should we migrate to OAuth2?" in state.open_questions

    def test_set_context(self, store):
        with Checkpoint("task-11", store=store) as cp:
            cp.set_context("Working on FastAPI auth refactor")
        state = store.load("task-11")
        assert "FastAPI" in state.context_summary


class TestCheckpointResume:
    def test_resume_loads_existing_session(self, store):
        with Checkpoint("resume-test", store=store) as cp:
            cp.set_goal("Original goal")
            cp.record_decision("Key decision")

        ctx = resume("resume-test", store=store)
        assert isinstance(ctx, str) and len(ctx) > 0
        assert "Original goal" in ctx

    def test_resume_nonexistent_returns_none(self, store):
        ctx = resume("no-such-session", store=store)
        assert ctx == ""

    def test_checkpoint_resumes_existing_on_open(self, store):
        # First session
        with Checkpoint("resumable", store=store) as cp:
            cp.set_goal("Goal 1")
            cp.record_decision("Decision 1")

        # Second session — same ID should load existing state
        with Checkpoint("resumable", store=store) as cp:
            assert "Goal 1" in cp.state.goals
            assert len(cp.state.decisions) == 1

    def test_resume_prompt_is_non_empty(self, store):
        with Checkpoint("prompt-test", store=store) as cp:
            cp.set_goal("Build something")
            cp.record_decision("Chose Python")
        state = store.load("prompt-test")
        prompt = state.resume_prompt()
        assert len(prompt) > 50
        assert "Build something" in prompt


class TestCheckpointMonitors:
    def test_cost_monitor_attached(self, store):
        monitor = CostMonitor(budget_usd=0.01, warn_at=0.5, raise_on_exceed=False)
        with Checkpoint("monitored", store=store, monitors=[monitor]) as cp:
            # Simulate high cost
            cp._state.cost_usd = 0.009
            cp._run_monitors("on_tokens")
        state = store.load("monitored")
        alert_types = [a.alert_type.value for a in state.alerts]
        assert "cost_warning" in alert_types

    def test_cost_exceeded_saves_before_raising(self, store):
        monitor = CostMonitor(budget_usd=0.01, raise_on_exceed=True)
        try:
            with Checkpoint("budget-exceeded", store=store, monitors=[monitor]) as cp:
                cp._state.cost_usd = 0.02
                cp._run_monitors("on_tokens")
        except CostBudgetExceeded:
            pass
        # Session should have been saved before raising
        assert store.exists("budget-exceeded")

    def test_handoff_payload(self, store):
        with Checkpoint("handoff-session", store=store) as cp:
            cp.set_goal("Deploy microservice")
            cp.add_constraint("No downtime")
            cp.record_decision("Blue-green deployment")
            cp.add_next_step("Monitor for 30min")

        state = store.load("handoff-session")
        handoff = state.as_handoff()
        assert handoff["goals"] == ["Deploy microservice"]
        assert "No downtime" in handoff["constraints"]
        assert len(handoff["decisions_summary"]) == 1
