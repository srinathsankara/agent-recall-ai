"""Tests for core state models."""
from __future__ import annotations

from agent_recall_ai.core.state import (
    AlertSeverity,
    AlertType,
    SessionStatus,
    TaskState,
    TokenUsage,
)


class TestTaskState:
    def test_default_construction(self):
        state = TaskState(session_id="test-session")
        assert state.session_id == "test-session"
        assert state.status == SessionStatus.ACTIVE
        assert state.goals == []
        assert state.decisions == []
        assert state.files_modified == []
        assert state.token_usage.total == 0
        assert state.cost_usd == 0.0
        assert state.checkpoint_seq == 0

    def test_add_decision(self):
        state = TaskState(session_id="s1")
        d = state.add_decision(
            "Use PostgreSQL",
            reasoning="ACID guarantees required",
            alternatives_rejected=["MongoDB", "DynamoDB"],
        )
        assert len(state.decisions) == 1
        assert d.summary == "Use PostgreSQL"
        assert d.reasoning == "ACID guarantees required"
        assert "MongoDB" in d.alternatives_rejected
        assert d.id  # has an ID

    def test_add_file(self):
        state = TaskState(session_id="s1")
        fc = state.add_file("auth/tokens.py", action="modified", description="Added JWT support")
        assert len(state.files_modified) == 1
        assert fc.path == "auth/tokens.py"
        assert fc.action == "modified"

    def test_add_tool_call(self):
        state = TaskState(session_id="s1")
        tc = state.add_tool_call(
            tool_name="bash",
            input_summary="ls -la",
            output_summary="total 42\n...",
            output_tokens=50,
        )
        assert len(state.tool_calls) == 1
        assert tc.tool_name == "bash"
        assert tc.output_tokens == 50

    def test_add_alert(self):
        state = TaskState(session_id="s1")
        alert = state.add_alert(
            AlertType.COST_WARNING,
            AlertSeverity.WARN,
            "Cost at 80%",
            detail={"cost_usd": 4.0, "budget_usd": 5.0},
        )
        assert len(state.alerts) == 1
        assert alert.severity == AlertSeverity.WARN
        assert alert.alert_type == AlertType.COST_WARNING

    def test_resume_prompt_contains_goals(self):
        state = TaskState(session_id="s1")
        state.goals = ["Refactor auth module", "Add JWT support"]
        state.constraints = ["Do not change public API"]
        prompt = state.resume_prompt()
        assert "Refactor auth module" in prompt
        assert "Add JWT support" in prompt
        assert "Do not change public API" in prompt
        assert "Resuming Agent Session" in prompt

    def test_resume_prompt_shows_decisions(self):
        state = TaskState(session_id="s1")
        state.add_decision("Used PyJWT", reasoning="More maintained")
        prompt = state.resume_prompt()
        assert "Used PyJWT" in prompt
        assert "More maintained" in prompt

    def test_resume_prompt_shows_files(self):
        state = TaskState(session_id="s1")
        state.add_file("auth/tokens.py")
        state.add_file("auth/middleware.py")
        prompt = state.resume_prompt()
        assert "auth/tokens.py" in prompt

    def test_as_handoff(self):
        state = TaskState(session_id="s1")
        state.goals = ["Deploy to prod"]
        state.constraints = ["No downtime"]
        state.add_decision("Blue-green deploy", reasoning="Zero downtime")
        state.next_steps = ["Monitor for 30 min"]

        handoff = state.as_handoff()
        assert handoff["session_id"] == "s1"
        assert "Deploy to prod" in handoff["goals"]
        assert len(handoff["decisions_summary"]) == 1
        assert "Monitor for 30 min" in handoff["next_steps"]

    def test_updated_at_changes_on_mutation(self):
        state = TaskState(session_id="s1")
        before = state.updated_at
        state.add_decision("A decision")
        assert state.updated_at >= before


class TestTokenUsage:
    def test_total_property(self):
        usage = TokenUsage(prompt=1000, completion=200)
        assert usage.total == 1200

    def test_add(self):
        usage = TokenUsage()
        usage.add(prompt=500, completion=100)
        usage.add(prompt=300, completion=50)
        assert usage.prompt == 800
        assert usage.completion == 150
        assert usage.total == 950
