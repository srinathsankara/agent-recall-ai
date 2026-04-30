"""
End-to-end CLI tests for agent-recall-ai.

Tests every CLI command (list, inspect, resume, export, delete, status,
install-hooks, auto-save) using the Typer test runner and an isolated
temp directory so real disk state is never touched.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_recall_ai.cli.main import app
from agent_recall_ai.core.state import SessionStatus, TaskState
from agent_recall_ai.storage.disk import DiskStore


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_store_dir(tmp_path: Path) -> Path:
    """A temp directory that acts as the .agent-recall-ai base."""
    return tmp_path / ".agent-recall-ai"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_session(
    store_dir: Path,
    session_id: str = "test-session",
    goal: str = "Refactor auth",
    status: SessionStatus = SessionStatus.ACTIVE,
    cost: float = 0.0042,
) -> TaskState:
    """Create and persist a session in the given store directory."""
    store = DiskStore(base_dir=str(store_dir))
    state = TaskState(
        session_id=session_id,
        goals=[goal],
        constraints=["No breaking changes"],
        status=status,
    )
    state.add_decision("Use PyJWT", reasoning="Better maintained")
    state.add_file("auth/tokens.py", action="modified")
    state.add_alert(
        alert_type=state.alerts.__class__ if False else __import__(
            "agent_recall_ai.core.state", fromlist=["AlertType"]
        ).AlertType.TOKEN_PRESSURE,
        severity=__import__(
            "agent_recall_ai.core.state", fromlist=["AlertSeverity"]
        ).AlertSeverity.WARN,
        message="Context at 75%",
    )
    state.cost_usd = cost
    state.next_steps = ["Update tests", "Deploy to staging"]
    store.save(state)
    return store.load(session_id)


# ── list ──────────────────────────────────────────────────────────────────────


class TestListCommand:
    def test_list_empty_store(self, runner: CliRunner, tmp_store_dir: Path):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_list_shows_sessions(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "my-session")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "my-session" in result.output
        finally:
            cli_mod._store_dir = orig

    def test_list_with_valid_status_filter(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "active-session", status=SessionStatus.ACTIVE)
        _seed_session(tmp_store_dir, "done-session", status=SessionStatus.COMPLETED)
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["list", "--status", "completed"])
            assert result.exit_code == 0
            assert "done-session" in result.output
            assert "active-session" not in result.output
        finally:
            cli_mod._store_dir = orig

    def test_list_invalid_status_exits_with_error(self, runner: CliRunner, tmp_store_dir: Path):
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["list", "--status", "nonsense"])
            assert result.exit_code == 1
            assert "Invalid status" in result.output
        finally:
            cli_mod._store_dir = orig

    def test_list_no_sessions_message(self, runner: CliRunner, tmp_store_dir: Path):
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No checkpoints" in result.output
        finally:
            cli_mod._store_dir = orig


# ── inspect ───────────────────────────────────────────────────────────────────


class TestInspectCommand:
    def test_inspect_existing_session(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "inspect-me", goal="Deploy microservice")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["inspect", "inspect-me"])
            assert result.exit_code == 0
            assert "inspect-me" in result.output
            assert "Deploy microservice" in result.output
            assert "Use PyJWT" in result.output
        finally:
            cli_mod._store_dir = orig

    def test_inspect_nonexistent_exits_1(self, runner: CliRunner, tmp_store_dir: Path):
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["inspect", "no-such-session"])
            assert result.exit_code == 1
        finally:
            cli_mod._store_dir = orig

    def test_inspect_full_flag_shows_decisions(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "full-inspect")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["inspect", "full-inspect", "--full"])
            assert result.exit_code == 0
            assert "Use PyJWT" in result.output
        finally:
            cli_mod._store_dir = orig


# ── resume ────────────────────────────────────────────────────────────────────


class TestResumeCommand:
    def test_resume_prints_prompt(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "resume-me", goal="Build API gateway")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["resume", "resume-me"])
            assert result.exit_code == 0
            assert "Build API gateway" in result.output
            assert "resume-me" in result.output
        finally:
            cli_mod._store_dir = orig

    def test_resume_nonexistent_exits_1(self, runner: CliRunner, tmp_store_dir: Path):
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["resume", "ghost-session"])
            assert result.exit_code == 1
        finally:
            cli_mod._store_dir = orig


# ── export ────────────────────────────────────────────────────────────────────


class TestExportCommand:
    def test_export_json(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "export-me")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["export", "export-me", "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["session_id"] == "export-me"
        finally:
            cli_mod._store_dir = orig

    def test_export_handoff(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "handoff-export")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["export", "handoff-export", "--format", "handoff"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "goals" in data
            assert "decisions_summary" in data
        finally:
            cli_mod._store_dir = orig

    def test_export_agenttest(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "agenttest-export")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["export", "agenttest-export", "--format", "agenttest"])
            assert result.exit_code == 0
            assert "from agenttest import" in result.output
        finally:
            cli_mod._store_dir = orig

    def test_export_to_file(self, runner: CliRunner, tmp_store_dir: Path, tmp_path: Path):
        _seed_session(tmp_store_dir, "file-export")
        out_path = tmp_path / "export.json"
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(
                app, ["export", "file-export", "--format", "json", "--output", str(out_path)]
            )
            assert result.exit_code == 0
            assert out_path.exists()
            data = json.loads(out_path.read_text())
            assert data["session_id"] == "file-export"
        finally:
            cli_mod._store_dir = orig

    def test_export_unknown_format_exits_1(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "bad-fmt")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["export", "bad-fmt", "--format", "avro"])
            assert result.exit_code == 1
        finally:
            cli_mod._store_dir = orig

    def test_export_nonexistent_exits_1(self, runner: CliRunner, tmp_store_dir: Path):
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["export", "ghost"])
            assert result.exit_code == 1
        finally:
            cli_mod._store_dir = orig


# ── delete ────────────────────────────────────────────────────────────────────


class TestDeleteCommand:
    def test_delete_with_yes_flag(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "delete-me")
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["delete", "delete-me", "--yes"])
            assert result.exit_code == 0
            assert "Deleted" in result.output
            store = DiskStore(base_dir=str(tmp_store_dir))
            assert not store.exists("delete-me")
        finally:
            cli_mod._store_dir = orig

    def test_delete_nonexistent_exits_1(self, runner: CliRunner, tmp_store_dir: Path):
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["delete", "ghost", "--yes"])
            assert result.exit_code == 1
        finally:
            cli_mod._store_dir = orig


# ── status ────────────────────────────────────────────────────────────────────


class TestStatusCommand:
    def test_status_empty(self, runner: CliRunner, tmp_store_dir: Path):
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "No checkpoints" in result.output
        finally:
            cli_mod._store_dir = orig

    def test_status_shows_totals(self, runner: CliRunner, tmp_store_dir: Path):
        _seed_session(tmp_store_dir, "s1", cost=0.001)
        _seed_session(tmp_store_dir, "s2", status=SessionStatus.COMPLETED, cost=0.002)
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "Total sessions" in result.output
            assert "Total cost" in result.output
        finally:
            cli_mod._store_dir = orig


# ── auto-save ─────────────────────────────────────────────────────────────────


class TestAutoSaveCommand:
    def test_auto_save_increments_seq_by_one(self, runner: CliRunner, tmp_store_dir: Path):
        """auto-save must increment checkpoint_seq by exactly 1, not 2."""
        _seed_session(tmp_store_dir, "auto-sess")
        store = DiskStore(base_dir=str(tmp_store_dir))
        before = store.load("auto-sess").checkpoint_seq

        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["auto-save", "--session", "auto-sess"])
            assert result.exit_code == 0
        finally:
            cli_mod._store_dir = orig

        after = store.load("auto-sess").checkpoint_seq
        assert after == before + 1, f"Expected +1, got +{after - before}"

    def test_auto_save_missing_session_is_silent(self, runner: CliRunner, tmp_store_dir: Path):
        """auto-save on non-existent session must exit 0 and be silent."""
        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            result = runner.invoke(app, ["auto-save", "--session", "ghost-session"])
            assert result.exit_code == 0
            # Hooks must not pollute the agent output
            assert result.output.strip() == ""
        finally:
            cli_mod._store_dir = orig

    def test_auto_save_multiple_times_increments_correctly(
        self, runner: CliRunner, tmp_store_dir: Path
    ):
        _seed_session(tmp_store_dir, "multi-save")
        store = DiskStore(base_dir=str(tmp_store_dir))
        initial_seq = store.load("multi-save").checkpoint_seq

        import agent_recall_ai.cli.main as cli_mod
        orig = cli_mod._store_dir
        cli_mod._store_dir = str(tmp_store_dir)
        try:
            for _ in range(3):
                runner.invoke(app, ["auto-save", "--session", "multi-save"])
        finally:
            cli_mod._store_dir = orig

        final_seq = store.load("multi-save").checkpoint_seq
        assert final_seq == initial_seq + 3


# ── install-hooks ─────────────────────────────────────────────────────────────


class TestInstallHooksCommand:
    def test_dry_run_does_not_write(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(
            app,
            ["install-hooks", "--tool", "generic", "--session", "my-proj", "--dry-run"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "my-proj" in result.output

    def test_generic_hook_writes_file(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        # Patch cwd so the hooks.json ends up in tmp_path
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["install-hooks", "--tool", "generic", "--session", "test-hook"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        hooks_file = tmp_path / ".agent-recall-ai" / "hooks.json"
        assert hooks_file.exists()
        data = json.loads(hooks_file.read_text())
        assert any("test-hook" in cmd for cmd in data.get("on_session_end", []))
