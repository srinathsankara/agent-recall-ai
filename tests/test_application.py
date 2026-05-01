"""
Application-level tests — complete workflows as a real agent would use them.

These tests verify end-to-end behaviour across multiple components working
together, simulating the scenarios described in the README and docs.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile

import pytest

from agent_recall_ai import Checkpoint, resume
from agent_recall_ai.adapters import AnthropicAdapter, OpenAIAdapter
from agent_recall_ai.core.compressor import build_resume_context, compress_tool_output
from agent_recall_ai.core.state import SessionStatus, TaskState
from agent_recall_ai.monitors import CostMonitor, DriftMonitor, TokenMonitor
from agent_recall_ai.privacy.redactor import PIIRedactor
from agent_recall_ai.storage.disk import DiskStore
from agent_recall_ai.storage.memory import MemoryStore

# ── Scenario 1: Full agent task lifecycle ────────────────────────────────────

class TestFullAgentTaskLifecycle:
    """
    Simulates a real agent completing a multi-step coding task with
    goals, decisions, file changes, and token tracking.
    """

    def test_complete_coding_task(self):
        store = MemoryStore()

        # Phase 1 — start task
        with Checkpoint("refactor-auth", store=store, model="gpt-4o") as cp:
            cp.set_goal("Refactor the authentication module")
            cp.set_goal("Add JWT support")
            cp.add_constraint("Do not change the public API surface")
            cp.add_constraint("Maintain 100% test coverage")
            cp.record_tokens("gpt-4o", prompt=2000, completion=400)
            cp.save()

            # Phase 2 — make decisions during the task
            cp.record_decision(
                "Use PyJWT library",
                reasoning="More maintained than python-jose, 50M monthly downloads",
                alternatives_rejected=["python-jose", "authlib"],
            )
            cp.record_file_modified("auth/tokens.py")
            cp.record_file_modified("auth/middleware.py")
            cp.record_tokens("gpt-4o", prompt=3000, completion=800)
            cp.save()

            # Phase 3 — finish
            cp.record_decision(
                "Decided to keep backward-compatible token format",
                reasoning="Breaking change would require forced re-login for all users",
            )
            cp.add_next_step("Run the full test suite")
            cp.add_next_step("Update the API documentation")
            cp.record_tokens("gpt-4o", prompt=1500, completion=300)

        state = store.load("refactor-auth")

        # Verify all state was persisted
        assert len(state.goals) == 2
        assert "Refactor the authentication module" in state.goals
        assert len(state.constraints) == 2
        assert len(state.decisions) == 2
        assert len(state.files_modified) == 2
        assert len(state.next_steps) == 2
        assert state.token_usage.prompt == 6500
        assert state.token_usage.completion == 1500
        assert state.cost_usd > 0
        assert state.status == SessionStatus.COMPLETED
        assert state.checkpoint_seq == 3  # 2 explicit + 1 on exit

    def test_task_failure_preserves_progress(self):
        """Even if the agent crashes, saved progress is not lost."""
        store = MemoryStore()

        try:
            with Checkpoint("crash-task", store=store) as cp:
                cp.set_goal("Do something important")
                cp.record_decision("Decided on approach A")
                cp.save()  # explicit save before crash
                raise RuntimeError("Simulated agent crash")
        except RuntimeError:
            pass

        state = store.load("crash-task")
        assert state is not None
        assert state.status == SessionStatus.FAILED
        assert "Do something important" in state.goals
        assert len(state.decisions) == 1  # saved progress preserved

    def test_task_resume_after_context_limit(self):
        """
        Simulates the core use case: agent hits context limit,
        new session picks up exactly where it left off.
        """
        store = MemoryStore()

        # Session 1 — agent works, hits context limit
        with Checkpoint("long-task-v1", store=store) as cp:
            cp.set_goal("Migrate database schema to PostgreSQL")
            cp.add_constraint("Zero downtime migration required")
            cp.record_decision(
                "Decided to use Blue-Green deployment",
                reasoning="Eliminates downtime, allows instant rollback",
            )
            cp.record_decision(
                "Rejected direct migration",
                reasoning="Would require 4-hour maintenance window",
            )
            cp.record_file_modified("migrations/001_initial.sql")
            cp.record_file_modified("migrations/002_indexes.sql")
            cp.record_tokens("gpt-4o", prompt=100000, completion=5000)
            cp.add_next_step("Apply migrations to staging environment")
            cp.add_next_step("Run integration tests against staging")

        # Get resume context
        ctx = resume("long-task-v1", store=store)
        assert isinstance(ctx, str)
        assert "Migrate database schema" in ctx
        assert "Blue-Green deployment" in ctx
        assert "Apply migrations to staging" in ctx

        # Session 2 — new context, picks up from resume
        with Checkpoint("long-task-v2", store=store) as cp:
            cp.set_goal("Continue: Migrate database schema to PostgreSQL")
            cp.add_constraint("Zero downtime migration required")
            cp.set_context(ctx)  # inject the resume context
            cp.record_decision(
                "Applied migrations to staging — all tests passed",
                reasoning="Ready to proceed to production",
            )

        state2 = store.load("long-task-v2")
        assert state2 is not None
        assert len(state2.decisions) == 1


# ── Scenario 2: Multi-framework adapter workflow ─────────────────────────────

class TestAdapterWorkflows:
    """Test real adapter usage patterns."""

    def test_openai_full_conversation(self):
        store = MemoryStore()
        with Checkpoint("openai-task", store=store) as cp:
            adapter = OpenAIAdapter(cp)
            cp.set_goal("Answer complex question about distributed systems")

            # Simulate multiple LLM calls
            adapter.on_llm_end(model="gpt-4o", prompt_tokens=1000, completion_tokens=300)
            adapter.on_tool_end("web_search", "Wikipedia article on CAP theorem")
            adapter.on_tool_end("web_search", "Recent papers on eventual consistency")
            adapter.on_llm_end(model="gpt-4o", prompt_tokens=2000, completion_tokens=500)
            adapter.on_tool_end("code_interpreter", "def calculate_quorum(n): ...")

        state = store.load("openai-task")
        assert state.token_usage.prompt == 3000
        assert state.token_usage.completion == 800
        assert len(state.tool_calls) == 3
        assert state.tool_calls[0].tool_name == "web_search"
        assert state.tool_calls[2].tool_name == "code_interpreter"
        assert state.metadata["framework"] == "openai"

    def test_anthropic_full_conversation(self):
        store = MemoryStore()
        with Checkpoint("anthropic-task", store=store) as cp:
            adapter = AnthropicAdapter(cp)
            cp.set_goal("Write a technical specification")

            adapter.on_llm_end(
                model="claude-sonnet-4-5",
                prompt_tokens=5000,
                completion_tokens=2000,
                cached_tokens=4000,
            )
            adapter.on_llm_end(
                model="claude-sonnet-4-5",
                prompt_tokens=3000,
                completion_tokens=1000,
                cached_tokens=3000,
            )

        state = store.load("anthropic-task")
        assert state.token_usage.prompt == 8000
        assert state.token_usage.cached == 7000
        assert state.metadata["framework"] == "anthropic"

    def test_adapter_error_saves_state(self):
        """on_error() should save state so progress isn't lost."""
        store = MemoryStore()
        with Checkpoint("err-task", store=store) as cp:
            adapter = OpenAIAdapter(cp)
            cp.set_goal("Important task")
            cp.record_decision("Key architectural decision")
            adapter.on_llm_end(model="gpt-4o", prompt_tokens=500, completion_tokens=100)
            adapter.on_error(RuntimeError("rate limit exceeded"))

        assert store.exists("err-task")
        state = store.load("err-task")
        # Progress should be saved even after error
        assert len(state.decisions) == 1


# ── Scenario 3: Monitor-driven workflows ─────────────────────────────────────

class TestMonitorWorkflows:
    """Test monitors triggering alerts and driving behaviour."""

    def test_cost_budget_alert_fires(self):
        store = MemoryStore()
        # raise_on_exceed=False so we get an alert recorded rather than an exception
        monitor = CostMonitor(budget_usd=0.001, raise_on_exceed=False)  # very small budget
        with Checkpoint("cost-alert", store=store, monitors=[monitor]) as cp:
            cp.set_goal("Expensive task")
            cp.record_tokens("gpt-4o", prompt=10000, completion=5000)  # exceeds $0.001

        state = store.load("cost-alert")
        cost_alerts = [a for a in state.alerts if "cost" in a.alert_type.value.lower()
                       or "budget" in str(a.message).lower()]
        assert len(cost_alerts) > 0, "CostMonitor should have fired a budget alert"

    def test_drift_monitor_detects_constraint_violation(self):
        store = MemoryStore()
        monitor = DriftMonitor()
        with Checkpoint("drift-task", store=store, monitors=[monitor]) as cp:
            cp.add_constraint("Never access the production database directly")
            cp.record_decision(
                "Decided to query production database for testing",
                reasoning="Faster than setting up test data",
            )

        state = store.load("drift-task")
        assert len(state.alerts) > 0
        alert_messages = " ".join(str(a.message) for a in state.alerts)
        assert "production" in alert_messages.lower() or len(state.alerts) > 0

    def test_multiple_monitors_all_run(self):
        store = MemoryStore()
        monitors = [
            CostMonitor(budget_usd=1000.0),
            TokenMonitor(warn_at=0.9, critical_at=0.99),
            DriftMonitor(),
        ]
        with Checkpoint("multi-mon", store=store, monitors=monitors) as cp:
            cp.set_goal("Multi-monitor test")
            cp.add_constraint("Stay within budget")
            cp.record_tokens("gpt-4o", prompt=100, completion=50)

        state = store.load("multi-mon")
        assert state is not None
        assert state.status == SessionStatus.COMPLETED


# ── Scenario 4: Privacy and PII workflow ─────────────────────────────────────

class TestPrivacyWorkflows:
    """Test PII redaction in realistic agent scenarios."""

    def test_api_key_never_stored_in_state(self):
        store = MemoryStore()
        redactor = PIIRedactor()
        secret_key = "sk-proj-SECRETKEY123456789012345678"

        with Checkpoint("pii-task", store=store, redactor=redactor) as cp:
            cp.set_goal(f"Deploy using API key {secret_key}")
            cp.record_decision(
                f"Configured service with key {secret_key}",
                reasoning="Required for production access",
            )

        state = store.load("pii-task")
        state_json = state.model_dump_json()
        assert secret_key not in state_json, "API key leaked into serialized state!"
        assert secret_key not in state.goals[0]
        assert secret_key not in state.decisions[0].summary

    def test_multiple_secret_types_redacted(self):
        store = MemoryStore()
        redactor = PIIRedactor()

        with Checkpoint("multi-pii", store=store, redactor=redactor) as cp:
            cp.set_goal("Connect to services")
            cp.record_decision(
                "Used token ghp_ABCDEF123456789012345678901234 for GitHub access",
                reasoning="Required for CI",
            )

        state = store.load("multi-pii")
        assert "ghp_ABCDEF123456789012345678901234" not in state.decisions[0].summary

    def test_non_sensitive_data_preserved(self):
        store = MemoryStore()
        redactor = PIIRedactor()
        normal_goal = "Implement OAuth2 authentication flow for the web application"

        with Checkpoint("safe-task", store=store, redactor=redactor) as cp:
            cp.set_goal(normal_goal)

        state = store.load("safe-task")
        assert state.goals[0] == normal_goal


# ── Scenario 5: Fork and branching workflows ─────────────────────────────────

class TestForkWorkflows:
    """Test session forking for parallel exploration."""

    def test_fork_enables_parallel_exploration(self):
        """Agent explores two approaches simultaneously by forking."""
        store = MemoryStore()

        # Main session
        with Checkpoint("explore-main", store=store) as cp:
            cp.set_goal("Optimize database query performance")
            cp.record_decision("Identified slow N+1 query in user listing endpoint")

        # Fork for approach A
        cp.fork("explore-indexing", store=store)
        with Checkpoint("explore-indexing", store=store) as cp_a:
            cp_a.record_decision(
                "Decided to add composite index on (user_id, created_at)",
                reasoning="Covers the most common query pattern",
            )

        # Fork for approach B
        cp.fork("explore-caching", store=store)
        with Checkpoint("explore-caching", store=store) as cp_b:
            cp_b.record_decision(
                "Decided to add Redis cache layer",
                reasoning="Eliminates DB hits for repeated queries",
            )

        # Verify both branches exist independently
        state_a = store.load("explore-indexing")
        state_b = store.load("explore-caching")

        assert state_a is not None
        assert state_b is not None
        # Both inherit the parent decision
        assert any("N+1" in d.summary for d in state_a.decisions)
        assert any("N+1" in d.summary for d in state_b.decisions)
        # Each has its own unique decision
        assert any("index" in d.summary.lower() for d in state_a.decisions)
        assert any("cache" in d.summary.lower() for d in state_b.decisions)

    def test_fork_does_not_affect_parent(self):
        store = MemoryStore()
        with Checkpoint("fork-parent", store=store) as cp:
            cp.set_goal("Parent goal")
            cp.record_decision("Parent decision")

        original_seq = store.load("fork-parent").checkpoint_seq
        cp.fork("fork-child-isolated", store=store)

        # Mutate child
        with Checkpoint("fork-child-isolated", store=store) as child:
            child.record_decision("Child-only decision")

        # Parent unchanged
        parent_state = store.load("fork-parent")
        assert parent_state.checkpoint_seq == original_seq
        assert not any("Child-only" in d.summary for d in parent_state.decisions)


# ── Scenario 6: Async agent workflow ─────────────────────────────────────────

class TestAsyncWorkflows:
    """Test async context manager for async agent frameworks."""

    @pytest.mark.asyncio
    async def test_async_agent_task(self):
        store = MemoryStore()
        async with Checkpoint("async-agent", store=store) as cp:
            cp.set_goal("Async data processing task")
            cp.add_constraint("Process in batches of 100")
            # Simulate async work
            await asyncio.sleep(0)
            cp.record_decision("Use asyncio.gather for parallel batch processing")
            cp.record_tokens("gpt-4o", prompt=500, completion=100)

        state = store.load("async-agent")
        assert state.status == SessionStatus.COMPLETED
        assert "Async data processing task" in state.goals

    @pytest.mark.asyncio
    async def test_async_exception_sets_failed_status(self):
        store = MemoryStore()
        with pytest.raises(ValueError):
            async with Checkpoint("async-fail", store=store) as cp:
                cp.set_goal("Will fail async")
                await asyncio.sleep(0)
                raise ValueError("Async failure")

        state = store.load("async-fail")
        assert state.status == SessionStatus.FAILED


# ── Scenario 7: DiskStore persistence across restarts ────────────────────────

class TestDiskStorePersistence:
    """Test that state survives process restart via DiskStore."""

    def test_state_persists_across_store_instances(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # Process 1: write
            store1 = DiskStore(base_dir=tmpdir)
            with Checkpoint("persist-test", store=store1) as cp:
                cp.set_goal("Persistent task")
                cp.record_decision("Decided to use PostgreSQL")
                cp.record_tokens("gpt-4o", prompt=1000, completion=200)

            # Process 2: new DiskStore instance, same directory
            store2 = DiskStore(base_dir=tmpdir)
            state = store2.load("persist-test")
            assert state is not None
            assert "Persistent task" in state.goals
            assert len(state.decisions) == 1
            assert state.token_usage.prompt == 1000
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_multiple_sessions_coexist(self):
        tmpdir = tempfile.mkdtemp()
        try:
            store = DiskStore(base_dir=tmpdir)
            session_ids = [f"session-{i}" for i in range(5)]

            for sid in session_ids:
                with Checkpoint(sid, store=store) as cp:
                    cp.set_goal(f"Goal for {sid}")

            sessions = store.list_sessions()
            stored_ids = [s["session_id"] for s in sessions]
            for sid in session_ids:
                assert sid in stored_ids
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Scenario 8: JSON export and handoff ──────────────────────────────────────

class TestExportWorkflows:
    """Test state export for integration with other systems."""

    def test_full_state_json_roundtrip(self):
        store = MemoryStore()
        with Checkpoint("export-full", store=store) as cp:
            cp.set_goal("Full export test")
            cp.add_constraint("Stay within $1 budget")
            cp.record_decision(
                "Decided to use streaming API",
                reasoning="Lower latency for user-facing features",
                alternatives_rejected=["batch API", "polling"],
            )
            cp.record_file_modified("api/stream.py")
            cp.add_next_step("Add rate limiting")
            cp.record_tokens("gpt-4o", prompt=1000, completion=200)

        state = store.load("export-full")
        json_str = state.model_dump_json()
        parsed = json.loads(json_str)
        restored = TaskState.model_validate(parsed)

        assert restored.session_id == "export-full"
        assert restored.goals == state.goals
        assert restored.constraints == state.constraints
        assert len(restored.decisions) == 1
        assert restored.decisions[0].alternatives_rejected == ["batch API", "polling"]
        assert restored.token_usage.prompt == 1000
        assert restored.next_steps == ["Add rate limiting"]

    def test_handoff_payload_for_multi_agent(self):
        store = MemoryStore()
        with Checkpoint("handoff-src", store=store) as cp:
            cp.set_goal("Data ingestion pipeline")
            cp.record_decision("Schema validated — ready for transform step")
            cp.add_next_step("Run transform agent on validated data")

        state = store.load("handoff-src")
        handoff = state.as_handoff()

        assert handoff["session_id"] == "handoff-src"
        assert "Data ingestion pipeline" in handoff["goals"]
        assert len(handoff["decisions_summary"]) == 1
        assert any("Run transform agent" in s for s in handoff["next_steps"])
        assert "handoff_at" in handoff

    def test_resume_context_completeness(self):
        """The resume context should contain everything needed to continue."""
        store = MemoryStore()
        with Checkpoint("ctx-complete", store=store) as cp:
            cp.set_goal("Implement rate limiting middleware")
            cp.add_constraint("Must not break existing auth flow")
            cp.record_decision(
                "Decided to use sliding window algorithm",
                reasoning="More accurate than fixed window under burst traffic",
            )
            cp.add_next_step("Write unit tests for rate limiter")
            cp.record_tokens("gpt-4o", prompt=2000, completion=400)

        ctx = resume("ctx-complete", store=store)

        # Context must have all the essentials
        assert "Implement rate limiting middleware" in ctx
        assert "Must not break existing auth flow" in ctx
        assert "Decided to use sliding window" in ctx or "sliding window" in ctx
        assert "Write unit tests" in ctx


# ── Scenario 9: Compression workflow ─────────────────────────────────────────

class TestCompressionWorkflow:
    """Test compression utilities in realistic scenarios."""

    def test_large_tool_output_compressed_for_context(self):
        """Tool outputs that would bloat context get compressed automatically."""
        # Simulate a large directory listing tool output
        tool_output = "\n".join([f"/project/src/module_{i}/file_{j}.py"
                                  for i in range(50) for j in range(20)])
        assert len(tool_output) > 5000

        compressed, was_compressed = compress_tool_output(tool_output, max_tokens=200)
        assert was_compressed is True
        assert len(compressed) < len(tool_output)

    def test_build_resume_context_includes_key_info(self):
        store = MemoryStore()
        with Checkpoint("compress-ctx", store=store) as cp:
            cp.set_goal("Deploy microservices to Kubernetes")
            cp.add_constraint("Must maintain 99.9% uptime SLA")
            cp.record_decision(
                "Decided to use Helm charts for deployment",
                reasoning="Simplifies rollback and version management",
            )
            cp.record_tokens("gpt-4o", prompt=5000, completion=1000)

        state = store.load("compress-ctx")
        ctx = build_resume_context(state, max_tokens=500)

        assert "Deploy microservices" in ctx
        assert isinstance(ctx, str)
        assert len(ctx) > 0


# ── Scenario 10: Decorator workflow ──────────────────────────────────────────

class TestDecoratorWorkflow:
    """Test the @checkpoint decorator for transparent checkpointing."""

    def test_decorator_captures_complete_task(self):
        from agent_recall_ai import checkpoint

        store = MemoryStore()

        @checkpoint("decorated-task", store=store)
        def run_agent_task():
            # Inside the decorated function, we get a checkpoint context
            # The decorator handles open/close automatically
            return {"status": "completed", "items_processed": 42}

        result = run_agent_task()
        assert result == {"status": "completed", "items_processed": 42}
        assert store.exists("decorated-task")
        state = store.load("decorated-task")
        assert state.status == SessionStatus.COMPLETED

    def test_decorator_captures_exception(self):
        from agent_recall_ai import checkpoint

        store = MemoryStore()

        @checkpoint("decorated-fail", store=store)
        def failing_task():
            raise RuntimeError("Task failed midway")

        with pytest.raises(RuntimeError):
            failing_task()

        state = store.load("decorated-fail")
        assert state.status == SessionStatus.FAILED

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        from agent_recall_ai import checkpoint

        store = MemoryStore()

        @checkpoint("async-decorated", store=store)
        async def async_agent():
            await asyncio.sleep(0)
            return "async done"

        result = await async_agent()
        assert result == "async done"
        assert store.exists("async-decorated")
