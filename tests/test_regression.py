"""
Regression tests — one test per bug that was found and fixed.

Each test is named after the bug it prevents from regressing.
If any of these fail, a previously-fixed bug has been reintroduced.
"""
from __future__ import annotations

from agent_recall_ai import Checkpoint, resume
from agent_recall_ai.core.compressor import build_resume_context, compress_tool_output
from agent_recall_ai.core.semantic_pruner import SemanticPruner
from agent_recall_ai.core.state import TaskState
from agent_recall_ai.monitors import TokenMonitor, ToolBloatMonitor
from agent_recall_ai.privacy.redactor import PIIRedactor
from agent_recall_ai.storage.memory import MemoryStore

# ── Bug 1: record_tokens() positional model arg ───────────────────────────────

class TestRecordTokensModelArg:
    """
    Bug: record_tokens("gpt-4o", prompt=N) crashed with
    'got multiple values for argument prompt' because model was the last param.
    Fix: moved model to first positional param.
    """

    def test_record_tokens_model_first_positional(self):
        store = MemoryStore()
        with Checkpoint("rt-pos", store=store) as cp:
            cp.record_tokens("gpt-4o", prompt=1000, completion=200)
        state = store.load("rt-pos")
        assert state.token_usage.prompt == 1000
        assert state.token_usage.completion == 200

    def test_record_tokens_model_keyword(self):
        store = MemoryStore()
        with Checkpoint("rt-kw", store=store) as cp:
            cp.record_tokens(model="gpt-4o", prompt=500, completion=100)
        state = store.load("rt-kw")
        assert state.token_usage.prompt == 500

    def test_record_tokens_no_model_uses_default(self):
        store = MemoryStore()
        with Checkpoint("rt-nomodel", store=store, model="claude-3-5-haiku-20241022") as cp:
            cp.record_tokens(prompt=100, completion=50)
        state = store.load("rt-nomodel")
        assert state.token_usage.prompt == 100
        assert state.cost_usd > 0  # cost should be calculated from default model

    def test_record_tokens_returns_cost_float(self):
        store = MemoryStore()
        with Checkpoint("rt-cost", store=store) as cp:
            cost = cp.record_tokens("gpt-4o", prompt=1000, completion=200)
        assert isinstance(cost, float)
        assert cost > 0


# ── Bug 2 & 3: resume() return type ─────────────────────────────────────────

class TestResumeReturnType:
    """
    Bug: resume() returned TaskState | None instead of a context string.
    Fix: resume() now returns the resume prompt string (empty string if not found).
    """

    def test_resume_returns_string(self):
        store = MemoryStore()
        with Checkpoint("rv-str", store=store) as cp:
            cp.set_goal("Test goal")
        ctx = resume("rv-str", store=store)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_resume_contains_goal(self):
        store = MemoryStore()
        with Checkpoint("rv-goal", store=store) as cp:
            cp.set_goal("Refactor authentication module")
        ctx = resume("rv-goal", store=store)
        assert "Refactor authentication module" in ctx

    def test_resume_contains_decision(self):
        store = MemoryStore()
        with Checkpoint("rv-dec", store=store) as cp:
            cp.record_decision("Use PyJWT", reasoning="Well maintained library")
        ctx = resume("rv-dec", store=store)
        assert "Use PyJWT" in ctx

    def test_resume_nonexistent_returns_empty_string(self):
        store = MemoryStore()
        ctx = resume("no-such-session", store=store)
        assert ctx == ""
        assert isinstance(ctx, str)


# ── Bug 4: TaskState.as_resume_context() alias ───────────────────────────────

class TestAsResumeContextAlias:
    """
    Bug: TaskState.as_resume_context() didn't exist; method was resume_prompt().
    Fix: added as_resume_context() as an alias.
    """

    def test_as_resume_context_exists(self):
        state = TaskState(session_id="arc-test")
        assert hasattr(state, "as_resume_context")

    def test_as_resume_context_returns_string(self):
        state = TaskState(session_id="arc-str")
        state.goals = ["Test goal"]
        ctx = state.as_resume_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_as_resume_context_matches_resume_prompt(self):
        state = TaskState(session_id="arc-match")
        state.goals = ["Goal A"]
        state.add_decision("Decision B")
        assert state.as_resume_context() == state.resume_prompt()


# ── Bug 5: compress_tool_output max_chars param ──────────────────────────────

class TestCompressToolOutputMaxChars:
    """
    Bug: compress_tool_output(text, max_chars=200) crashed with
    'unexpected keyword argument max_chars'.
    Fix: added max_chars as an accepted param (overrides max_tokens).
    """

    def test_max_chars_param_accepted(self):
        result, was_compressed = compress_tool_output("x" * 1000, max_chars=200)
        assert isinstance(result, str)

    def test_max_chars_actually_limits_output(self):
        long_text = "a" * 2000
        result, was_compressed = compress_tool_output(long_text, max_chars=100)
        assert was_compressed is True
        assert len(result) < len(long_text)

    def test_max_tokens_still_works(self):
        result, was_compressed = compress_tool_output("x" * 10000, max_tokens=100)
        assert was_compressed is True

    def test_short_text_not_compressed(self):
        short = "hello world"
        result, was_compressed = compress_tool_output(short, max_chars=500)
        assert was_compressed is False
        assert result == short


# ── Bug 6: build_resume_context accepts TaskState ────────────────────────────

class TestBuildResumeContextTaskState:
    """
    Bug: build_resume_context(state) only accepted dict, crashed on TaskState.
    Fix: now auto-converts TaskState via model_dump().
    """

    def test_accepts_task_state(self):
        store = MemoryStore()
        with Checkpoint("brc-state", store=store) as cp:
            cp.set_goal("Build something")
            cp.record_decision("Use FastAPI")
        state = store.load("brc-state")
        result = build_resume_context(state)  # TaskState directly
        assert isinstance(result, str)
        assert len(result) > 0

    def test_accepts_dict(self):
        store = MemoryStore()
        with Checkpoint("brc-dict", store=store) as cp:
            cp.set_goal("Dict test")
        state = store.load("brc-dict")
        result = build_resume_context(state.model_dump())
        assert isinstance(result, str)

    def test_contains_goal_from_state(self):
        store = MemoryStore()
        with Checkpoint("brc-goal", store=store) as cp:
            cp.set_goal("Migrate to PostgreSQL")
        state = store.load("brc-goal")
        result = build_resume_context(state)
        assert "Migrate to PostgreSQL" in result


# ── Bug 7: PIIRedactor.redact() alias ────────────────────────────────────────

class TestPIIRedactorRedactAlias:
    """
    Bug: PIIRedactor.redact() didn't exist; method was redact_text().
    Fix: added redact() as a convenience alias returning just the string.
    """

    def test_redact_method_exists(self):
        r = PIIRedactor()
        assert hasattr(r, "redact")

    def test_redact_returns_string(self):
        r = PIIRedactor()
        result = r.redact("some text with no secrets")
        assert isinstance(result, str)

    def test_redact_removes_api_key(self):
        r = PIIRedactor()
        text = "use key sk-proj-ABCDEF123456789012345678 please"
        result = r.redact(text)
        assert "sk-proj-ABCDEF123456789012345678" not in result

    def test_redact_removes_openai_key(self):
        r = PIIRedactor()
        text = "token=sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcd"
        result = r.redact(text)
        assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcd" not in result

    def test_redact_text_still_works(self):
        """Ensure the original method wasn't broken."""
        r = PIIRedactor()
        redacted, categories = r.redact_text("sk-proj-TESTKEY123456789012 here")
        assert isinstance(redacted, str)
        assert isinstance(categories, list)


# ── Bug 8: TokenMonitor critical_at param ────────────────────────────────────

class TestTokenMonitorCriticalAt:
    """
    Bug: TokenMonitor(critical_at=0.9) crashed with unexpected keyword argument.
    Fix: added critical_at as alias for compress_at.
    """

    def test_critical_at_param_accepted(self):
        m = TokenMonitor(warn_at=0.5, critical_at=0.9)
        assert m is not None

    def test_critical_at_sets_compress_threshold(self):
        m = TokenMonitor(critical_at=0.85)
        assert m.compress_at == 0.85

    def test_compress_at_still_works(self):
        m = TokenMonitor(warn_at=0.6, compress_at=0.85)
        assert m.compress_at == 0.85

    def test_critical_at_overrides_compress_at(self):
        m = TokenMonitor(compress_at=0.88, critical_at=0.95)
        assert m.compress_at == 0.95  # critical_at wins


# ── Bug 9: ToolBloatMonitor max_calls param ──────────────────────────────────

class TestToolBloatMonitorMaxCalls:
    """
    Bug: ToolBloatMonitor(max_calls=50) crashed with unexpected keyword argument.
    Fix: added max_calls param.
    """

    def test_max_calls_param_accepted(self):
        m = ToolBloatMonitor(max_calls=50)
        assert m is not None
        assert m.max_calls == 50

    def test_max_calls_does_not_override_max_output_tokens(self):
        """max_calls must NOT clobber max_output_tokens — they are separate concerns."""
        m = ToolBloatMonitor(max_output_tokens=1000, max_calls=50)
        assert m.max_output_tokens == 1000
        assert m.max_calls == 50

    def test_max_output_tokens_still_works(self):
        m = ToolBloatMonitor(max_output_tokens=500)
        assert m.max_output_tokens == 500

    def test_monitor_attaches_to_checkpoint(self):
        store = MemoryStore()
        m = ToolBloatMonitor(max_calls=50)
        with Checkpoint("tbm-test", store=store, monitors=[m]) as cp:
            cp.set_goal("Tool bloat test")
        state = store.load("tbm-test")
        assert state is not None


# ── Bug 10: SemanticPruner compress() and max_tokens ─────────────────────────

class TestSemanticPrunerAPI:
    """
    Bug: SemanticPruner(max_tokens=100) and .compress() didn't exist.
    Fix: added max_tokens init param and compress() as alias for compress_context().
    """

    def test_max_tokens_param_accepted(self):
        pruner = SemanticPruner(max_tokens=1000)
        assert pruner is not None

    def test_compress_method_exists(self):
        pruner = SemanticPruner()
        assert hasattr(pruner, "compress")

    def test_compress_alias_works(self):
        pruner = SemanticPruner()
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        result, stats = pruner.compress(messages)
        assert isinstance(result, list)
        assert isinstance(stats, dict)

    def test_compress_context_still_works(self):
        pruner = SemanticPruner()
        messages = [{"role": "user", "content": "Hello"}]
        result, stats = pruner.compress_context(messages)
        assert isinstance(result, list)

    def test_compress_protects_decision_anchors(self):
        pruner = SemanticPruner(use_embeddings=False)
        messages = [
            {"role": "user", "content": "I decided to use PostgreSQL because ACID compliance"},
        ] + [{"role": "user", "content": f"Filler {i}"} for i in range(50)]
        compressed, stats = pruner.compress(messages)
        texts = [m["content"] for m in compressed]
        assert any("PostgreSQL" in t for t in texts), "Anchor decision was pruned!"
        assert stats["anchors_protected"] >= 1


# ── Bug 11: __main__.py missing (CLI broken) ─────────────────────────────────

class TestCLIEntryPoint:
    """
    Bug: python -m agent_recall_ai failed with 'No module named agent_recall_ai.__main__'.
    Fix: added agent_recall_ai/__main__.py.
    """

    def test_main_module_exists(self):
        import importlib.util
        spec = importlib.util.find_spec("agent_recall_ai.__main__")
        assert spec is not None, "__main__.py is missing from agent_recall_ai package"

    def test_cli_importable(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "agent_recall_ai", "--help"],
            capture_output=True,
        )
        assert r.returncode == 0, f"CLI failed: {r.stderr[:200]}"

    def test_cli_list_command(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "agent_recall_ai", "list"],
            capture_output=True,
        )
        assert r.returncode == 0

    def test_cli_status_command(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "agent_recall_ai", "status"],
            capture_output=True,
        )
        assert r.returncode == 0
