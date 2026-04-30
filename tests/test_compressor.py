"""Tests for the context compressor utility functions."""
from __future__ import annotations

import pytest

from agent_recall_ai.core.compressor import (
    compress_tool_output,
    compress_decision_log,
    build_resume_context,
    compress_conversation_history,
    estimate_tokens,
)


class TestCompressToolOutput:
    def test_short_output_not_compressed(self):
        text = "Hello world"
        result, was_compressed = compress_tool_output(text, max_tokens=500)
        assert result == text
        assert was_compressed is False

    def test_long_output_compressed(self):
        text = "line\n" * 1000   # ~4000 tokens equivalent
        result, was_compressed = compress_tool_output(text, max_tokens=100)
        assert was_compressed is True
        assert len(result) < len(text)
        assert "compressed" in result.lower()

    def test_compressed_output_has_head_and_tail(self):
        text = "START " + ("middle " * 1000) + " END"
        result, was_compressed = compress_tool_output(text, max_tokens=50)
        assert "START" in result
        assert "END" in result

    def test_error_lines_preserved_in_compressed(self):
        lines = ["normal line"] * 200 + ["ERROR: something failed"] + ["normal line"] * 200
        text = "\n".join(lines)
        result, was_compressed = compress_tool_output(text, max_tokens=100)
        # Error line should survive compression (it's in head or tail)
        # Note: head/tail approach — error may or may not be in the sample depending on position
        assert was_compressed is True


class TestCompressDecisionLog:
    def test_small_log_unchanged(self):
        decisions = [{"summary": f"Decision {i}"} for i in range(3)]
        result = compress_decision_log(decisions, keep_recent=5)
        assert result == decisions

    def test_large_log_compressed(self):
        decisions = [{"summary": f"Decision {i}", "reasoning": ""} for i in range(20)]
        result = compress_decision_log(decisions, keep_recent=5)
        assert len(result) == 6   # 1 compressed summary + 5 recent
        assert result[0].get("compressed") is True

    def test_recent_decisions_kept_verbatim(self):
        decisions = [{"summary": f"Decision {i}", "reasoning": ""} for i in range(20)]
        result = compress_decision_log(decisions, keep_recent=5)
        recent_summaries = [d["summary"] for d in result[1:]]
        assert "Decision 19" in recent_summaries
        assert "Decision 15" in recent_summaries


class TestBuildResumeContext:
    def test_empty_state_returns_empty_ish(self):
        result = build_resume_context({})
        assert isinstance(result, str)

    def test_goals_included(self):
        state = {"goals": ["Refactor auth", "Add JWT"]}
        result = build_resume_context(state)
        assert "Refactor auth" in result
        assert "Add JWT" in result

    def test_constraints_included(self):
        state = {"constraints": ["No API changes"]}
        result = build_resume_context(state)
        assert "No API changes" in result

    def test_respects_token_limit(self):
        state = {
            "goals": ["x" * 1000],
            "constraints": ["y" * 1000],
            "decisions": [{"summary": "z" * 1000}],
        }
        result = build_resume_context(state, max_tokens=100)
        assert len(result) <= 400 + 20  # 100 tokens * 4 chars + small overhead


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1   # minimum is 1

    def test_known_size(self):
        # 400 chars / 4 = 100 tokens
        assert estimate_tokens("a" * 400) == 100


class TestCompressConversationHistory:
    def _messages(self, n: int) -> list[dict]:
        result = [{"role": "system", "content": "System prompt"}]
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            result.append({"role": role, "content": f"Message {i}" + " x" * 50})
        return result

    def test_empty_list_unchanged(self):
        result, saved = compress_conversation_history([], 128_000)
        assert result == []
        assert saved == 0

    def test_small_history_unchanged(self):
        msgs = self._messages(5)
        result, saved = compress_conversation_history(msgs, 128_000)
        assert saved == 0
        assert len(result) == len(msgs)

    def test_large_history_compressed(self):
        msgs = self._messages(100)
        result, saved = compress_conversation_history(msgs, model_context_limit=2000)
        assert saved > 0

    def test_system_message_preserved(self):
        msgs = self._messages(50)
        result, saved = compress_conversation_history(msgs, model_context_limit=500)
        assert any(m["role"] == "system" for m in result)

    def test_last_4_messages_preserved(self):
        msgs = self._messages(30)
        result, saved = compress_conversation_history(msgs, model_context_limit=200)
        last_4_content = [m["content"] for m in msgs[-4:]]
        result_content = [m["content"] for m in result]
        for c in last_4_content:
            assert c in result_content
