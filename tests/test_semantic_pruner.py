"""Tests for the SemanticPruner — especially decision anchor protection."""
from __future__ import annotations

from agent_recall_ai.core.semantic_pruner import (
    SemanticPruner,
    _estimate_tokens,
    _is_decision_anchor,
    _keyword_score,
)


class TestDecisionAnchorDetection:
    def test_decided_is_anchor(self):
        assert _is_decision_anchor("We decided to use PostgreSQL over MongoDB")

    def test_rejected_is_anchor(self):
        assert _is_decision_anchor("Rejected the Redis approach because of cost")

    def test_architecture_is_anchor(self):
        assert _is_decision_anchor("This is an architecture decision")

    def test_because_is_anchor(self):
        assert _is_decision_anchor("Chose FastAPI because it has better async support")

    def test_must_not_is_anchor(self):
        assert _is_decision_anchor("Must not change the public API surface")

    def test_constraint_is_anchor(self):
        assert _is_decision_anchor("Constraint: no external dependencies")

    def test_plain_message_not_anchor(self):
        assert not _is_decision_anchor("The current directory has 42 files")
        assert not _is_decision_anchor("Running pip install requests")
        assert not _is_decision_anchor("ls -la /home/user")


class TestKeywordScore:
    def test_system_scores_max(self):
        assert _keyword_score("anything", "system") == 1.0

    def test_error_boosts_score(self):
        score = _keyword_score("ERROR: connection failed", "assistant")
        assert score > 0.5

    def test_tool_role_penalised(self):
        score = _keyword_score("directory listing output", "tool")
        assert score < 0.3

    def test_long_tool_output_penalised(self):
        long_text = "a" * 5000
        score = _keyword_score(long_text, "tool")
        assert score <= 0.3

    def test_decision_content_boosted(self):
        score = _keyword_score("We decided to use Kubernetes for orchestration", "assistant")
        assert score > 0.5


class TestSemanticPruner:
    """Tests run without sentence-transformers (keyword scoring path)."""

    def setup_method(self):
        # Force keyword-only mode so tests don't require GPU/internet
        self.pruner = SemanticPruner(use_embeddings=False)

    def _make_messages(self, n: int) -> list[dict]:
        messages = []
        messages.append({"role": "system", "content": "You are a helpful assistant"})
        for i in range(n - 1):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}: " + "x" * 100})
        return messages

    def test_compress_empty(self):
        result, stats = self.pruner.compress_context([])
        assert result == []
        assert stats["original_tokens"] == 0

    def test_system_message_never_pruned(self):
        msgs = self._make_messages(30)
        result, stats = self.pruner.compress_context(msgs, model_context_limit=2000)
        roles = [m["role"] for m in result]
        assert "system" in roles

    def test_anchor_messages_never_pruned(self):
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "We decided to use PostgreSQL because of ACID"},
            *[{"role": "user", "content": "x" * 200} for _ in range(20)],
            {"role": "assistant", "content": "rejected MongoDB as too complex"},
        ]
        result, stats = self.pruner.compress_context(msgs, model_context_limit=1000)
        all_content = " ".join(m["content"] for m in result)
        assert "PostgreSQL" in all_content
        assert "MongoDB" in all_content

    def test_last_messages_always_kept(self):
        msgs = self._make_messages(20)
        result, stats = self.pruner.compress_context(msgs, model_context_limit=500)
        # Last 4 messages (min_messages_kept default) should be in output
        last_4 = [m["content"] for m in msgs[-4:]]
        result_content = [m["content"] for m in result]
        for content in last_4:
            assert content in result_content

    def test_tokens_actually_reduced(self):
        msgs = self._make_messages(50)
        _, stats = self.pruner.compress_context(msgs, model_context_limit=5000)
        assert stats["compressed_tokens"] <= stats["original_tokens"]

    def test_stats_dict_keys(self):
        msgs = self._make_messages(10)
        _, stats = self.pruner.compress_context(msgs)
        required_keys = {
            "original_tokens", "compressed_tokens", "tokens_saved",
            "messages_kept", "messages_pruned", "anchors_protected",
        }
        assert required_keys.issubset(stats.keys())

    def test_small_history_not_changed(self):
        msgs = self._make_messages(3)
        result, stats = self.pruner.compress_context(msgs, model_context_limit=128_000)
        # Under budget — should keep everything
        assert stats["messages_pruned"] == 0

    def test_extract_decision_log(self):
        msgs = [
            {"role": "assistant", "content": "We decided to use Redis for caching"},
            {"role": "user", "content": "okay"},
            {"role": "assistant", "content": "The directory has 42 files"},
            {"role": "assistant", "content": "Rejected DynamoDB because of vendor lock-in"},
        ]
        log = self.pruner.extract_decision_log(msgs)
        assert any("Redis" in entry for entry in log)
        assert any("DynamoDB" in entry for entry in log)
        # Plain messages should not be in the log
        assert not any("42 files" in entry for entry in log)

    def test_score_messages_returns_correct_count(self):
        msgs = self._make_messages(10)
        scored = self.pruner.score_messages(msgs)
        assert len(scored) == 10

    def test_anchor_scores_one(self):
        msgs = [
            {"role": "assistant", "content": "We decided to use JWT because of statelessness"}
        ]
        scored = self.pruner.score_messages(msgs)
        assert scored[0].is_anchor is True
        assert scored[0].score == 1.0

    def test_token_estimate(self):
        assert _estimate_tokens("hello") == 1
        assert _estimate_tokens("a" * 400) == 100
