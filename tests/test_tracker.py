"""Tests for the token/cost tracker."""
from __future__ import annotations

from agent_recall_ai.core.tracker import TokenCostTracker


class TestTokenCostTracker:
    def test_zero_on_init(self):
        t = TokenCostTracker(model="gpt-4o-mini")
        total = t.total()
        assert total.prompt_tokens == 0
        assert total.completion_tokens == 0
        assert total.cost_usd == 0.0
        assert total.call_count == 0

    def test_record_accumulates(self):
        t = TokenCostTracker(model="gpt-4o-mini")
        t.record(prompt_tokens=1000, completion_tokens=200)
        t.record(prompt_tokens=500, completion_tokens=100)
        total = t.total()
        assert total.prompt_tokens == 1500
        assert total.completion_tokens == 300
        assert total.call_count == 2

    def test_cost_is_positive(self):
        t = TokenCostTracker(model="gpt-4o-mini")
        cost = t.record(prompt_tokens=10_000, completion_tokens=2_000)
        assert cost > 0

    def test_gpt4o_costs_more_than_mini(self):
        t_mini = TokenCostTracker(model="gpt-4o-mini")
        t_full = TokenCostTracker(model="gpt-4o")
        cost_mini = t_mini.record(prompt_tokens=10_000, completion_tokens=1_000)
        cost_full = t_full.record(prompt_tokens=10_000, completion_tokens=1_000)
        assert cost_full > cost_mini

    def test_cached_tokens_reduce_cost(self):
        t = TokenCostTracker(model="gpt-4o")
        cost_no_cache = t.record(prompt_tokens=10_000, completion_tokens=0, cached_tokens=0)
        t2 = TokenCostTracker(model="gpt-4o")
        cost_with_cache = t2.record(prompt_tokens=10_000, completion_tokens=0, cached_tokens=5_000)
        assert cost_with_cache < cost_no_cache

    def test_context_utilization_within_bounds(self):
        t = TokenCostTracker(model="gpt-4o")
        util = t.context_utilization(64_000, model="gpt-4o")
        assert 0.0 < util < 1.0
        assert abs(util - 0.5) < 0.01   # gpt-4o has 128k context

    def test_context_utilization_caps_at_one(self):
        t = TokenCostTracker(model="gpt-4o")
        util = t.context_utilization(200_000, model="gpt-4o")
        assert util == 1.0

    def test_unknown_model_falls_back(self):
        t = TokenCostTracker(model="some-future-model-xyz")
        cost = t.record(prompt_tokens=1_000, completion_tokens=100)
        assert cost > 0   # should not raise, uses conservative fallback

    def test_context_limit_lookup(self):
        assert TokenCostTracker.context_limit("gpt-4o") == 128_000
        assert TokenCostTracker.context_limit("claude-opus-4-7") == 200_000
        assert TokenCostTracker.context_limit("o1") == 200_000
