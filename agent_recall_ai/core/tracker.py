"""
Real-time token and cost tracker.

Tracks token usage across LLM calls and computes running cost estimates.
Framework-agnostic — update it from any callback or wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Cost per 1M tokens (USD) — updated June 2025
_MODEL_COSTS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 5.00, "output": 15.00, "cached": 2.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached": 0.075},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "cached": 5.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "cached": 0.25},
    "o1": {"input": 15.00, "output": 60.00, "cached": 7.50},
    "o1-mini": {"input": 3.00, "output": 12.00, "cached": 1.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached": 0.55},
    # Anthropic
    "claude-opus-4-7": {"input": 15.00, "output": 75.00, "cached": 1.50},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cached": 0.30},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00, "cached": 0.08},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cached": 0.30},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00, "cached": 0.08},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "cached": 0.025},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "cached": 0.3125},
}

# Context window sizes (in tokens)
_MODEL_CONTEXT: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
}


@dataclass
class UsageSnapshot:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenCostTracker:
    """
    Tracks cumulative token usage and cost across all LLM calls in a session.
    Thread-safe via simple append-only design (no shared mutable state per call).
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._snapshots: list[UsageSnapshot] = []

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        model: Optional[str] = None,
    ) -> float:
        """Record a single LLM call. Returns the cost of this call in USD."""
        m = model or self.model
        cost = self._compute_cost(m, prompt_tokens, completion_tokens, cached_tokens)
        snap = UsageSnapshot(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            call_count=1,
        )
        self._snapshots.append(snap)
        return cost

    def total(self) -> UsageSnapshot:
        total = UsageSnapshot()
        for s in self._snapshots:
            total.prompt_tokens += s.prompt_tokens
            total.completion_tokens += s.completion_tokens
            total.cached_tokens += s.cached_tokens
            total.cost_usd += s.cost_usd
            total.call_count += s.call_count
        return total

    def context_utilization(self, current_prompt_tokens: int, model: Optional[str] = None) -> float:
        """Returns 0.0–1.0 how full the current context window is."""
        m = model or self.model
        limit = _MODEL_CONTEXT.get(m, 128_000)
        return min(current_prompt_tokens / limit, 1.0)

    def estimated_remaining_calls(self, avg_tokens_per_call: Optional[int] = None) -> int:
        """Rough estimate of how many more LLM calls fit in the context window."""
        t = self.total()
        if t.call_count == 0:
            return 999
        avg = avg_tokens_per_call or (t.total_tokens // t.call_count)
        if avg == 0:
            return 999
        limit = _MODEL_CONTEXT.get(self.model, 128_000)
        used = t.prompt_tokens
        remaining_tokens = limit - used
        return max(0, remaining_tokens // avg)

    @staticmethod
    def _compute_cost(model: str, prompt: int, completion: int, cached: int) -> float:
        costs = _MODEL_COSTS.get(model)
        if costs is None:
            # Unknown model — use a conservative estimate
            costs = {"input": 5.00, "output": 15.00, "cached": 2.50}
        non_cached_prompt = max(0, prompt - cached)
        return (
            non_cached_prompt * costs["input"]
            + completion * costs["output"]
            + cached * costs.get("cached", costs["input"] * 0.5)
        ) / 1_000_000

    @staticmethod
    def context_limit(model: str) -> int:
        return _MODEL_CONTEXT.get(model, 128_000)

    @staticmethod
    def list_models() -> list[str]:
        return sorted(_MODEL_COSTS.keys())
