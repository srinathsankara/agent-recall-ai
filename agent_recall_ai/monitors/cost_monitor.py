"""
Cost runaway monitor.

Fires WARN when session cost crosses warn_at fraction of budget.
Fires CRITICAL and raises CostBudgetExceeded when cost exceeds budget_usd.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.state import AlertSeverity, AlertType
from .base import BaseMonitor

if TYPE_CHECKING:
    from ..core.state import TaskState


class CostBudgetExceeded(Exception):
    """Raised when a session exceeds its cost budget."""

    def __init__(self, cost_usd: float, budget_usd: float) -> None:
        self.cost_usd = cost_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Session cost ${cost_usd:.4f} exceeds budget ${budget_usd:.2f}. "
            "Checkpoint saved — resume from last checkpoint."
        )


class CostMonitor(BaseMonitor):
    """
    Enforces a per-session USD spend cap.

    Args:
        budget_usd: Hard stop limit in USD (default $5.00).
        warn_at: Fraction of budget at which to emit a warning (default 0.80 → 80%).
        raise_on_exceed: If True, raises CostBudgetExceeded when budget is hit.
    """

    def __init__(
        self,
        budget_usd: float = 5.00,
        warn_at: float = 0.80,
        raise_on_exceed: bool = True,
    ) -> None:
        self.budget_usd = budget_usd
        self.warn_at = warn_at
        self.raise_on_exceed = raise_on_exceed
        self._warned = False
        self._exceeded = False

    def check(self, state: "TaskState") -> list[dict]:
        alerts: list[dict] = []
        cost = state.cost_usd

        if cost >= self.budget_usd and not self._exceeded:
            self._exceeded = True
            self._warned = True   # exceed implies warn — prevent spurious warn on next call
            alerts.append({
                "alert_type": AlertType.COST_EXCEEDED,
                "severity": AlertSeverity.CRITICAL,
                "message": (
                    f"Session cost ${cost:.4f} exceeds budget ${self.budget_usd:.2f}. "
                    "Checkpoint saved automatically."
                ),
                "detail": {
                    "cost_usd": cost,
                    "budget_usd": self.budget_usd,
                    "overage_usd": cost - self.budget_usd,
                },
            })
            if self.raise_on_exceed:
                raise CostBudgetExceeded(cost, self.budget_usd)

        elif cost >= self.budget_usd * self.warn_at and not self._warned:
            self._warned = True
            remaining = self.budget_usd - cost
            alerts.append({
                "alert_type": AlertType.COST_WARNING,
                "severity": AlertSeverity.WARN,
                "message": (
                    f"Session cost ${cost:.4f} ({cost/self.budget_usd*100:.0f}% of budget). "
                    f"${remaining:.4f} remaining."
                ),
                "detail": {
                    "cost_usd": cost,
                    "budget_usd": self.budget_usd,
                    "remaining_usd": remaining,
                    "utilization": cost / self.budget_usd,
                },
            })

        return alerts

    on_tokens = check
