"""
Token pressure monitor.

Fires WARN when context utilization passes warn_at.
Fires CRITICAL at compress_at — triggering proactive conversation compression.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..core.state import AlertSeverity, AlertType
from ..core.tracker import TokenCostTracker
from .base import BaseMonitor

if TYPE_CHECKING:
    from ..core.state import TaskState


class TokenMonitor(BaseMonitor):
    """
    Monitors context window utilization and triggers compression before hitting the limit.

    Args:
        model: LLM model name (used to look up context window size).
        warn_at: Utilization fraction for a WARN alert (default 0.75).
        compress_at: Utilization fraction to trigger proactive compression (default 0.88).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        warn_at: float = 0.75,
        compress_at: float = 0.88,
    ) -> None:
        self.model = model
        self.warn_at = warn_at
        self.compress_at = compress_at
        self._warn_fired = False
        self._compress_fired = False
        self._context_limit = TokenCostTracker.context_limit(model)

    def check(self, state: "TaskState") -> list[dict]:
        alerts: list[dict] = []
        util = state.context_utilization

        if util >= self.compress_at and not self._compress_fired:
            self._compress_fired = True
            alerts.append({
                "alert_type": AlertType.TOKEN_CRITICAL,
                "severity": AlertSeverity.ERROR,
                "message": (
                    f"Context at {util*100:.0f}% ({state.token_usage.total:,} tokens). "
                    "Proactive compression triggered. Checkpoint saved."
                ),
                "detail": {
                    "utilization": util,
                    "total_tokens": state.token_usage.total,
                    "context_limit": self._context_limit,
                    "compression_triggered": True,
                },
            })

        elif util >= self.warn_at and not self._warn_fired:
            self._warn_fired = True
            remaining_tokens = int(self._context_limit * (1 - util))
            alerts.append({
                "alert_type": AlertType.TOKEN_PRESSURE,
                "severity": AlertSeverity.WARN,
                "message": (
                    f"Context at {util*100:.0f}% — "
                    f"{remaining_tokens:,} tokens remaining before limit."
                ),
                "detail": {
                    "utilization": util,
                    "total_tokens": state.token_usage.total,
                    "remaining_tokens": remaining_tokens,
                    "context_limit": self._context_limit,
                },
            })

        return alerts

    on_tokens = check
