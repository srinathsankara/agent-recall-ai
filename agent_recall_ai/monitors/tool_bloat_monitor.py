"""
Tool bloat monitor.

Detects when tool call outputs are consuming excessive context tokens.
Triggers automatic compression of bloated tool results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.compressor import compress_tool_output
from ..core.state import AlertSeverity, AlertType
from .base import BaseMonitor

if TYPE_CHECKING:
    from ..core.state import TaskState


class ToolBloatMonitor(BaseMonitor):
    """
    Fires when individual tool outputs are unusually large, or when the cumulative
    token cost of all tool outputs is disproportionate.

    Args:
        max_output_tokens: Max acceptable tokens for a single tool output (default 1000).
        max_tool_fraction: Max fraction of total context that tool outputs should consume (default 0.40).
        auto_compress: If True, compresses bloated outputs in-place (default True).
    """

    def __init__(
        self,
        max_output_tokens: int = 1000,
        max_tool_fraction: float = 0.40,
        auto_compress: bool = True,
    ) -> None:
        self.max_output_tokens = max_output_tokens
        self.max_tool_fraction = max_tool_fraction
        self.auto_compress = auto_compress
        self._alerted_tools: set[str] = set()
        self._fraction_alerted: bool = False

    def check(self, state: TaskState) -> list[dict]:
        return []

    def on_tool_call(self, state: TaskState) -> list[dict]:
        if not state.tool_calls:
            return []

        alerts: list[dict] = []
        last_call = state.tool_calls[-1]

        # Check single tool output size
        if last_call.output_tokens > self.max_output_tokens and not last_call.compressed:
            alert_key = f"{last_call.tool_name}:{last_call.timestamp.isoformat()}"
            if alert_key not in self._alerted_tools:
                self._alerted_tools.add(alert_key)

                if self.auto_compress:
                    compressed, was_compressed = compress_tool_output(
                        last_call.output_summary,
                        max_tokens=self.max_output_tokens,
                    )
                    if was_compressed:
                        last_call.output_summary = compressed
                        last_call.compressed = True
                        last_call.output_tokens = len(compressed) // 4

                alerts.append({
                    "alert_type": AlertType.TOOL_BLOAT,
                    "severity": AlertSeverity.WARN,
                    "message": (
                        f"Tool '{last_call.tool_name}' output is {last_call.output_tokens} tokens "
                        f"(>{self.max_output_tokens} limit). "
                        + ("Auto-compressed." if self.auto_compress else "Consider compressing.")
                    ),
                    "detail": {
                        "tool_name": last_call.tool_name,
                        "output_tokens": last_call.output_tokens,
                        "max_output_tokens": self.max_output_tokens,
                        "compressed": self.auto_compress,
                    },
                })

        # Check cumulative tool fraction (fire only once per crossing)
        if state.token_usage.total > 0 and not self._fraction_alerted:
            tool_tokens = sum(tc.output_tokens for tc in state.tool_calls)
            fraction = tool_tokens / max(state.token_usage.total, 1)
            if fraction > self.max_tool_fraction:
                self._fraction_alerted = True
                alerts.append({
                    "alert_type": AlertType.TOOL_BLOAT,
                    "severity": AlertSeverity.WARN,
                    "message": (
                        f"Tool outputs consuming {fraction*100:.0f}% of context "
                        f"({tool_tokens:,} of {state.token_usage.total:,} tokens). "
                        "Consider compressing older tool results."
                    ),
                    "detail": {
                        "tool_fraction": fraction,
                        "tool_tokens": tool_tokens,
                        "total_tokens": state.token_usage.total,
                    },
                })

        return alerts
