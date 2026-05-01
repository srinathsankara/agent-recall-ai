"""Base class for all real-time monitors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.state import TaskState


class BaseMonitor(ABC):
    """
    A monitor observes session state and fires alerts when thresholds are crossed.

    Monitors are called:
    - after each token usage update (on_tokens)
    - after each tool call (on_tool_call)
    - after each decision (on_decision)
    - on explicit check() calls
    """

    @abstractmethod
    def check(self, state: TaskState) -> list[dict]:
        """
        Inspect state and return a list of alert dicts to attach.
        Each dict must include: alert_type, severity, message, detail.
        """
        ...

    def on_tokens(self, state: TaskState) -> list[dict]:
        return self.check(state)

    def on_tool_call(self, state: TaskState) -> list[dict]:
        return []

    def on_decision(self, state: TaskState) -> list[dict]:
        return []
