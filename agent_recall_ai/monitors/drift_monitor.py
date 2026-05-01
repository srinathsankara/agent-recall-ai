"""
Behavioral drift monitor.

Detects when agent decisions may be drifting away from session constraints.
Uses keyword/pattern matching against the constraint list — no LLM required.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.state import AlertSeverity, AlertType
from .base import BaseMonitor

if TYPE_CHECKING:
    from ..core.state import TaskState


# Patterns that suggest a constraint may be violated
_VIOLATION_SIGNALS = [
    (r"\bpublic\s+api\b.*\bchange[ds]?\b", "public API modification detected"),
    (r"\bbreak.*compat", "breaking compatibility detected"),
    (r"\bdelete\b.*\bwithout\b.*\bconfirm", "deletion without confirmation"),
    (r"\bdrop\b.*\btable\b", "database table drop detected"),
    (r"\bforce\s+push\b", "force push detected"),
    (r"\broot\b.*\baccess\b", "root/elevated access attempted"),
    (r"\bprod(uction)?\b.*\bdeploy", "production deployment detected"),
    (r"\boverwrite\b.*\bwithout\b", "overwrite without check detected"),
]


class DriftMonitor(BaseMonitor):
    """
    Monitors decisions against session constraints and fires alerts when drift is detected.

    Args:
        sensitivity: 'low', 'medium', 'high' — controls how aggressively to scan.
    """

    def __init__(self, sensitivity: str = "medium") -> None:
        self.sensitivity = sensitivity
        self._alerted_decisions: set[str] = set()

    def check(self, state: TaskState) -> list[dict]:
        if not state.constraints:
            return []

        alerts: list[dict] = []
        # Scan recent decisions (last 5) for constraint violations
        recent_decisions = state.decisions[-5:]

        for decision in recent_decisions:
            if decision.id in self._alerted_decisions:
                continue

            violation = self._check_decision_against_constraints(
                decision.summary + " " + decision.reasoning,
                state.constraints,
            )
            if violation:
                self._alerted_decisions.add(decision.id)
                alerts.append({
                    "alert_type": AlertType.BEHAVIORAL_DRIFT,
                    "severity": AlertSeverity.WARN,
                    "message": (
                        f"Decision may violate constraint: {violation}. "
                        f"Decision: '{decision.summary[:80]}'"
                    ),
                    "detail": {
                        "decision_id": decision.id,
                        "decision_summary": decision.summary,
                        "violation_pattern": violation,
                        "constraints": state.constraints,
                    },
                })

        return alerts

    def _check_decision_against_constraints(
        self, decision_text: str, constraints: list[str]
    ) -> str | None:
        """Returns a violation description string, or None if no violation."""
        text_lower = decision_text.lower()

        # Check built-in signal patterns
        for pattern, description in _VIOLATION_SIGNALS:
            if re.search(pattern, text_lower):
                # Check if any constraint would be violated
                for constraint in constraints:
                    constraint_lower = constraint.lower()
                    # Simple heuristic: if constraint mentions the same domain as the signal
                    if any(
                        keyword in constraint_lower
                        for keyword in ["public api", "compat", "delete", "drop", "prod", "deploy"]
                        if keyword in text_lower
                    ):
                        return description

        # Check constraints directly against decision text
        for constraint in constraints:
            # Extract key nouns from constraint and check if decision contradicts them
            constraint_lower = constraint.lower()
            if "do not" in constraint_lower or "never" in constraint_lower or "must not" in constraint_lower:
                # Extract the forbidden action
                forbidden = re.sub(r"(do not|never|must not)\s+", "", constraint_lower).strip()
                forbidden_words = [w for w in forbidden.split() if len(w) > 3]
                if self.sensitivity in ("medium", "high"):
                    match_count = sum(1 for w in forbidden_words if w in text_lower)
                    if match_count >= max(1, len(forbidden_words) // 2):
                        return f"potential violation of: '{constraint[:60]}'"

        return None

    on_decision = check
