"""
Core state models for agent-recall-ai.

TaskState is the canonical checkpoint record — everything needed to resume
a dead session cold, with no prior context.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"   # paused intentionally


class AlertSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    COST_WARNING = "cost_warning"
    COST_EXCEEDED = "cost_exceeded"
    TOKEN_PRESSURE = "token_pressure"
    TOKEN_CRITICAL = "token_critical"
    TOOL_BLOAT = "tool_bloat"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    PACKAGE_HALLUCINATION = "package_hallucination"
    COMPRESSION_TRIGGERED = "compression_triggered"


class Decision(BaseModel):
    """A single decision made during the session, with reasoning."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    summary: str
    reasoning: str = ""
    alternatives_rejected: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    tags: list[str] = Field(default_factory=list)


class FileChange(BaseModel):
    """A file touched during the session."""
    path: str
    action: str = "modified"    # created | modified | deleted | read
    description: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class ToolCall(BaseModel):
    """A recorded tool invocation — used for bloat detection and replay."""
    tool_name: str
    input_summary: str = ""
    output_summary: str = ""
    output_tokens: int = 0
    compressed: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class Alert(BaseModel):
    """A real-time warning raised by a monitor."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    acknowledged: bool = False


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    cached: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def add(self, prompt: int = 0, completion: int = 0, cached: int = 0) -> None:
        self.prompt += prompt
        self.completion += completion
        self.cached += cached


class TaskState(BaseModel):
    """
    The full checkpoint record for an agent session.

    Designed to be self-contained — reading this alone should be enough
    to resume the task in a brand-new context window.
    """
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    status: SessionStatus = SessionStatus.ACTIVE

    # What the agent is trying to accomplish
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    context_summary: str = ""     # optional freeform context (e.g. "fixing auth module in FastAPI app")

    # What has been done
    decisions: list[Decision] = Field(default_factory=list)
    files_modified: list[FileChange] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # What remains
    next_steps: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    # Real-time telemetry
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    context_utilization: float = 0.0   # 0.0–1.0, current fill level

    # Monitor alerts raised this session
    alerts: list[Alert] = Field(default_factory=list)

    # Arbitrary metadata (model name, framework, user-defined tags)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Checkpoint sequence number — increments on each save
    checkpoint_seq: int = 0

    # Schema version — stamped by VersionedSchema before serialization
    schema_version: Optional[str] = None

    def add_decision(
        self,
        summary: str,
        reasoning: str = "",
        alternatives_rejected: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> Decision:
        d = Decision(
            summary=summary,
            reasoning=reasoning,
            alternatives_rejected=alternatives_rejected or [],
            tags=tags or [],
        )
        self.decisions.append(d)
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return d

    def add_file(self, path: str, action: str = "modified", description: str = "") -> FileChange:
        fc = FileChange(path=path, action=action, description=description)
        self.files_modified.append(fc)
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return fc

    def add_tool_call(
        self,
        tool_name: str,
        input_summary: str = "",
        output_summary: str = "",
        output_tokens: int = 0,
    ) -> ToolCall:
        tc = ToolCall(
            tool_name=tool_name,
            input_summary=input_summary,
            output_summary=output_summary,
            output_tokens=output_tokens,
        )
        self.tool_calls.append(tc)
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return tc

    def add_alert(
        self, alert_type: AlertType, severity: AlertSeverity, message: str, detail: Optional[dict] = None
    ) -> Alert:
        a = Alert(alert_type=alert_type, severity=severity, message=message, detail=detail or {})
        self.alerts.append(a)
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        return a

    def resume_prompt(self) -> str:
        """
        Generate a structured resume prompt — paste this into a new session
        to restore full context without rehashing the conversation.
        """
        lines: list[str] = [
            "## Resuming Agent Session",
            f"**Session:** {self.session_id}  |  **Checkpoint:** #{self.checkpoint_seq}",
            f"**Started:** {self.created_at.strftime('%Y-%m-%d %H:%M')} UTC",
            "",
        ]

        if self.goals:
            lines.append("### Goals")
            for g in self.goals:
                lines.append(f"- {g}")
            lines.append("")

        if self.constraints:
            lines.append("### Active Constraints")
            for c in self.constraints:
                lines.append(f"- {c}")
            lines.append("")

        if self.context_summary:
            lines.append("### Context")
            lines.append(self.context_summary)
            lines.append("")

        if self.decisions:
            lines.append("### Decisions Made")
            for d in self.decisions[-10:]:   # last 10 to stay concise
                lines.append(f"- **{d.summary}**")
                if d.reasoning:
                    lines.append(f"  Reason: {d.reasoning}")
                if d.alternatives_rejected:
                    lines.append(f"  Rejected: {', '.join(d.alternatives_rejected)}")
            lines.append("")

        if self.files_modified:
            lines.append("### Files Modified")
            unique_files = {fc.path for fc in self.files_modified}
            for path in sorted(unique_files):
                lines.append(f"- `{path}`")
            lines.append("")

        if self.next_steps:
            lines.append("### Next Steps")
            for step in self.next_steps:
                lines.append(f"- {step}")
            lines.append("")

        if self.open_questions:
            lines.append("### Open Questions")
            for q in self.open_questions:
                lines.append(f"- {q}")
            lines.append("")

        lines.append(
            f"**Token usage so far:** {self.token_usage.total:,} tokens  |  "
            f"**Cost:** ${self.cost_usd:.4f}"
        )

        return "\n".join(lines)

    def as_handoff(self) -> dict[str, Any]:
        """Export as a structured multi-agent handoff payload."""
        return {
            "session_id": self.session_id,
            "handoff_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "goals": self.goals,
            "constraints": self.constraints,
            "context_summary": self.context_summary,
            "decisions_summary": [
                {"summary": d.summary, "reasoning": d.reasoning}
                for d in self.decisions
            ],
            "files_modified": [fc.path for fc in self.files_modified],
            "next_steps": self.next_steps,
        }
