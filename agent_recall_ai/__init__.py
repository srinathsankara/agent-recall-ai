"""
agent-recall-ai — Structured session checkpointing for AI agents.

Survive context limits, cost overruns, and session death.

Quick start (sync):
    from agent_recall_ai import Checkpoint, resume

    with Checkpoint("my-task") as cp:
        cp.set_goal("Refactor authentication module")
        cp.add_constraint("Do not break public API")
        cp.record_decision("Using PyJWT", reasoning="More actively maintained")
        cp.record_file_modified("auth/tokens.py")
        # Auto-saves on exit

Quick start (async):
    async with Checkpoint("my-task") as cp:
        cp.set_goal("Async agent task")
        result = await some_llm_call()

Quick start (decorator):
    @checkpoint("my-task")
    async def run_agent(goal: str, cp=None):
        cp.set_goal(goal)
        ...

    # In a new session:
    state = resume("my-task")
    print(state.resume_prompt())

Thread forking:
    with Checkpoint("main-task") as cp:
        cp.set_goal("Refactor auth module")
        cp.record_decision("Use PyJWT", reasoning="Better maintained")

        # Explore alternative without touching parent
        alt = cp.fork("main-task-alt")
        alt.record_decision("Try python-jose", reasoning="Lighter weight")
        alt.save()

LangGraph drop-in:
    from agent_recall_ai.adapters import LangGraphAdapter
    checkpointer = LangGraphAdapter.from_sqlite("checkpoints.db")
    graph = builder.compile(checkpointer=checkpointer)

OTLP export (Datadog / Jaeger / Grafana):
    from agent_recall_ai.exporters import OTLPExporter
    exporter = OTLPExporter(endpoint="http://localhost:4317")
    exporter.attach(cp)  # auto-exports on every save

Anthropic prompt caching (automatic, 90% cost reduction):
    from agent_recall_ai.adapters import AnthropicAdapter
    adapter = AnthropicAdapter(cp)  # enable_prompt_caching=True by default
    client = adapter.wrap(anthropic.Anthropic())
"""
from .checkpoint import Checkpoint, checkpoint, resume
from .privacy import PIIRedactor, SensitivityLevel, VersionedSchema
from .core.state import (
    Alert,
    AlertSeverity,
    AlertType,
    Decision,
    FileChange,
    SessionStatus,
    TaskState,
    TokenUsage,
    ToolCall,
)
from .monitors import (
    CostBudgetExceeded,
    CostMonitor,
    DriftMonitor,
    PackageHallucinationMonitor,
    TokenMonitor,
    ToolBloatMonitor,
)

__version__ = "0.2.0"

__all__ = [
    # Primary API
    "Checkpoint",
    "checkpoint",       # factory + decorator + context manager
    "resume",
    # Privacy & Schema
    "PIIRedactor",
    "SensitivityLevel",
    "VersionedSchema",
    # State models
    "TaskState",
    "Decision",
    "FileChange",
    "ToolCall",
    "TokenUsage",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "SessionStatus",
    # Monitors
    "CostMonitor",
    "CostBudgetExceeded",
    "TokenMonitor",
    "DriftMonitor",
    "PackageHallucinationMonitor",
    "ToolBloatMonitor",
]
