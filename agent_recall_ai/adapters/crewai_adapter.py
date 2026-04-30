"""
CrewAIAdapter — checkpointing for CrewAI multi-agent workflows.

Solves session death in long CrewAI runs where crews with many tasks can
exhaust the context window before finishing.

Supports:
  - Crew.kickoff() — auto-saves a checkpoint before and after each task
  - Task result recording (decisions with tool usage)
  - Agent → file change tracking
  - Cost accumulation across the full crew run

Usage:
    from crewai import Crew, Agent, Task
    from agent_recall_ai import Checkpoint
    from agent_recall_ai.adapters import CrewAIAdapter

    with Checkpoint("my-crew-run") as cp:
        adapter = CrewAIAdapter(cp)
        crew = adapter.wrap(
            Crew(agents=[...], tasks=[...])
        )
        result = crew.kickoff()

    # On resume after session death:
    state = resume("my-crew-run")
    print(state.resume_prompt())
    # → shows which tasks completed, which are pending, all decisions made

How wrapping works
------------------
CrewAI does not have a built-in callback system, so CrewAIAdapter uses
Python __class__ monkey-patching on the Crew instance (not the class) to
intercept kickoff(). Each Task's execute() is similarly wrapped to record
per-task decisions.

Agents are not wrapped — only the Task boundary is tracked, keeping the
checkpoint lightweight.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .base import BaseAdapter, register_adapter

logger = logging.getLogger(__name__)

try:
    import crewai as _crewai
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False


@register_adapter("crewai")
class CrewAIAdapter(BaseAdapter):
    """
    Wraps a CrewAI Crew to auto-checkpoint goal, task decisions, and cost.

    Args:
        checkpoint: A Checkpoint instance to record into.
        record_task_outputs: If True, record each task's output as a decision.
        max_output_chars: Truncate task outputs to this many characters.
    """

    framework = "crewai"

    def __init__(
        self,
        checkpoint: Any,
        record_task_outputs: bool = True,
        max_output_chars: int = 500,
    ) -> None:
        super().__init__(checkpoint)
        self._record_task_outputs = record_task_outputs
        self._max_output_chars = max_output_chars

    def wrap(self, crew: Any, **kwargs: Any) -> Any:
        """
        Wrap a CrewAI Crew instance.  Returns the same crew with an
        instrumented kickoff() method.
        """
        if not _CREWAI_AVAILABLE:
            raise ImportError(
                "crewai package is required: pip install 'agent-recall-ai[crewai]'"
            )

        adapter = self
        original_kickoff = crew.kickoff

        def _instrumented_kickoff(*args: Any, **kw: Any) -> Any:
            # Record the crew's goal as a checkpoint goal
            crew_description = getattr(crew, "description", None) or ""
            if crew_description:
                adapter.checkpoint.set_context(f"CrewAI run: {crew_description[:200]}")

            # Record each agent role as a goal
            for agent in getattr(crew, "agents", []):
                role = getattr(agent, "role", None)
                goal = getattr(agent, "goal", None)
                if role and goal:
                    adapter.checkpoint.set_goal(f"[{role}] {goal[:150]}")

            # Wrap each task to record its completion
            tasks = getattr(crew, "tasks", []) or []
            for task in tasks:
                _wrap_task(task, adapter)

            # Save a "started" checkpoint
            adapter.checkpoint.save()

            try:
                result = original_kickoff(*args, **kw)
            except Exception as exc:
                adapter.on_error(exc)
                raise

            # Record the overall result
            result_str = str(result)[:adapter._max_output_chars] if result else ""
            adapter.checkpoint.record_decision(
                "Crew run completed",
                reasoning=result_str,
                tags=["crewai", "final_result"],
            )
            return result

        # Bind to instance only (not the class)
        import types
        crew.kickoff = types.MethodType(lambda self, *a, **kw: _instrumented_kickoff(*a, **kw), crew)
        return crew


def _wrap_task(task: Any, adapter: "CrewAIAdapter") -> None:
    """Monkey-patch a single Task's execute() to record decisions."""
    original_execute = getattr(task, "execute", None)
    if original_execute is None:
        return

    import types

    def _instrumented_execute(self_task: Any, *args: Any, **kwargs: Any) -> Any:
        task_desc = getattr(self_task, "description", "")[:100]
        agent = getattr(self_task, "agent", None)
        agent_role = getattr(agent, "role", "unknown") if agent else "unknown"

        adapter.on_tool_start(
            tool_name=f"task:{agent_role}",
            input_summary=task_desc,
        )
        try:
            result = original_execute(*args, **kwargs)
        except Exception as exc:
            adapter.on_error(exc)
            raise

        output = str(result)[:adapter._max_output_chars] if result else ""

        if adapter._record_task_outputs:
            adapter.checkpoint.record_decision(
                summary=f"Task completed: {task_desc[:80]}",
                reasoning=output,
                tags=["crewai", f"agent:{agent_role}"],
            )

        # Track expected output file if specified
        output_file = getattr(self_task, "output_file", None)
        if output_file:
            adapter.checkpoint.record_file_modified(
                path=output_file,
                action="created",
                description=f"Output from task: {task_desc[:60]}",
            )

        adapter.on_tool_end(
            tool_name=f"task:{agent_role}",
            output_summary=output[:200],
        )
        return result

    task.execute = types.MethodType(_instrumented_execute, task)
