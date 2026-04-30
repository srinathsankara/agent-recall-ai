"""
smolagentsAdapter — checkpointing for HuggingFace smolagents.

smolagents (github.com/huggingface/smolagents) is a lightweight agent library
from HuggingFace. Its CodeAgent and ToolCallingAgent run multi-step loops that
can easily exhaust context windows on complex tasks.

This adapter hooks into the agent's step() lifecycle to record each reasoning
step as a Decision Anchor and each tool call with its output.

Usage:
    from smolagents import CodeAgent, HfApiModel
    from agent_recall_ai import Checkpoint
    from agent_recall_ai.adapters import smolagentsAdapter

    with Checkpoint("smolagents-run") as cp:
        adapter = smolagentsAdapter(cp)
        agent = adapter.wrap(
            CodeAgent(tools=[...], model=HfApiModel())
        )
        result = agent.run("Analyze the sales data in data.csv")

    # Resume after death:
    state = resume("smolagents-run")
    print(state.resume_prompt())

Integration approach
--------------------
smolagents agents expose a `run()` method that drives an internal step loop.
The adapter wraps `run()` and optionally the internal `step()` method (if
present) to record per-step decisions.

The agent's `logs` list (populated by smolagents internally) is scanned after
each step to extract tool calls and their outputs.
"""
from __future__ import annotations

import logging
import types
from typing import Any, Optional

from .base import BaseAdapter, register_adapter

logger = logging.getLogger(__name__)

try:
    import smolagents as _smolagents
    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False


@register_adapter("smolagents")
class smolagentsAdapter(BaseAdapter):
    """
    Wraps a smolagents agent (CodeAgent / ToolCallingAgent) to auto-checkpoint
    every reasoning step and tool call.

    Args:
        checkpoint: A Checkpoint instance to record into.
        record_steps: If True, record each agent step as a Decision Anchor.
        max_step_chars: Truncate step reasoning to this many characters.
    """

    framework = "smolagents"

    def __init__(
        self,
        checkpoint: Any,
        record_steps: bool = True,
        max_step_chars: int = 600,
    ) -> None:
        super().__init__(checkpoint)
        self._record_steps = record_steps
        self._max_step_chars = max_step_chars
        self._step_count = 0

    def wrap(self, agent: Any, **kwargs: Any) -> Any:
        """
        Wrap a smolagents agent.  Returns the same agent with an
        instrumented run() method.
        """
        if not _SMOLAGENTS_AVAILABLE:
            raise ImportError(
                "smolagents is required: pip install 'agent-recall-ai[smolagents]'"
            )

        adapter = self
        original_run = agent.run

        def _instrumented_run(task: str, *args: Any, **kw: Any) -> Any:
            adapter._checkpoint.set_goal(task[:200])
            adapter._checkpoint.save()

            # Wrap the internal step() method if accessible
            _wrap_step_method(agent, adapter)

            try:
                result = original_run(task, *args, **kw)
            except Exception as exc:
                adapter.on_error(exc)
                raise

            # Scan agent logs after the run for any steps not already captured
            _harvest_logs(agent, adapter)

            final_answer = str(result)[:adapter._max_step_chars] if result else ""
            adapter._checkpoint.record_decision(
                summary="Agent run completed",
                reasoning=final_answer,
                tags=["smolagents", "final_answer"],
            )
            return result

        agent.run = types.MethodType(lambda self_agent, task, *a, **kw: _instrumented_run(task, *a, **kw), agent)
        return agent


def _wrap_step_method(agent: Any, adapter: "smolagentsAdapter") -> None:
    """Wrap agent.step() if it exists (smolagents >= 0.2)."""
    original_step = getattr(agent, "step", None)
    if original_step is None or getattr(agent, "_ac_step_wrapped", False):
        return

    def _instrumented_step(self_agent: Any, *args: Any, **kwargs: Any) -> Any:
        adapter._step_count += 1
        step_num = adapter._step_count

        try:
            result = original_step(*args, **kwargs)
        except Exception as exc:
            adapter.on_error(exc)
            raise

        if adapter._record_steps and result is not None:
            # smolagents step() returns an ActionStep or similar
            thought = getattr(result, "rationale", None) or getattr(result, "thought", None) or ""
            tool_name = getattr(result, "tool_name", None) or ""
            tool_input = getattr(result, "tool_input", None) or ""
            tool_output = getattr(result, "observations", None) or ""

            if thought or tool_name:
                reasoning = (str(thought)[:300] + " → " + str(tool_output)[:200]).strip(" →")
                adapter._checkpoint.record_decision(
                    summary=f"Step {step_num}: {tool_name or 'reasoning'}",
                    reasoning=reasoning[:adapter._max_step_chars],
                    tags=["smolagents", f"step:{step_num}"],
                )

            if tool_name:
                adapter.on_tool_start(
                    tool_name=tool_name,
                    input_summary=str(tool_input)[:200],
                )
                adapter.on_tool_end(
                    tool_name=tool_name,
                    output_summary=str(tool_output)[:200],
                )

        return result

    agent.step = types.MethodType(_instrumented_step, agent)
    agent._ac_step_wrapped = True


def _harvest_logs(agent: Any, adapter: "smolagentsAdapter") -> None:
    """
    Scan agent.logs after a run to capture any steps not already recorded
    (e.g. when step() wrapping was not available).
    """
    logs = getattr(agent, "logs", None) or []
    for entry in logs:
        tool_name = getattr(entry, "tool_name", None) or ""
        observations = getattr(entry, "observations", None) or ""
        if tool_name and not getattr(entry, "_ac_recorded", False):
            adapter.on_tool_start(tool_name=tool_name, input_summary="")
            adapter.on_tool_end(
                tool_name=tool_name,
                output_summary=str(observations)[:300],
            )
            try:
                entry._ac_recorded = True
            except AttributeError:
                pass  # frozen objects — skip marking
