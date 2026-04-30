"""
The Checkpoint class — the primary public API for agent-recall-ai.

Usage (sync):
    with Checkpoint("my-task") as cp:
        cp.set_goal("Refactor authentication module")
        cp.add_constraint("Do not change the public API")
        cp.record_decision("Used PyJWT", reasoning="More maintained than python-jose")
        cp.record_file_modified("auth/tokens.py")
        cp.save()  # explicit save (also happens automatically on exit)

Usage (async):
    async with Checkpoint("my-task") as cp:
        cp.set_goal("Async agent task")
        result = await some_async_llm_call()
        cp.record_decision("Chose streaming", reasoning="Lower latency")
        # Auto-saves on exit (non-blocking via asyncio.to_thread)

Usage (decorator):
    @checkpoint("my-task")
    async def run_agent(goal: str):
        ...

    @checkpoint("my-task")
    def run_agent_sync(goal: str):
        ...
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Union

from .core.state import Alert, AlertSeverity, AlertType, SessionStatus, TaskState, TokenUsage
from .core.tracker import TokenCostTracker
from .monitors.base import BaseMonitor
from .storage.disk import DiskStore
from .storage.memory import MemoryStore

logger = logging.getLogger(__name__)

Store = Union[DiskStore, MemoryStore]


class Checkpoint:
    """
    Context manager and session tracker for a single agent task.

    Args:
        session_id: Unique name for this task (e.g. "refactor-auth-v2").
        store: Storage backend. Defaults to DiskStore at .agent-recall-ai/.
        monitors: List of BaseMonitor instances to run after state updates.
        model: LLM model name (used for cost/token calculations).
        auto_save_every: Auto-save every N token updates (0 = disabled).
    """

    def __init__(
        self,
        session_id: str,
        store: Optional[Store] = None,
        monitors: Optional[list[BaseMonitor]] = None,
        model: str = "gpt-4o-mini",
        auto_save_every: int = 10,
        redactor: Optional[Any] = None,       # PIIRedactor instance
        schema: Optional[Any] = None,         # VersionedSchema instance
    ) -> None:
        self.session_id = session_id
        self._store = store or DiskStore()
        self._monitors = monitors or []
        self._model = model
        self._auto_save_every = auto_save_every
        self._redactor = redactor
        self._schema = schema
        self._tracker = TokenCostTracker(model=model)
        self._token_update_count = 0

        # Load existing state or create new
        existing = self._store.load(session_id)
        if existing is not None:
            self._state = existing
            logger.info("Resumed checkpoint: %s (seq #%d)", session_id, existing.checkpoint_seq)
        else:
            self._state = TaskState(
                session_id=session_id,
                metadata={"model": model},
            )

    # ── Sync context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._state.status = SessionStatus.FAILED
            # Truncate exc_val to prevent PII/secrets leaking into stored alerts.
            # The full traceback stays in application logs, not in the checkpoint.
            exc_msg = str(exc_val)[:200] if exc_val else ""
            self._state.add_alert(
                AlertType.BEHAVIORAL_DRIFT,
                AlertSeverity.ERROR,
                f"Session ended with exception: {exc_type.__name__}: {exc_msg}",
            )
        else:
            self._state.status = SessionStatus.COMPLETED
        self.save()

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "Checkpoint":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._state.status = SessionStatus.FAILED
            exc_msg = str(exc_val)[:200] if exc_val else ""
            self._state.add_alert(
                AlertType.BEHAVIORAL_DRIFT,
                AlertSeverity.ERROR,
                f"Session ended with exception: {exc_type.__name__}: {exc_msg}",
            )
        else:
            self._state.status = SessionStatus.COMPLETED
        # Non-blocking save — disk I/O off the event loop
        await asyncio.to_thread(self.save)

    # ── Goal and constraint management ────────────────────────────────────────

    def set_goal(self, goal: str) -> None:
        if goal not in self._state.goals:
            self._state.goals.append(goal)
            self._state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    def set_goals(self, goals: list[str]) -> None:
        for g in goals:
            self.set_goal(g)

    def add_constraint(self, constraint: str) -> None:
        if constraint not in self._state.constraints:
            self._state.constraints.append(constraint)
            self._state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    def set_context(self, summary: str) -> None:
        """Freeform context description — included verbatim in resume prompts."""
        self._state.context_summary = summary
        self._state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # ── Decision recording ────────────────────────────────────────────────────

    def record_decision(
        self,
        summary: str,
        reasoning: str = "",
        alternatives_rejected: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        decision = self._state.add_decision(
            summary=summary,
            reasoning=reasoning,
            alternatives_rejected=alternatives_rejected or [],
            tags=tags or [],
        )
        self._run_monitors("on_decision")

    def record_file_modified(self, path: str, action: str = "modified", description: str = "") -> None:
        self._state.add_file(path=path, action=action, description=description)

    def add_next_step(self, step: str) -> None:
        if step not in self._state.next_steps:
            self._state.next_steps.append(step)
            self._state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    def add_open_question(self, question: str) -> None:
        if question not in self._state.open_questions:
            self._state.open_questions.append(question)
            self._state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # ── Token/cost recording ──────────────────────────────────────────────────

    def record_tokens(
        self,
        prompt: int = 0,
        completion: int = 0,
        cached: int = 0,
        model: Optional[str] = None,
    ) -> float:
        """Record a token usage event. Returns cost of this call in USD."""
        cost = self._tracker.record(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            model=model or self._model,
        )
        total = self._tracker.total()
        self._state.token_usage.add(prompt=prompt, completion=completion, cached=cached)
        self._state.cost_usd += cost
        self._state.context_utilization = self._tracker.context_utilization(
            total.prompt_tokens, model=model or self._model
        )

        self._token_update_count += 1
        self._run_monitors("on_tokens")

        if self._auto_save_every > 0 and self._token_update_count % self._auto_save_every == 0:
            self.save()

        return cost

    def record_tool_call(
        self,
        tool_name: str,
        input_summary: str = "",
        output_summary: str = "",
        output_tokens: int = 0,
    ) -> None:
        self._state.add_tool_call(
            tool_name=tool_name,
            input_summary=input_summary,
            output_summary=output_summary,
            output_tokens=output_tokens,
        )
        self._run_monitors("on_tool_call")

    # ── State access ──────────────────────────────────────────────────────────

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def session_id_str(self) -> str:
        return self.session_id

    @property
    def cost_usd(self) -> float:
        return self._state.cost_usd

    @property
    def token_total(self) -> int:
        return self._state.token_usage.total

    @property
    def context_utilization(self) -> float:
        return self._state.context_utilization

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist the current state to storage.

        If a PIIRedactor is attached, the state dict is scanned and redacted
        BEFORE serialization — secrets never hit disk or Redis.

        If a VersionedSchema is attached, the schema_version is stamped
        onto the saved dict.
        """
        if self._redactor is not None or self._schema is not None:
            # Work on a deep copy so the in-memory state retains original values
            state_dict = json.loads(self._state.model_dump_json())

            if self._schema is not None:
                state_dict = self._schema.stamp(state_dict)

            if self._redactor is not None:
                self._redactor.redact_state(state_dict, self.session_id)

            # Reconstruct TaskState from the (potentially modified) dict and save
            redacted_state = TaskState.model_validate(state_dict)
            self._store.save(redacted_state)
        else:
            self._store.save(self._state)

    def resume_prompt(self) -> str:
        """Return a structured resume prompt — paste into a new session."""
        return self._state.resume_prompt()

    def as_handoff(self) -> dict[str, Any]:
        """Export as multi-agent handoff payload."""
        return self._state.as_handoff()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_monitors(self, event: str) -> None:
        for monitor in self._monitors:
            fn = getattr(monitor, event, None)
            if fn is None:
                continue
            try:
                alerts = fn(self._state)
                for alert_dict in alerts or []:
                    self._state.add_alert(
                        alert_type=alert_dict["alert_type"],
                        severity=alert_dict["severity"],
                        message=alert_dict["message"],
                        detail=alert_dict.get("detail"),
                    )
            except Exception as exc:
                # CostBudgetExceeded and similar should propagate
                from .monitors.cost_monitor import CostBudgetExceeded
                if isinstance(exc, CostBudgetExceeded):
                    self.save()
                    raise
                logger.warning("Monitor %s.%s raised: %s", type(monitor).__name__, event, exc)


    # ── Thread forking ────────────────────────────────────────────────────────

    def fork(
        self,
        target_session_id: str,
        store: Optional[Store] = None,
        monitors: Optional[list[BaseMonitor]] = None,
    ) -> "Checkpoint":
        """
        Fork this session into a new thread so you can explore an alternative
        reasoning path from the current checkpoint without modifying the original.

        The forked session starts with a copy of all state (goals, decisions,
        files, constraints, context) and sets ``metadata["parent_thread_id"]``
        so the lineage is traceable.

        Example::

            with Checkpoint("main-task") as cp:
                cp.set_goal("Refactor auth module")
                cp.record_decision("Use PyJWT", reasoning="Better maintained")

                # Explore an alternative approach in parallel
                alt = cp.fork("main-task-alt")
                alt.record_decision("Try python-jose instead", reasoning="Lighter weight")
                alt.save()

        Returns the forked Checkpoint (not a context manager — call ``.save()``
        or use ``with forked_cp:`` manually).
        """
        # Deep copy state via JSON round-trip (safe for all Pydantic models)
        state_dict = json.loads(self._state.model_dump_json())
        state_dict["session_id"] = target_session_id
        state_dict["checkpoint_seq"] = 0
        _now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        state_dict["created_at"] = _now
        state_dict["updated_at"] = _now
        state_dict["metadata"] = {
            **state_dict.get("metadata", {}),
            "parent_thread_id": self.session_id,
            "forked_at": _now,
        }

        forked_state = TaskState.model_validate(state_dict)

        forked_store = store or self._store
        forked_store.save(forked_state)

        forked_cp = Checkpoint(
            session_id=target_session_id,
            store=forked_store,
            monitors=monitors or list(self._monitors),
            model=self._model,
            auto_save_every=self._auto_save_every,
            redactor=self._redactor,
            schema=self._schema,
        )
        logger.info(
            "Forked session %s -> %s (decisions=%d, files=%d)",
            self.session_id, target_session_id,
            len(forked_state.decisions), len(forked_state.files_modified),
        )
        return forked_cp


def resume(session_id: str, store: Optional[Store] = None) -> Optional[TaskState]:
    """
    Load a saved checkpoint by session_id.
    Returns None if no checkpoint exists.
    """
    s = store or DiskStore()
    return s.load(session_id)


def checkpoint(
    session_id: str,
    store: Optional[Store] = None,
    monitors: Optional[list[BaseMonitor]] = None,
    model: str = "gpt-4o-mini",
    auto_save_every: int = 10,
    redactor: Optional[Any] = None,
    schema: Optional[Any] = None,
) -> Union[Checkpoint, Callable]:
    """
    Decorator **and** factory function.

    As a factory (returns a Checkpoint instance):
        cp = checkpoint("my-task")
        with cp:
            ...

    As a decorator (wraps sync or async functions):
        @checkpoint("my-task")
        async def run_agent():
            ...

        @checkpoint("my-task")
        def run_agent_sync():
            ...

    The decorated function receives the Checkpoint as its first keyword
    argument ``cp`` (injected automatically). If the function already
    defines a ``cp`` parameter, the checkpoint is passed there; otherwise
    it is injected silently and accessible via the ``_checkpoint`` attribute
    on the returned wrapper.
    """

    def _make_cp() -> Checkpoint:
        return Checkpoint(
            session_id=session_id,
            store=store,
            monitors=monitors,
            model=model,
            auto_save_every=auto_save_every,
            redactor=redactor,
            schema=schema,
        )

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        _injects_cp = "cp" in sig.parameters

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                cp = _make_cp()
                async_wrapper._checkpoint = cp  # type: ignore[attr-defined]
                if _injects_cp:
                    kwargs["cp"] = cp
                async with cp:
                    return await fn(*args, **kwargs)
            async_wrapper._checkpoint = None  # type: ignore[attr-defined]
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                cp = _make_cp()
                sync_wrapper._checkpoint = cp  # type: ignore[attr-defined]
                if _injects_cp:
                    kwargs["cp"] = cp
                with cp:
                    return fn(*args, **kwargs)
            sync_wrapper._checkpoint = None  # type: ignore[attr-defined]
            return sync_wrapper

    # Allow both `@checkpoint("id")` and `cp = checkpoint("id")` usage.
    # If called without a callable argument it acts as a decorator factory;
    # the factory itself is also directly usable as a Checkpoint instance.
    class _CheckpointFactoryOrDecorator:
        """Returned by checkpoint() — usable as decorator or context manager."""

        def __call__(self, fn: Callable) -> Callable:           # @checkpoint("id")
            return decorator(fn)

        def __enter__(self) -> Checkpoint:                      # with checkpoint("id") as cp:
            self._cp = _make_cp()
            return self._cp.__enter__()

        def __exit__(self, *a: Any) -> None:
            self._cp.__exit__(*a)

        async def __aenter__(self) -> Checkpoint:               # async with checkpoint("id") as cp:
            self._cp = _make_cp()
            return await self._cp.__aenter__()

        async def __aexit__(self, *a: Any) -> None:
            await self._cp.__aexit__(*a)

    return _CheckpointFactoryOrDecorator()
