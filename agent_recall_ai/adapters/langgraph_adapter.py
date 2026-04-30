"""
LangGraphAdapter — drop-in BaseCheckpointSaver for LangGraph.

Gives any LangGraph graph persistent, resumable checkpoints via agent-recall-ai's
storage backends (SQLite, Redis, Memory) without changing a single line of graph code.

Solves the context-loss pattern from:
  github.com/langchain-ai/langgraph/issues/2946

Usage
-----
    from langgraph.graph import StateGraph
    from agent_recall_ai.adapters import LangGraphAdapter

    # Use in-memory store (testing)
    checkpointer = LangGraphAdapter.from_memory()

    # Use SQLite (production single-machine)
    checkpointer = LangGraphAdapter.from_sqlite("checkpoints.db")

    # Use Redis (distributed / multi-agent)
    checkpointer = LangGraphAdapter.from_redis("redis://localhost:6379")

    # Wire into graph
    graph = builder.compile(checkpointer=checkpointer)

    # The thread_id maps to an agent-recall-ai session_id
    config = {"configurable": {"thread_id": "my-session"}}
    result = graph.invoke({"messages": [...]}, config)

    # Resume after session death — agent-recall-ai has the state
    result = graph.invoke(None, config)   # LangGraph pattern for resume

Thread forking
--------------
    fork_config = {"configurable": {"thread_id": "fork-explore-alt"}}
    checkpointer.fork("my-session", "fork-explore-alt")
    result = graph.invoke({"messages": [...]}, fork_config)

The fork copies all state from the parent thread so the graph resumes from
the same point but can diverge from there.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        get_checkpoint_id,
    )
    from langgraph.checkpoint.base import ChannelVersions
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    BaseCheckpointSaver = object  # type: ignore[assignment,misc]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _thread_from_config(config: dict) -> str:
    """Extract thread_id from a LangGraph RunnableConfig."""
    return config.get("configurable", {}).get("thread_id", "default")


def _checkpoint_id_from_config(config: dict) -> Optional[str]:
    return config.get("configurable", {}).get("checkpoint_id")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


# ── Adapter ───────────────────────────────────────────────────────────────────

if _LANGGRAPH_AVAILABLE:
    class LangGraphAdapter(BaseCheckpointSaver):
        """
        A LangGraph BaseCheckpointSaver backed by agent-recall-ai storage.

        This is a drop-in replacement for LangGraph's built-in SQLiteSaver or
        MemorySaver — just swap the checkpointer argument in `graph.compile()`.

        State is stored as JSON under the key
        ``langgraph:{thread_id}:{checkpoint_id}`` in the configured store,
        making it fully portable across processes and machines.
        """

        def __init__(self, store: Any) -> None:
            """
            Parameters
            ----------
            store:
                Any agent-recall-ai store (DiskStore, MemoryStore, RedisProvider).
                Must implement ``.save(session_id, state_dict)`` and
                ``.load(session_id) -> Optional[dict]``.
            """
            super().__init__()
            self._store = store

        # ── Factory helpers ───────────────────────────────────────────────────

        @classmethod
        def from_memory(cls) -> "LangGraphAdapter":
            """In-memory store — perfect for tests and short-lived agents."""
            from ..storage.memory import MemoryStore
            return cls(MemoryStore())

        @classmethod
        def from_sqlite(cls, db_path: str = "agent_recall_ai.db") -> "LangGraphAdapter":
            """SQLite-backed persistent store."""
            from ..storage.disk import DiskStore
            return cls(DiskStore(db_path))

        @classmethod
        def from_redis(cls, url: str = "redis://localhost:6379", **kwargs: Any) -> "LangGraphAdapter":
            """Redis-backed distributed store."""
            from ..persistence.redis_provider import RedisProvider
            return cls(RedisProvider(url, **kwargs))

        # ── Internal key helpers ──────────────────────────────────────────────

        def _session_key(self, thread_id: str, checkpoint_id: Optional[str] = None) -> str:
            if checkpoint_id:
                return f"langgraph:{thread_id}:{checkpoint_id}"
            return f"langgraph:{thread_id}:latest"

        def _write(self, thread_id: str, checkpoint_id: str, data: dict) -> None:
            """Write checkpoint data under both versioned and latest keys."""
            versioned_key = self._session_key(thread_id, checkpoint_id)
            latest_key = self._session_key(thread_id)
            self._store.save(versioned_key, data)
            self._store.save(latest_key, data)

        def _read(self, key: str) -> Optional[dict]:
            try:
                return self._store.load(key)
            except Exception:
                return None

        # ── BaseCheckpointSaver interface ─────────────────────────────────────

        def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
            """Load the most recent (or specified) checkpoint for a thread."""
            thread_id = _thread_from_config(config)
            checkpoint_id = _checkpoint_id_from_config(config)
            key = self._session_key(thread_id, checkpoint_id)
            data = self._read(key)
            if data is None:
                return None
            return self._dict_to_tuple(config, data)

        def put(
            self,
            config: dict,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> dict:
            """Persist a checkpoint and return the updated config."""
            thread_id = _thread_from_config(config)
            cp_id = checkpoint.get("id") or get_checkpoint_id(checkpoint)

            data = {
                "checkpoint": checkpoint,
                "metadata": metadata,
                "new_versions": new_versions,
                "thread_id": thread_id,
                "checkpoint_id": cp_id,
                "saved_at": _now_iso(),
            }
            self._write(thread_id, cp_id, data)
            logger.debug("LangGraphAdapter.put thread=%s checkpoint=%s", thread_id, cp_id)

            return {
                **config,
                "configurable": {
                    **config.get("configurable", {}),
                    "thread_id": thread_id,
                    "checkpoint_id": cp_id,
                },
            }

        def list(
            self,
            config: Optional[dict],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[dict] = None,
            limit: Optional[int] = None,
        ) -> Iterator[CheckpointTuple]:
            """
            List checkpoints for a thread.

            Note: full history listing requires a store that supports prefix
            scanning.  The base implementation returns only the latest checkpoint
            to avoid requiring specific store APIs.
            """
            if config is None:
                return
            latest = self.get_tuple(config)
            if latest is not None:
                yield latest

        async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
            """Async version of get_tuple."""
            import asyncio
            return await asyncio.to_thread(self.get_tuple, config)

        async def aput(
            self,
            config: dict,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> dict:
            """Async version of put."""
            import asyncio
            return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

        async def alist(
            self,
            config: Optional[dict],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[dict] = None,
            limit: Optional[int] = None,
        ) -> AsyncIterator[CheckpointTuple]:
            """Async version of list."""
            for item in self.list(config, filter=filter, before=before, limit=limit):
                yield item

        # ── Thread forking ────────────────────────────────────────────────────

        def fork(self, source_thread_id: str, target_thread_id: str) -> bool:
            """
            Fork an existing thread into a new thread_id.

            Copies all state from ``source_thread_id`` to ``target_thread_id``
            so the graph can explore an alternative reasoning path from the
            same checkpoint without modifying the original.

            Returns True if the fork succeeded, False if source was not found.

            Example::

                checkpointer.fork("main-session", "alt-branch-1")
                config = {"configurable": {"thread_id": "alt-branch-1"}}
                result = graph.invoke({"messages": [...]}, config)
            """
            source_key = self._session_key(source_thread_id)
            data = self._read(source_key)
            if data is None:
                logger.warning(
                    "LangGraphAdapter.fork: source thread '%s' not found", source_thread_id
                )
                return False

            # Stamp the fork metadata
            forked_data = {
                **data,
                "thread_id": target_thread_id,
                "forked_from": source_thread_id,
                "forked_at": _now_iso(),
            }
            cp_id = data.get("checkpoint_id", "latest")
            self._write(target_thread_id, cp_id, forked_data)
            logger.info(
                "LangGraphAdapter.fork: %s → %s (checkpoint=%s)",
                source_thread_id, target_thread_id, cp_id,
            )
            return True

        # ── Internal helpers ──────────────────────────────────────────────────

        def _dict_to_tuple(self, config: dict, data: dict) -> CheckpointTuple:
            thread_id = data.get("thread_id", _thread_from_config(config))
            cp_id = data.get("checkpoint_id", "unknown")
            checkpoint = data.get("checkpoint", {})
            metadata = data.get("metadata", {})

            parent_config = None
            parent_id = data.get("parent_checkpoint_id")
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": parent_id,
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": cp_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=data.get("pending_writes"),
            )

else:
    # Stub when langgraph is not installed
    class LangGraphAdapter:  # type: ignore[no-redef]
        """
        Stub LangGraphAdapter — install langgraph to activate.

            pip install 'agent-recall-ai[langgraph]'
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "langgraph is required: pip install 'agent-recall-ai[langgraph]'"
            )

        @classmethod
        def from_memory(cls) -> "LangGraphAdapter":
            raise ImportError("pip install 'agent-recall-ai[langgraph]'")

        @classmethod
        def from_sqlite(cls, *args: Any, **kwargs: Any) -> "LangGraphAdapter":
            raise ImportError("pip install 'agent-recall-ai[langgraph]'")

        @classmethod
        def from_redis(cls, *args: Any, **kwargs: Any) -> "LangGraphAdapter":
            raise ImportError("pip install 'agent-recall-ai[langgraph]'")
