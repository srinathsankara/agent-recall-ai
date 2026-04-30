"""
LangChainAdapter — custom BaseChatMessageHistory that auto-checkpoints.

Solves the context-loss pattern from:
  github.com/google/adk-python/issues/1738

Two integration modes:

Mode 1 — Message History (preferred for chains with memory):
    from agent_recall_ai.adapters import LangChainAdapter
    history = LangChainAdapter(cp).as_message_history()
    chain_with_history = RunnableWithMessageHistory(chain, lambda _: history)

Mode 2 — Callback Handler (for any chain):
    from agent_recall_ai.adapters import LangChainAdapter
    handler = LangChainAdapter(cp).as_callback_handler()
    chain.invoke(input, config={"callbacks": [handler]})
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from uuid import UUID

from .base import BaseAdapter, register_adapter

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # Stubs so the module is importable without langchain
    BaseChatMessageHistory = object  # type: ignore[assignment,misc]
    BaseCallbackHandler = object     # type: ignore[assignment,misc]


@register_adapter("langchain")
class LangChainAdapter(BaseAdapter):
    """
    Adapter for LangChain chains and agents.

    Provides two instruments:
    1. CheckpointMessageHistory  — a BaseChatMessageHistory that auto-saves on add
    2. CheckpointCallbackHandler — a BaseCallbackHandler for any chain
    """

    framework = "langchain"

    def wrap(self, client: Any, **kwargs: Any) -> "CheckpointCallbackHandler":
        """Default: wrap returns a callback handler."""
        return self.as_callback_handler()

    def as_message_history(self) -> "CheckpointMessageHistory":
        """Return a BaseChatMessageHistory that auto-checkpoints."""
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError("pip install 'agent-recall-ai[langchain]'")
        return CheckpointMessageHistory(self)

    def as_callback_handler(self) -> "CheckpointCallbackHandler":
        """Return a BaseCallbackHandler for use in chain.invoke(config=...)."""
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError("pip install 'agent-recall-ai[langchain]'")
        return CheckpointCallbackHandler(self)


if _LANGCHAIN_AVAILABLE:
    class CheckpointMessageHistory(BaseChatMessageHistory):
        """
        A LangChain BaseChatMessageHistory that auto-checkpoints on every add.

        Use with RunnableWithMessageHistory to get automatic state persistence
        without any extra code in your chain.
        """

        def __init__(self, adapter: LangChainAdapter) -> None:
            self._adapter = adapter
            self._messages: list[BaseMessage] = []

        @property
        def messages(self) -> list[BaseMessage]:
            return list(self._messages)

        def add_message(self, message: BaseMessage) -> None:
            self._messages.append(message)
            # Record decision anchors when the AI responds
            if isinstance(message, AIMessage):
                content = message.content if isinstance(message.content, str) else str(message.content)
                from ..core.semantic_pruner import _is_decision_anchor
                if _is_decision_anchor(content):
                    self._adapter.checkpoint.record_decision(
                        summary=content[:120],
                        reasoning="auto-extracted from AI response",
                        tags=["auto-extracted"],
                    )
            # Auto-save every 5 messages
            if len(self._messages) % 5 == 0:
                self._adapter.checkpoint.save()

        def clear(self) -> None:
            self._messages.clear()

    class CheckpointCallbackHandler(BaseCallbackHandler):
        """
        LangChain CallbackHandler — plugs into any chain via config={"callbacks": [handler]}.
        """

        def __init__(self, adapter: LangChainAdapter) -> None:
            super().__init__()
            self._adapter = adapter
            self._active_tool: Optional[str] = None

        def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            try:
                for gen_list in response.generations:
                    for gen in gen_list:
                        info = getattr(gen, "generation_info", None) or {}
                        usage = info.get("token_usage") or {}
                        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                        if prompt or completion:
                            self._adapter.on_llm_end(
                                model=info.get("model", "unknown"),
                                prompt_tokens=prompt,
                                completion_tokens=completion,
                            )
            except Exception:
                pass

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._active_tool = serialized.get("name", "unknown_tool")
            self._adapter.on_tool_start(self._active_tool, input_str[:200])

        def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
            tool_name = self._active_tool or "unknown_tool"
            self._active_tool = None
            self._adapter.on_tool_end(tool_name, str(output)[:200])

        def on_tool_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
            self._active_tool = None
            self._adapter.on_error(error)

        def on_chain_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
            self._adapter.on_error(error)

else:
    # No-op stubs when langchain is not installed
    class CheckpointMessageHistory:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("pip install 'agent-recall-ai[langchain]'")

    class CheckpointCallbackHandler:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("pip install 'agent-recall-ai[langchain]'")
