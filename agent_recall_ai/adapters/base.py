"""
Base adapter interface for the plugin-based adapter system.

Every adapter wraps a specific framework (Anthropic, OpenAI, LangChain, custom)
and provides a uniform interface for:
  - Intercepting LLM calls to record token usage
  - Intercepting tool calls to record outputs
  - Auto-saving checkpoints at configurable intervals
  - Compressing context when token pressure is detected

Adapter registration is done via the _ADAPTER_REGISTRY dict.
Use `get_adapter(framework)` to retrieve a registered adapter class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..checkpoint import Checkpoint

_ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register_adapter(name: str):
    """Class decorator to register an adapter under a framework name."""
    def decorator(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        _ADAPTER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_adapter(framework: str) -> type[BaseAdapter]:
    """
    Retrieve a registered adapter class by framework name.

    Raises:
        KeyError: If no adapter is registered for the given framework name.
    """
    key = framework.lower()
    if key not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise KeyError(
            f"No adapter registered for '{framework}'. "
            f"Available: {available or 'none'}"
        )
    return _ADAPTER_REGISTRY[key]


def list_adapters() -> list[str]:
    """Return the names of all registered adapter frameworks."""
    return sorted(_ADAPTER_REGISTRY.keys())


class BaseAdapter(ABC):
    """
    Abstract base class for all agent-recall-ai framework adapters.

    Subclasses must implement `wrap()` which returns a context-manager-aware
    wrapper for the underlying client/chain.
    """

    #: Framework name — set by subclasses
    framework: str = "custom"

    def __init__(self, checkpoint: Checkpoint) -> None:
        self.checkpoint = checkpoint
        # Tag the checkpoint with the framework it was created from
        self.checkpoint.state.metadata["framework"] = self.framework

    @abstractmethod
    def wrap(self, client: Any, **kwargs: Any) -> Any:
        """
        Wrap the framework client or chain with checkpoint instrumentation.
        Returns the wrapped object — use it exactly like the original.
        """
        ...

    def on_llm_start(self, model: str, messages: list[dict]) -> None:
        """Called before an LLM API call."""

    def on_llm_end(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> None:
        """Called after a successful LLM API call with usage stats."""
        self.checkpoint.record_tokens(
            prompt=prompt_tokens,
            completion=completion_tokens,
            cached=cached_tokens,
            model=model,
        )

    def on_tool_start(self, tool_name: str, tool_input: str) -> None:
        """Called before a tool invocation."""

    def on_tool_end(self, tool_name: str, tool_output: str) -> None:
        """Called after a tool invocation completes."""
        self.checkpoint.record_tool_call(
            tool_name=tool_name,
            output_summary=tool_output[:200],
            output_tokens=len(tool_output) // 4,
        )

    def on_error(self, error: Exception) -> None:
        """Called when an unhandled error occurs. Triggers emergency save."""
        self.checkpoint.save()
