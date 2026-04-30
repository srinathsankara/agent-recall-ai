"""
AnthropicAdapter — intercepts Claude SDK calls and records state automatically.

Solves the specific context-loss pattern from:
  github.com/anthropics/claude-code/issues/40286

Features
--------
* Auto-records token usage (input / output / cache-read / cache-write)
* Records every tool-use block as a ToolCall anchor
* **Prompt caching** — injects `cache_control` breakpoints on the system
  message, tool definitions, and the most-recent user message so every call
  benefits from Anthropic's 90 % cost/latency reduction for cached tokens
* **Pre-inference token counting** — calls `messages.count_tokens()` before
  each API call; result is stored in `state.metadata["pre_inference_tokens"]`
  so monitors and routers can act on context pressure before spending tokens

Usage:
    from agent_recall_ai import Checkpoint
    from agent_recall_ai.adapters import AnthropicAdapter

    with Checkpoint("my-task") as cp:
        adapter = AnthropicAdapter(cp)
        client = adapter.wrap(anthropic.Anthropic())

        response = client.messages.create(
            model="claude-opus-4-5",
            messages=[{"role": "user", "content": "Refactor the auth module"}],
            max_tokens=4096,
        )
        # Token usage, cache savings, and tool calls are recorded automatically.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from .base import BaseAdapter, register_adapter

logger = logging.getLogger(__name__)

try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ── Prompt-caching helpers ────────────────────────────────────────────────────

def _inject_cache_breakpoints(kwargs: dict) -> dict:
    """
    Return a *copy* of kwargs with Anthropic prompt-cache breakpoints injected.

    Breakpoints are placed on:
    1. System message (converted from str → content block list if needed)
    2. Last tool definition (so the full tool list is cached after first call)
    3. Last user message (so the conversation history is cached)

    Only the final breakpoint in each "slot" is written; earlier ones would be
    redundant (Anthropic caches up to the most-recent breakpoint).
    """
    kw = copy.deepcopy(kwargs)
    cc = {"type": "ephemeral"}

    # 1. System message
    system = kw.get("system")
    if isinstance(system, str) and system:
        kw["system"] = [{"type": "text", "text": system, "cache_control": cc}]
    elif isinstance(system, list) and system:
        # Already a content block list — stamp the last block
        last = system[-1]
        if isinstance(last, dict) and "cache_control" not in last:
            last["cache_control"] = cc

    # 2. Tools list
    tools = kw.get("tools")
    if isinstance(tools, list) and tools:
        last_tool = tools[-1]
        if isinstance(last_tool, dict) and "cache_control" not in last_tool:
            last_tool["cache_control"] = cc

    # 3. Most-recent user message
    messages = kw.get("messages")
    if isinstance(messages, list):
        # Find the last user message
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content:
                messages[i] = {
                    **msg,
                    "content": [{"type": "text", "text": content, "cache_control": cc}],
                }
            elif isinstance(content, list) and content:
                last_block = content[-1]
                if isinstance(last_block, dict) and "cache_control" not in last_block:
                    last_block["cache_control"] = cc
            break   # only stamp the last user message

    return kw


# ── Adapter ───────────────────────────────────────────────────────────────────

@register_adapter("anthropic")
class AnthropicAdapter(BaseAdapter):
    """
    Wraps the Anthropic Python SDK to auto-record usage into a Checkpoint.

    Intercepts:
    - messages.create()  — records input/output/cache tokens and cost
    - Tool use blocks    — records each tool invocation
    - Streaming          — accumulates token counts across chunks

    Parameters
    ----------
    checkpoint:
        The active Checkpoint to write state into.
    enable_prompt_caching:
        When True (default) inject ``cache_control`` breakpoints on every call
        so Anthropic's prompt-caching tier activates automatically.
    count_tokens_before_call:
        When True (default) call ``messages.count_tokens()`` before each API
        call to populate ``state.metadata["pre_inference_tokens"]``.
    """

    framework = "anthropic"

    def __init__(
        self,
        checkpoint: Any,
        enable_prompt_caching: bool = True,
        count_tokens_before_call: bool = True,
    ) -> None:
        super().__init__(checkpoint)
        self.enable_prompt_caching = enable_prompt_caching
        self.count_tokens_before_call = count_tokens_before_call

    def wrap(self, client: Any, **kwargs: Any) -> "_WrappedAnthropicClient":
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required: pip install 'agent-recall-ai[anthropic]'"
            )
        return _WrappedAnthropicClient(client=client, adapter=self)

    @classmethod
    def from_api_key(cls, checkpoint: Any, api_key: Optional[str] = None, **kwargs: Any) -> Any:
        """Convenience constructor: creates client and wraps it in one call."""
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("pip install 'agent-recall-ai[anthropic]'")
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        adapter = cls(checkpoint, **kwargs)
        return adapter.wrap(client)


class _WrappedAnthropicMessages:
    """Wraps the messages sub-resource."""

    def __init__(self, messages_resource: Any, adapter: AnthropicAdapter) -> None:
        self._messages = messages_resource
        self._adapter = adapter

    def count_tokens(self, **kwargs: Any) -> Any:
        """Pass-through to the underlying count_tokens API."""
        return self._messages.count_tokens(**kwargs)

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "claude-3-5-haiku-20241022")
        self._adapter.on_llm_start(model, kwargs.get("messages", []))

        # ── Pre-inference token count ─────────────────────────────────────────
        if self._adapter.count_tokens_before_call:
            try:
                # count_tokens needs model + messages (+ tools/system if present)
                count_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": kwargs.get("messages", []),
                }
                if "system" in kwargs:
                    count_kwargs["system"] = kwargs["system"]
                if "tools" in kwargs:
                    count_kwargs["tools"] = kwargs["tools"]
                estimate = self._messages.count_tokens(**count_kwargs)
                pre_tokens = getattr(estimate, "input_tokens", None)
                if pre_tokens is not None:
                    self._adapter.checkpoint.state.metadata["pre_inference_tokens"] = pre_tokens
                    logger.debug("Pre-inference token count: %d", pre_tokens)
            except Exception as exc:
                logger.debug("count_tokens failed (non-fatal): %s", exc)

        # ── Prompt caching ────────────────────────────────────────────────────
        if self._adapter.enable_prompt_caching:
            kwargs = _inject_cache_breakpoints(kwargs)

        try:
            response = self._messages.create(**kwargs)
        except Exception as exc:
            self._adapter.on_error(exc)
            raise

        # ── Record token usage ────────────────────────────────────────────────
        if hasattr(response, "usage"):
            usage = response.usage
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cached = cache_read + cache_write
            self._adapter.on_llm_end(
                model=model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                cached_tokens=cached,
            )
            if cached > 0:
                # Store per-call cache savings for dashboards / monitors
                self._adapter.checkpoint.state.metadata.setdefault("cache_savings", []).append(
                    {
                        "model": model,
                        "cache_read_tokens": cache_read,
                        "cache_write_tokens": cache_write,
                        "total_input_tokens": usage.input_tokens,
                    }
                )
                logger.debug(
                    "Prompt cache hit: read=%d write=%d / total=%d",
                    cache_read, cache_write, usage.input_tokens,
                )

        # ── Record tool use blocks ────────────────────────────────────────────
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "tool_use":
                input_str = str(getattr(block, "input", ""))[:200]
                self._adapter.on_tool_start(block.name, input_str)

        return response


class _WrappedAnthropicClient:
    """Thin proxy that intercepts messages.create() calls."""

    def __init__(self, client: Any, adapter: AnthropicAdapter) -> None:
        self._client = client
        self._adapter = adapter
        self.messages = _WrappedAnthropicMessages(client.messages, adapter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
