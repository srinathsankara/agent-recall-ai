"""
OpenAIAdapter — intercepts GPT-4o / OpenAI SDK calls.

Solves the context-loss pattern from:
  github.com/openai/codex/issues/3997

Supports:
  - chat.completions.create() (sync + async)
  - Function/tool call recording
  - Token usage including prompt caching (o1, gpt-4o-mini cached reads)
  - ConversationRepair: detects orphaned tool_call IDs and injects synthetic
    error ToolMessages so the API never receives a corrupted history

ConversationRepair background
------------------------------
When an agent loop is interrupted mid-turn (context kill, network error,
process restart) the message history can contain an assistant message with
tool_calls that have no matching tool result.  The OpenAI API rejects this
with a 400 "invalid_request_error" that is often hard to debug.

ConversationRepair scans messages before each API call and injects a
synthetic error result for every orphaned call_id:

    {"role": "tool", "tool_call_id": "<id>",
     "content": "Error: tool result was not received (session interrupted)"}

This keeps the history valid while making the failure explicit to the model,
so it can recover gracefully instead of looping on a corrupt state.

Usage:
    from agent_recall_ai import Checkpoint
    from agent_recall_ai.adapters import OpenAIAdapter

    with Checkpoint("my-task") as cp:
        adapter = OpenAIAdapter(cp)
        client = adapter.wrap(openai.OpenAI())

        # Messages with orphaned tool calls are auto-repaired before each call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
        )
"""
from __future__ import annotations

import logging
from typing import Any

from .base import BaseAdapter, register_adapter

logger = logging.getLogger(__name__)

try:
    import openai as _openai_sdk  # noqa: F401
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ── ConversationRepair ────────────────────────────────────────────────────────

def repair_conversation(messages: list[dict]) -> tuple[list[dict], int]:
    """
    Scan a message list for orphaned tool_call IDs and inject synthetic
    error tool results so the OpenAI API receives a valid history.

    Returns:
        (repaired_messages, n_repaired) — n_repaired is 0 when no changes were made.

    Algorithm:
        1. Collect all tool_call IDs referenced in assistant messages.
        2. Collect all tool_call IDs that have a matching 'tool' message.
        3. For each missing match, inject a synthetic error ToolMessage
           immediately after the assistant message that introduced the call.
    """
    if not messages:
        return messages, 0

    # Pass 1: find all tool_call IDs declared by assistant messages, in order
    declared: list[tuple[int, str]] = []   # (message_index, call_id)
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                call_id = (tc.get("id") or "")
                if call_id:
                    declared.append((i, call_id))

    if not declared:
        return messages, 0

    # Pass 2: find all tool_call IDs that already have a result
    resolved: set[str] = {
        msg["tool_call_id"]
        for msg in messages
        if msg.get("role") == "tool" and msg.get("tool_call_id")
    }

    orphaned = [(idx, cid) for idx, cid in declared if cid not in resolved]
    if not orphaned:
        return messages, 0

    # Pass 3: inject synthetic error messages after the assistant message
    # Work in reverse order so index offsets stay valid
    repaired = list(messages)
    for assistant_idx, call_id in reversed(orphaned):
        synthetic = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": (
                "Error: tool result was not received. "
                "The session was interrupted before this tool call completed. "
                "Please retry or skip this step."
            ),
        }
        insert_at = assistant_idx + 1
        repaired.insert(insert_at, synthetic)
        logger.warning(
            "ConversationRepair: injected synthetic error for orphaned tool_call_id=%s "
            "(assistant message index=%d)",
            call_id, assistant_idx,
        )

    return repaired, len(orphaned)


# ── Adapter ───────────────────────────────────────────────────────────────────

@register_adapter("openai")
class OpenAIAdapter(BaseAdapter):
    """
    Wraps the OpenAI Python SDK to auto-record usage into a Checkpoint.

    Intercepts:
    - chat.completions.create() — records prompt/completion/cached tokens
    - Tool calls in the response — records each function invocation
    - Orphaned tool_call IDs — repaired via ConversationRepair before each call
    """

    framework = "openai"

    def __init__(self, checkpoint: Any, repair_conversations: bool = True) -> None:
        super().__init__(checkpoint)
        self.repair_conversations = repair_conversations

    def wrap(self, client: Any, **kwargs: Any) -> _WrappedOpenAIClient:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required: pip install 'agent-recall-ai[openai]'"
            )
        return _WrappedOpenAIClient(client=client, adapter=self)

    @classmethod
    def from_api_key(cls, checkpoint: Any, api_key: str | None = None, **kwargs: Any) -> Any:
        if not _OPENAI_AVAILABLE:
            raise ImportError("pip install 'agent-recall-ai[openai]'")
        import openai
        client = openai.OpenAI(api_key=api_key, **kwargs)
        adapter = cls(checkpoint)
        return adapter.wrap(client)


class _WrappedChatCompletions:
    def __init__(self, completions: Any, adapter: OpenAIAdapter) -> None:
        self._completions = completions
        self._adapter = adapter

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "gpt-4o-mini")
        messages = kwargs.get("messages", [])

        # ── ConversationRepair — fix orphaned tool_call IDs before the call ──
        if self._adapter.repair_conversations and messages:
            repaired, n = repair_conversation(messages)
            if n > 0:
                kwargs = {**kwargs, "messages": repaired}
                from ..core.state import AlertSeverity, AlertType
                self._adapter.checkpoint.state.add_alert(
                    alert_type=AlertType.BEHAVIORAL_DRIFT,
                    severity=AlertSeverity.WARN,
                    message=f"ConversationRepair: fixed {n} orphaned tool_call ID(s) before API call",
                    detail={"repaired_count": n},
                )

        self._adapter.on_llm_start(model, kwargs.get("messages", []))

        try:
            response = self._completions.create(**kwargs)
        except Exception as exc:
            self._adapter.on_error(exc)
            raise

        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            cached = 0
            if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            self._adapter.on_llm_end(
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=cached,
            )

        # Record tool/function calls
        for choice in getattr(response, "choices", []):
            msg = getattr(choice, "message", None)
            if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = tc.function
                    self._adapter.on_tool_start(fn.name, fn.arguments[:200] if fn.arguments else "")

        return response


class _WrappedChat:
    def __init__(self, chat: Any, adapter: OpenAIAdapter) -> None:
        self.completions = _WrappedChatCompletions(chat.completions, adapter)
        self._chat = chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _WrappedOpenAIClient:
    def __init__(self, client: Any, adapter: OpenAIAdapter) -> None:
        self._client = client
        self.chat = _WrappedChat(client.chat, adapter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
