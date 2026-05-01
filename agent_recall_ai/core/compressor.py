"""
Semantic compressor — reduces tool output and conversation history before hitting context limit.

Strategy:
1. Tool output compression: truncate + summarize long tool responses
2. Decision log compression: keep last N decisions verbatim, summarize earlier ones
3. File list deduplication: collapse repeated references to the same file
"""
from __future__ import annotations

import re


def compress_tool_output(
    text: str,
    max_tokens: int = 500,
    max_chars: int | None = None,
) -> tuple[str, bool]:
    """
    Compress a tool output to fit within a token/character budget.
    Returns (compressed_text, was_compressed).

    Args:
        text:       The tool output string to compress.
        max_tokens: Approximate token budget (1 token ≈ 4 chars). Default 500.
        max_chars:  Explicit character budget. Overrides max_tokens if provided.

    Uses character count as a cheap proxy for tokens (1 token ≈ 4 chars).
    """
    max_chars = max_chars if max_chars is not None else (max_tokens * 4)
    if len(text) <= max_chars:
        return text, False

    # Extract high-signal lines first
    lines = text.splitlines()
    important_lines: list[str] = []
    filler_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Error lines, stack traces, return values, JSON keys are high-signal
        if any([
            re.search(r"error|warning|exception|traceback|failed|success", stripped, re.I),
            stripped.startswith(("{", "[", "}")),
            stripped.startswith(("return", "result", "output", "status")),
            len(stripped) < 10,  # short lines often matter (like exit codes)
        ]):
            important_lines.append(line)
        else:
            filler_lines.append(line)

    # Build compressed output
    head = text[:max_chars // 3]
    tail = text[-(max_chars // 6):]

    omitted = len(text) - len(head) - len(tail)
    compressed = (
        f"{head}\n"
        f"... [{omitted} characters compressed — {len(lines)} total lines] ...\n"
        f"{tail}"
    )
    return compressed, True


def compress_decision_log(decisions: list[dict], keep_recent: int = 5) -> list[dict]:
    """
    Compress a list of decisions by summarizing older ones.
    Keeps the last `keep_recent` decisions verbatim.
    """
    if len(decisions) <= keep_recent:
        return decisions

    older = decisions[:-keep_recent]
    recent = decisions[-keep_recent:]

    summaries = [d.get("summary", "") for d in older]
    compressed_older = {
        "summary": f"[Compressed: {len(older)} earlier decisions] " + "; ".join(summaries[:5]),
        "reasoning": "",
        "compressed": True,
        "alternatives_rejected": [],
    }
    return [compressed_older] + recent


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: 1 token ≈ 4 characters (English text)."""
    return max(1, len(text) // 4)


def compress_conversation_history(
    messages: list[dict],
    model_context_limit: int,
    target_utilization: float = 0.70,
    system_reserved: int = 500,
) -> tuple[list[dict], int]:
    """
    Compress a conversation history list to fit within target_utilization of context.

    Returns (compressed_messages, tokens_saved).
    Keeps: system message, last 4 user/assistant turns, compresses earlier tool results
    and long messages.
    """
    if not messages:
        return messages, 0

    target_tokens = max(100, int(model_context_limit * target_utilization) - system_reserved)
    current_estimate = sum(estimate_tokens(str(m)) for m in messages)

    if current_estimate <= target_tokens:
        return messages, 0

    tokens_saved = 0
    result = []
    # Each non-protected message gets at most this many tokens before being compressed
    per_message_limit = max(20, target_tokens // max(len(messages), 1))

    running_total = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        content_str = str(content)
        original_tokens = estimate_tokens(content_str)

        # Always keep system messages and last 4 messages verbatim
        if role == "system" or i >= len(messages) - 4:
            result.append(msg)
            running_total += original_tokens
            continue

        # If we're still under budget, keep as-is
        if running_total + original_tokens <= target_tokens:
            result.append(msg)
            running_total += original_tokens
            continue

        # Over budget: compress tool results and long messages
        if role == "tool" or original_tokens > per_message_limit:
            compressed, was_compressed = compress_tool_output(content_str, max_tokens=per_message_limit)
            compressed_tokens = estimate_tokens(compressed)
            tokens_saved += original_tokens - compressed_tokens
            result.append({**msg, "content": compressed})
            running_total += compressed_tokens
        else:
            # Drop this message — budget exhausted; insert a notice so the model
            # knows context was trimmed rather than silently losing information.
            tokens_saved += original_tokens
            result.append({
                "role": "system",
                "content": f"[1 message omitted — context budget exhausted ({original_tokens} tokens saved)]",
            })

    return result, tokens_saved


def build_resume_context(state_dict, max_tokens: int = 2000) -> str:
    """
    Build a compact resume context string from a TaskState or its dict representation.
    Intended to be prepended to the system prompt when resuming.

    Args:
        state_dict: A TaskState object or a dict from TaskState.model_dump().
        max_tokens: Approximate maximum token budget for the output.
    """
    # Accept both a TaskState object and a plain dict
    if hasattr(state_dict, "model_dump"):
        state_dict = state_dict.model_dump()

    parts: list[str] = []

    goals = state_dict.get("goals", [])
    if goals:
        parts.append("GOALS: " + " | ".join(goals))

    constraints = state_dict.get("constraints", [])
    if constraints:
        parts.append("CONSTRAINTS: " + " | ".join(constraints))

    decisions = state_dict.get("decisions", [])
    if decisions:
        recent = decisions[-5:]
        summaries = [d.get("summary", "") for d in recent]
        parts.append("RECENT DECISIONS: " + " → ".join(summaries))

    files = state_dict.get("files_modified", [])
    if files:
        paths = [f.get("path", "") if isinstance(f, dict) else str(f) for f in files]
        unique_paths = list(dict.fromkeys(paths))[:20]   # keep order, deduplicate
        parts.append("FILES TOUCHED: " + ", ".join(unique_paths))

    next_steps = state_dict.get("next_steps", [])
    if next_steps:
        parts.append("NEXT STEPS: " + " | ".join(next_steps[:5]))

    context = "\n".join(parts)
    # Trim to token budget
    max_chars = max_tokens * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[context truncated]"
    return context
