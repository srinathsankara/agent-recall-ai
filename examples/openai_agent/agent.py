"""
Example: OpenAI agent with automatic checkpointing.

This agent performs a multi-step code refactoring task. Without checkpointing,
a long session would lose all decisions and progress when the context window fills.
With agent-recall-ai, every decision is preserved and the session is resumable.

Run:
    OPENAI_API_KEY=sk-... python examples/openai_agent/agent.py

To resume a dead session:
    python -c "
    from agent_recall_ai import resume
    state = resume('code-refactor-demo')
    print(state.resume_prompt())
    "
"""
from __future__ import annotations

import os
import sys

# Check for API key before importing OpenAI
if not os.getenv("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY to run this example")
    print("Demo mode: showing what the checkpoint would look like")

from agent_recall_ai import Checkpoint
from agent_recall_ai.adapters import OpenAIAdapter
from agent_recall_ai.monitors import CostMonitor, TokenMonitor, DriftMonitor


def run_refactor_agent():
    """Simulate a multi-step code refactoring agent with full checkpointing."""

    with Checkpoint(
        "code-refactor-demo",
        model="gpt-4o-mini",
        monitors=[
            CostMonitor(budget_usd=2.00, warn_at=0.70),
            TokenMonitor(model="gpt-4o-mini", warn_at=0.75, compress_at=0.88),
            DriftMonitor(sensitivity="medium"),
        ],
    ) as cp:
        # Define the task
        cp.set_goal("Refactor the authentication module from session-based to JWT")
        cp.set_goal("Maintain full backward compatibility with existing API endpoints")
        cp.add_constraint("Do not change the public API surface (/login, /logout, /me)")
        cp.add_constraint("Must pass all existing tests without modification")
        cp.set_context(
            "Working on a FastAPI application. Auth is in auth/session.py. "
            "The app has 3000+ MAU and cannot have downtime."
        )

        print(f"Session: {cp.session_id}")
        print(f"Goals: {cp.state.goals}")
        print()

        if not os.getenv("OPENAI_API_KEY"):
            # Demo mode — simulate decisions without real API calls
            _simulate_demo_session(cp)
            return

        try:
            import openai
        except ImportError:
            print("Install openai: pip install 'agent-recall-ai[openai]'")
            return

        client = OpenAIAdapter(cp).wrap(openai.OpenAI())

        # Step 1: Analyse existing code
        print("Step 1: Analysing existing auth implementation...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python backend engineer specialising in security.",
                },
                {
                    "role": "user",
                    "content": (
                        "I need to refactor a session-based auth system to use JWT. "
                        "The current implementation uses Flask-Login with database-backed sessions. "
                        "What are the key decisions I need to make before starting?"
                    ),
                },
            ],
        )

        analysis = response.choices[0].message.content
        print(f"Analysis: {analysis[:200]}...")

        # Record architectural decisions
        cp.record_decision(
            "Use PyJWT library for JWT implementation",
            reasoning="PyJWT is the most widely used Python JWT library with 10M+ weekly downloads. "
                      "python-jose is slower and less maintained.",
            alternatives_rejected=["python-jose", "authlib", "python-jwt"],
        )

        cp.record_decision(
            "Store refresh tokens in Redis, not the database",
            reasoning="Redis provides O(1) lookup and built-in TTL expiry. "
                      "Database-backed tokens require a query on every request.",
            alternatives_rejected=["database table", "in-memory dict"],
        )

        cp.record_file_modified("auth/session.py", action="modified", description="Add JWT generation")
        cp.record_file_modified("auth/middleware.py", action="modified", description="Add JWT validation")
        cp.record_file_modified("auth/models.py", action="modified", description="Add TokenBlacklist model")
        cp.add_next_step("Implement token refresh endpoint")
        cp.add_next_step("Update integration tests")
        cp.add_open_question("Should we support multiple active sessions per user?")

        print(f"\nTokens used: {cp.token_total:,}")
        print(f"Cost so far: ${cp.cost_usd:.4f}")
        print(f"\nCheckpoint saved. To resume:\n  agent-recall-ai resume {cp.session_id}")


def _simulate_demo_session(cp: Checkpoint) -> None:
    """Demo mode — shows what a real session would produce."""
    print("(Demo mode — no API calls made)\n")

    cp.record_decision(
        "Use PyJWT library for JWT implementation",
        reasoning="PyJWT has 10M+ weekly downloads, well-maintained, pure Python",
        alternatives_rejected=["python-jose", "authlib"],
    )
    cp.record_decision(
        "Store refresh tokens in Redis",
        reasoning="O(1) lookup, built-in TTL, no DB queries on hot path",
        alternatives_rejected=["PostgreSQL table", "in-memory cache"],
    )
    cp.record_file_modified("auth/session.py", action="modified")
    cp.record_file_modified("auth/middleware.py", action="modified")
    cp.add_next_step("Implement /token/refresh endpoint")
    cp.add_next_step("Update Postman collection")
    cp.add_open_question("Multiple active sessions per user?")

    # Simulate some token usage
    cp.record_tokens(prompt=15_000, completion=3_000)

    print("Checkpoint created!")
    print(f"\nSession ID: {cp.session_id}")
    print(f"Decisions recorded: {len(cp.state.decisions)}")
    print(f"Files touched: {len(cp.state.files_modified)}")
    print(f"Next steps: {len(cp.state.next_steps)}")
    print(f"Simulated tokens: {cp.token_total:,}")
    print(f"Simulated cost: ${cp.cost_usd:.4f}")
    print()
    print("Resume prompt preview:")
    print("-" * 60)
    print(cp.state.resume_prompt())


if __name__ == "__main__":
    run_refactor_agent()
