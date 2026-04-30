"""
Example: Anthropic Claude agent with automatic checkpointing.

Demonstrates the AnthropicAdapter wrapping claude-haiku for a
data pipeline analysis task, with cost and drift monitoring.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_agent/agent.py
"""
from __future__ import annotations

import os

from agent_recall_ai import Checkpoint
from agent_recall_ai.adapters import AnthropicAdapter
from agent_recall_ai.monitors import CostMonitor, DriftMonitor, TokenMonitor


def run_data_pipeline_agent():
    """Analyse and design a data pipeline with full decision tracking."""

    with Checkpoint(
        "data-pipeline-design",
        model="claude-haiku-4-5",
        monitors=[
            CostMonitor(budget_usd=1.00, warn_at=0.75),
            TokenMonitor(model="claude-haiku-4-5", warn_at=0.70, compress_at=0.85),
            DriftMonitor(),
        ],
    ) as cp:
        cp.set_goal("Design a real-time data pipeline for processing 10M events/day")
        cp.add_constraint("Must be deployable on AWS with <$500/month infra cost")
        cp.add_constraint("P99 latency must be under 200ms")
        cp.set_context("E-commerce platform, primarily order events and inventory updates")

        print(f"Session: {cp.session_id}")

        if not os.getenv("ANTHROPIC_API_KEY"):
            _demo_mode(cp)
            return

        try:
            import anthropic
        except ImportError:
            print("Install anthropic: pip install 'agent-recall-ai[anthropic]'")
            _demo_mode(cp)
            return

        client = AnthropicAdapter(cp).wrap(anthropic.Anthropic())

        print("Consulting Claude on pipeline architecture...")
        response = client.messages(
            messages=[{"role": "user", "content": (
                "Design a real-time data pipeline for an e-commerce platform "
                "processing 10M events/day. Budget: $500/month. P99 < 200ms. "
                "What are the key technology choices and why?"
            )}],
            system="You are a senior data engineer specialising in real-time systems.",
            model="claude-haiku-4-5",
            max_tokens=1024,
        )

        content = response.content[0].text
        print(f"\nClaude's analysis: {content[:300]}...\n")

        # Record decisions based on the analysis
        cp.record_decision(
            "Use Apache Kafka for event streaming",
            reasoning="Handles 10M+ events/day easily. Managed MSK costs ~$200/month.",
            alternatives_rejected=["SQS (polling overhead)", "Kinesis (expensive at scale)"],
        )
        cp.record_decision(
            "Use Apache Flink on Fargate for stream processing",
            reasoning="Serverless Flink avoids EC2 management. Scales to zero when idle.",
            alternatives_rejected=["Spark Streaming (JVM overhead)", "custom Python consumer"],
        )

        cp.add_next_step("Prototype Kafka producer with order event schema")
        cp.add_next_step("Set up MSK cluster in staging")

        print(f"Tokens: {cp.token_total:,}  Cost: ${cp.cost_usd:.4f}")


def _demo_mode(cp: Checkpoint) -> None:
    print("(Demo mode — no API key)\n")
    cp.record_decision(
        "Kafka for event streaming, Flink for processing",
        reasoning="Handles 10M events/day at ~$200/month on MSK",
    )
    cp.record_tokens(prompt=8_000, completion=1_500)
    print(cp.state.resume_prompt())


if __name__ == "__main__":
    run_data_pipeline_agent()
