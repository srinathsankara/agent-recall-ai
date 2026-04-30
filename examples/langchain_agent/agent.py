"""
Example: LangChain agent with automatic checkpointing via CallbackHandler.

Demonstrates two integration modes:
1. CheckpointCallbackHandler — attach to any chain
2. CheckpointMessageHistory — drop-in BaseChatMessageHistory

Run:
    OPENAI_API_KEY=sk-... python examples/langchain_agent/agent.py
"""
from __future__ import annotations

import os

from agent_recall_ai import Checkpoint
from agent_recall_ai.adapters import LangChainAdapter
from agent_recall_ai.monitors import CostMonitor


def run_langchain_agent():
    """Demonstrate LangChain integration with checkpointing."""

    with Checkpoint(
        "langchain-research-agent",
        monitors=[CostMonitor(budget_usd=1.50)],
    ) as cp:
        cp.set_goal("Research and summarise recent advances in vector databases")
        cp.add_constraint("Focus only on open-source solutions")

        if not os.getenv("OPENAI_API_KEY"):
            _demo_mode(cp)
            return

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
        except ImportError:
            print("Install: pip install 'agent-recall-ai[langchain]' langchain-openai")
            _demo_mode(cp)
            return

        # Mode 1: Callback Handler
        handler = LangChainAdapter(cp).as_callback_handler()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical researcher. Be concise."),
            ("human", "{input}"),
        ])
        chain = prompt | llm

        print("Running LangChain chain with checkpoint handler...")
        result = chain.invoke(
            {"input": "What are the top 3 open-source vector databases in 2025 and their key differences?"},
            config={"callbacks": [handler]},
        )
        print(f"\nResult: {result.content[:300]}")
        print(f"\nTokens: {cp.token_total:,}  Cost: ${cp.cost_usd:.4f}")

        cp.record_decision(
            "Identified Qdrant, Weaviate, and Milvus as top candidates",
            reasoning="All have active communities, production deployments, and MIT/Apache licenses",
        )


def _demo_mode(cp: Checkpoint) -> None:
    print("(Demo mode)\n")
    cp.record_decision(
        "LangChain callback approach chosen over direct wrapping",
        reasoning="Works with any chain without modifying the chain code itself",
    )
    cp.record_tokens(prompt=5000, completion=800)
    print(cp.state.resume_prompt())


if __name__ == "__main__":
    run_langchain_agent()
