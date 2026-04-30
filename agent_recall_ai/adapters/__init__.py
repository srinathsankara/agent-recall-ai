"""
Plugin-based adapter system for agent-recall-ai.

Adapters wrap specific frameworks and provide uniform instrumentation:
    Anthropic SDK  →  AnthropicAdapter  (+ prompt caching + pre-inference token count)
    OpenAI SDK     →  OpenAIAdapter     (+ ConversationRepair)
    LangChain      →  LangChainAdapter
    LangGraph      →  LangGraphAdapter  (BaseCheckpointSaver drop-in)
    CrewAI         →  CrewAIAdapter
    smolagents     →  smolagentsAdapter

Registry:
    from agent_recall_ai.adapters import get_adapter, list_adapters

    AdapterClass = get_adapter("anthropic")
    adapter = AdapterClass(checkpoint)
    client = adapter.wrap(anthropic.Anthropic())
"""
from .base import BaseAdapter, get_adapter, list_adapters, register_adapter
from .anthropic_adapter import AnthropicAdapter, _inject_cache_breakpoints
from .openai_adapter import OpenAIAdapter, repair_conversation
from .langchain_adapter import LangChainAdapter
from .langgraph_adapter import LangGraphAdapter

# Optional adapters — import only if the framework is installed
try:
    from .crewai_adapter import CrewAIAdapter
    _CREWAI = True
except ImportError:
    _CREWAI = False

try:
    from .smolagents_adapter import smolagentsAdapter
    _SMOLAGENTS = True
except ImportError:
    _SMOLAGENTS = False

__all__ = [
    "BaseAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    # Core adapters (always available)
    "AnthropicAdapter",
    "OpenAIAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "repair_conversation",
    "_inject_cache_breakpoints",
    # Optional adapters
    *( ["CrewAIAdapter"] if _CREWAI else [] ),
    *( ["smolagentsAdapter"] if _SMOLAGENTS else [] ),
]
