"""Tests for the plugin-based adapter registry and adapters."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent_recall_ai import Checkpoint
from agent_recall_ai.adapters.base import (
    BaseAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    _ADAPTER_REGISTRY,
)
from agent_recall_ai.adapters import AnthropicAdapter, OpenAIAdapter, LangChainAdapter
from agent_recall_ai.storage.memory import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def cp(store):
    checkpoint = Checkpoint("adapter-test", store=store)
    return checkpoint


class TestAdapterRegistry:
    def test_anthropic_registered(self):
        assert "anthropic" in list_adapters()

    def test_openai_registered(self):
        assert "openai" in list_adapters()

    def test_langchain_registered(self):
        assert "langchain" in list_adapters()

    def test_get_adapter_returns_class(self):
        cls = get_adapter("anthropic")
        assert cls is AnthropicAdapter

    def test_get_adapter_case_insensitive(self):
        cls = get_adapter("ANTHROPIC")
        assert cls is AnthropicAdapter

    def test_get_adapter_unknown_raises(self):
        with pytest.raises(KeyError, match="No adapter registered"):
            get_adapter("nonexistent-framework")

    def test_register_custom_adapter(self, cp):
        @register_adapter("my-custom-framework")
        class MyAdapter(BaseAdapter):
            framework = "my-custom-framework"
            def wrap(self, client, **kwargs):
                return client

        assert "my-custom-framework" in list_adapters()
        # Cleanup
        _ADAPTER_REGISTRY.pop("my-custom-framework", None)


class TestAnthropicAdapter:
    def test_sets_framework_metadata(self, cp):
        adapter = AnthropicAdapter(cp)
        assert cp.state.metadata.get("framework") == "anthropic"

    def test_on_llm_end_records_tokens(self, cp):
        adapter = AnthropicAdapter(cp)
        adapter.on_llm_end(
            model="claude-haiku-4-5",
            prompt_tokens=5000,
            completion_tokens=500,
            cached_tokens=1000,
        )
        assert cp.state.token_usage.prompt == 5000
        assert cp.state.token_usage.completion == 500

    def test_on_error_triggers_save(self, cp, store):
        adapter = AnthropicAdapter(cp)
        cp.set_goal("Some goal")
        adapter.on_error(RuntimeError("connection failed"))
        # Should have been saved
        assert store.exists("adapter-test")

    def test_wrap_without_anthropic_raises(self, cp):
        adapter = AnthropicAdapter(cp)
        with patch.dict("sys.modules", {"anthropic": None}):
            # Simulate package not available
            import agent_recall_ai.adapters.anthropic_adapter as mod
            original = mod._ANTHROPIC_AVAILABLE
            mod._ANTHROPIC_AVAILABLE = False
            try:
                with pytest.raises(ImportError):
                    adapter.wrap(MagicMock())
            finally:
                mod._ANTHROPIC_AVAILABLE = original


class TestOpenAIAdapter:
    def test_sets_framework_metadata(self, cp):
        adapter = OpenAIAdapter(cp)
        assert cp.state.metadata.get("framework") == "openai"

    def test_on_llm_end_records_tokens(self, cp):
        adapter = OpenAIAdapter(cp)
        adapter.on_llm_end(
            model="gpt-4o-mini",
            prompt_tokens=3000,
            completion_tokens=300,
        )
        assert cp.state.token_usage.prompt == 3000
        assert cp.state.token_usage.completion == 300

    def test_on_tool_end_records_call(self, cp):
        adapter = OpenAIAdapter(cp)
        adapter.on_tool_end("web_search", "Python documentation for asyncio")
        assert len(cp.state.tool_calls) == 1
        assert cp.state.tool_calls[0].tool_name == "web_search"


class TestLangChainAdapter:
    def test_sets_framework_metadata(self, cp):
        adapter = LangChainAdapter(cp)
        assert cp.state.metadata.get("framework") == "langchain"

    def test_wrap_returns_callback_handler(self, cp):
        try:
            from langchain_core.callbacks import BaseCallbackHandler
            adapter = LangChainAdapter(cp)
            handler = adapter.wrap(MagicMock())
            assert isinstance(handler, BaseCallbackHandler)
        except ImportError:
            pytest.skip("langchain-core not installed")

    def test_as_callback_handler_without_langchain(self, cp):
        import agent_recall_ai.adapters.langchain_adapter as mod
        if not mod._LANGCHAIN_AVAILABLE:
            adapter = LangChainAdapter(cp)
            with pytest.raises(ImportError):
                adapter.as_callback_handler()
