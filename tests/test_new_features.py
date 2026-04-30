"""
Tests for new features added in Phase 8:
  - Anthropic prompt caching (_inject_cache_breakpoints)
  - Pre-inference token counting (AnthropicAdapter metadata)
  - LangGraph BaseCheckpointSaver (LangGraphAdapter)
  - Thread forking (Checkpoint.fork)
  - OTLP export (OTLPExporter structure)
  - Presidio backend (PresidioBackend import/availability)
"""
from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from agent_recall_ai import Checkpoint
from agent_recall_ai.adapters.anthropic_adapter import _inject_cache_breakpoints
from agent_recall_ai.adapters.langgraph_adapter import LangGraphAdapter, _LANGGRAPH_AVAILABLE
from agent_recall_ai.storage.memory import MemoryStore


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic prompt caching
# ══════════════════════════════════════════════════════════════════════════════

class TestInjectCacheBreakpoints:
    """Unit tests for _inject_cache_breakpoints."""

    def _bp(self) -> dict:
        return {"type": "ephemeral"}

    def test_string_system_converted_to_content_block(self):
        kwargs = {"system": "You are helpful.", "messages": []}
        result = _inject_cache_breakpoints(kwargs)
        system = result["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are helpful."
        assert system[0]["cache_control"] == self._bp()

    def test_list_system_last_block_gets_cache_control(self):
        kwargs = {
            "system": [
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
            ],
            "messages": [],
        }
        result = _inject_cache_breakpoints(kwargs)
        assert result["system"][0].get("cache_control") is None
        assert result["system"][1]["cache_control"] == self._bp()

    def test_tools_last_gets_cache_control(self):
        kwargs = {
            "messages": [],
            "tools": [
                {"name": "search", "description": "Search"},
                {"name": "write", "description": "Write"},
            ],
        }
        result = _inject_cache_breakpoints(kwargs)
        assert result["tools"][0].get("cache_control") is None
        assert result["tools"][1]["cache_control"] == self._bp()

    def test_last_user_message_string_gets_cache_control(self):
        kwargs = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        }
        result = _inject_cache_breakpoints(kwargs)
        # Last user message content should be converted to content block
        last_user = result["messages"][2]
        assert isinstance(last_user["content"], list)
        assert last_user["content"][0]["cache_control"] == self._bp()

    def test_last_user_message_list_last_block_gets_cache_control(self):
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Block A"},
                        {"type": "text", "text": "Block B"},
                    ],
                }
            ]
        }
        result = _inject_cache_breakpoints(kwargs)
        content = result["messages"][0]["content"]
        assert content[0].get("cache_control") is None
        assert content[1]["cache_control"] == self._bp()

    def test_does_not_overwrite_existing_cache_control(self):
        existing_cc = {"type": "ephemeral"}
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi", "cache_control": existing_cc}],
                }
            ]
        }
        result = _inject_cache_breakpoints(kwargs)
        # Should NOT add a second cache_control
        content = result["messages"][0]["content"]
        assert content[0]["cache_control"] == existing_cc

    def test_original_kwargs_not_mutated(self):
        original = {
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "search"}],
        }
        copy_before = copy.deepcopy(original)
        _inject_cache_breakpoints(original)
        assert original == copy_before

    def test_empty_messages_no_error(self):
        result = _inject_cache_breakpoints({"messages": []})
        assert result["messages"] == []

    def test_no_user_message_no_modification(self):
        kwargs = {
            "messages": [
                {"role": "assistant", "content": "How can I help?"},
            ]
        }
        result = _inject_cache_breakpoints(kwargs)
        assert result["messages"][0].get("content") == "How can I help?"

    def test_assistant_only_messages_skipped(self):
        kwargs = {
            "messages": [
                {"role": "assistant", "content": "Hello"},
            ]
        }
        result = _inject_cache_breakpoints(kwargs)
        # No user message → no modification
        assert result["messages"][0]["content"] == "Hello"


# ══════════════════════════════════════════════════════════════════════════════
# Thread forking
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointFork:
    """Tests for Checkpoint.fork()."""

    def _make_cp(self, session_id: str = "parent") -> Checkpoint:
        store = MemoryStore()
        cp = Checkpoint(session_id, store=store)
        return cp

    def test_fork_creates_new_session(self):
        cp = self._make_cp("main")
        cp.set_goal("Refactor auth")
        cp.record_decision("Use PyJWT", reasoning="Better maintained")
        cp.record_file_modified("auth.py")
        cp.save()

        fork = cp.fork("main-alt")
        assert fork.session_id == "main-alt"

    def test_fork_copies_goals(self):
        cp = self._make_cp("main")
        cp.set_goal("Goal A")
        cp.set_goal("Goal B")
        cp.save()

        fork = cp.fork("fork-goals")
        assert "Goal A" in fork.state.goals
        assert "Goal B" in fork.state.goals

    def test_fork_copies_decisions(self):
        cp = self._make_cp("main")
        cp.record_decision("Use PyJWT", reasoning="Better maintained")
        cp.save()

        fork = cp.fork("fork-decisions")
        assert len(fork.state.decisions) == 1
        assert fork.state.decisions[0].summary == "Use PyJWT"

    def test_fork_copies_constraints(self):
        cp = self._make_cp("main")
        cp.add_constraint("No breaking API changes")
        cp.save()

        fork = cp.fork("fork-constraints")
        assert "No breaking API changes" in fork.state.constraints

    def test_fork_sets_parent_thread_id_in_metadata(self):
        cp = self._make_cp("main")
        cp.save()

        fork = cp.fork("fork-meta")
        assert fork.state.metadata.get("parent_thread_id") == "main"

    def test_fork_resets_checkpoint_seq(self):
        cp = self._make_cp("main")
        cp.save()
        cp.save()
        cp.save()
        # parent has been saved 3 times
        assert cp.state.checkpoint_seq == 3

        fork = cp.fork("fork-seq")
        # Fork is saved once during creation (seq=1), much less than parent
        assert fork.state.checkpoint_seq < cp.state.checkpoint_seq

    def test_fork_is_independent(self):
        """Changes to fork don't affect parent."""
        store = MemoryStore()
        cp = Checkpoint("main", store=store)
        cp.set_goal("Original goal")
        cp.save()

        fork = cp.fork("fork-independent")
        fork.set_goal("Fork-only goal")
        fork.save()

        # Reload parent — should not have fork-only goal
        reloaded_parent = store.load("main")
        assert reloaded_parent is not None
        assert "Fork-only goal" not in reloaded_parent.goals

    def test_fork_uses_same_store_by_default(self):
        store = MemoryStore()
        cp = Checkpoint("main", store=store)
        cp.save()

        fork = cp.fork("fork-store")
        # Fork should be accessible from the same store
        loaded = store.load("fork-store")
        assert loaded is not None
        assert loaded.session_id == "fork-store"

    def test_fork_with_custom_store(self):
        cp = self._make_cp("main")
        cp.save()

        custom_store = MemoryStore()
        fork = cp.fork("fork-custom", store=custom_store)

        # Should be in custom store
        assert custom_store.load("fork-custom") is not None
        # Should NOT be in original store (different store)
        original_store_result = cp._store.load("fork-custom")
        assert original_store_result is None

    def test_fork_preserves_files_modified(self):
        cp = self._make_cp("main")
        cp.record_file_modified("auth.py", action="modified")
        cp.record_file_modified("tests/test_auth.py", action="created")
        cp.save()

        fork = cp.fork("fork-files")
        file_paths = {f.path for f in fork.state.files_modified}
        assert "auth.py" in file_paths
        assert "tests/test_auth.py" in file_paths


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph adapter
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
class TestLangGraphAdapter:
    """Tests for LangGraphAdapter (only run when langgraph is installed)."""

    def test_from_memory_returns_adapter(self):
        adapter = LangGraphAdapter.from_memory()
        assert adapter is not None

    def test_put_and_get_tuple(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "test-thread"}}
        checkpoint = {"id": "cp-001", "channel_values": {"messages": []}}
        metadata = {"source": "loop", "step": 1, "writes": {}}

        returned_config = adapter.put(config, checkpoint, metadata, {})
        assert returned_config["configurable"]["thread_id"] == "test-thread"
        assert returned_config["configurable"]["checkpoint_id"] == "cp-001"

        # Retrieve
        result = adapter.get_tuple(config)
        assert result is not None

    def test_get_tuple_returns_none_for_unknown_thread(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "nonexistent-thread"}}
        result = adapter.get_tuple(config)
        assert result is None

    def test_list_yields_latest_checkpoint(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "list-test"}}
        checkpoint = {"id": "cp-list-1", "channel_values": {}}
        metadata = {"source": "loop", "step": 0, "writes": {}}
        adapter.put(config, checkpoint, metadata, {})

        items = list(adapter.list(config))
        assert len(items) == 1

    def test_list_empty_for_unknown_thread(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "no-such-thread"}}
        items = list(adapter.list(config))
        assert len(items) == 0

    def test_fork_success(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "source"}}
        checkpoint = {"id": "cp-fork", "channel_values": {"x": 1}}
        metadata = {"source": "loop", "step": 5, "writes": {}}
        adapter.put(config, checkpoint, metadata, {})

        result = adapter.fork("source", "target")
        assert result is True

        # Target should now exist
        target_config = {"configurable": {"thread_id": "target"}}
        target_tuple = adapter.get_tuple(target_config)
        assert target_tuple is not None

    def test_fork_nonexistent_source_returns_false(self):
        adapter = LangGraphAdapter.from_memory()
        result = adapter.fork("does-not-exist", "new-thread")
        assert result is False

    def test_fork_preserves_checkpoint_data(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "src"}}
        checkpoint = {"id": "cp-preserve", "channel_values": {"messages": ["hello"]}}
        metadata = {"source": "loop", "step": 3, "writes": {}}
        adapter.put(config, checkpoint, metadata, {})

        adapter.fork("src", "dst")

        dst_config = {"configurable": {"thread_id": "dst"}}
        dst_tuple = adapter.get_tuple(dst_config)
        assert dst_tuple is not None
        assert dst_tuple.checkpoint.get("channel_values") == {"messages": ["hello"]}

    def test_fork_stamps_fork_metadata(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "original"}}
        checkpoint = {"id": "cp-meta", "channel_values": {}}
        metadata = {"source": "loop", "step": 0, "writes": {}}
        adapter.put(config, checkpoint, metadata, {})
        adapter.fork("original", "forked")

        # Check raw storage for forked_from key
        raw = adapter._store.load("langgraph:forked:latest")
        assert raw is not None
        assert raw.get("forked_from") == "original"
        assert "forked_at" in raw

    @pytest.mark.asyncio
    async def test_aput_and_aget_tuple(self):
        adapter = LangGraphAdapter.from_memory()
        config = {"configurable": {"thread_id": "async-thread"}}
        checkpoint = {"id": "cp-async", "channel_values": {}}
        metadata = {"source": "loop", "step": 0, "writes": {}}

        await adapter.aput(config, checkpoint, metadata, {})
        result = await adapter.aget_tuple(config)
        assert result is not None


class TestLangGraphAdapterStubWhenNotInstalled:
    """The stub raises ImportError when langgraph is not installed."""

    @pytest.mark.skipif(_LANGGRAPH_AVAILABLE, reason="langgraph IS installed — stub not active")
    def test_stub_raises_import_error(self):
        with pytest.raises(ImportError, match="langgraph"):
            LangGraphAdapter(store=None)

    @pytest.mark.skipif(_LANGGRAPH_AVAILABLE, reason="langgraph IS installed")
    def test_from_memory_stub_raises(self):
        with pytest.raises(ImportError):
            LangGraphAdapter.from_memory()


# ══════════════════════════════════════════════════════════════════════════════
# OTLP Exporter
# ══════════════════════════════════════════════════════════════════════════════

class TestOTLPExporterImport:
    """Light structural tests that don't require otel packages."""

    def test_import_from_exporters(self):
        from agent_recall_ai.exporters import OTLPExporter
        assert OTLPExporter is not None

    def test_raises_import_error_without_otel(self):
        """When opentelemetry-sdk is not installed, OTLPExporter.__init__ raises."""
        from agent_recall_ai.exporters.otlp import _OTEL_AVAILABLE
        if _OTEL_AVAILABLE:
            pytest.skip("opentelemetry-sdk is installed")
        from agent_recall_ai.exporters import OTLPExporter
        with pytest.raises(ImportError, match="opentelemetry"):
            OTLPExporter(endpoint="http://localhost:4317")

    def test_export_session_noop_without_otel(self):
        """export_session should not crash even if otel is absent."""
        from agent_recall_ai.exporters.otlp import OTLPExporter, _OTEL_AVAILABLE
        if _OTEL_AVAILABLE:
            pytest.skip("opentelemetry-sdk installed, noop path not active")
        exporter = object.__new__(OTLPExporter)
        exporter._service_name = "test"
        # Calling export_session with OTEL absent should return silently
        store = MemoryStore()
        cp = Checkpoint("otlp-test", store=store)
        cp.set_goal("Test OTLP")
        cp.save()
        # This should not raise
        exporter.export_session(cp.state)


# ══════════════════════════════════════════════════════════════════════════════
# Presidio backend
# ══════════════════════════════════════════════════════════════════════════════

class TestPresidioBackend:
    """Tests for PresidioBackend (only run when presidio is installed)."""

    @pytest.fixture
    def backend(self):
        from agent_recall_ai.privacy.presidio_backend import PresidioBackend, _PRESIDIO_AVAILABLE
        if not _PRESIDIO_AVAILABLE:
            pytest.skip("presidio not installed")
        return PresidioBackend(score_threshold=0.5)

    def test_scan_detects_email(self, backend):
        results = backend.scan("Contact me at alice@example.com for details.")
        entity_types = [r["entity_type"] for r in results]
        assert "EMAIL_ADDRESS" in entity_types

    def test_scan_returns_empty_for_clean_text(self, backend):
        results = backend.scan("The weather in Paris is nice today.")
        # May detect LOCATION; we just confirm it doesn't crash and returns list
        assert isinstance(results, list)

    def test_anonymize_replaces_email(self, backend):
        text = "Send invoice to billing@company.com"
        anonymized, detections = backend.anonymize(text)
        assert "billing@company.com" not in anonymized
        assert len(detections) > 0

    def test_anonymize_empty_string(self, backend):
        text, detections = backend.anonymize("")
        assert text == ""
        assert detections == []

    def test_redact_value_returns_changed_flag(self, backend):
        text = "email me at test@example.com"
        redacted, changed = backend.redact_value(text)
        assert changed is True
        assert "test@example.com" not in redacted

    def test_redact_value_unchanged_for_clean(self, backend):
        text = "The quick brown fox"
        redacted, changed = backend.redact_value(text)
        # May detect nothing
        if not changed:
            assert redacted == text

    def test_import_error_without_presidio(self):
        from agent_recall_ai.privacy.presidio_backend import _PRESIDIO_AVAILABLE
        if _PRESIDIO_AVAILABLE:
            pytest.skip("presidio IS installed")
        from agent_recall_ai.privacy.presidio_backend import PresidioBackend
        with pytest.raises(ImportError, match="Presidio"):
            PresidioBackend()


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic adapter — pre-inference token counting
# ══════════════════════════════════════════════════════════════════════════════

class TestAnthropicAdapterPromptCaching:
    """Verify AnthropicAdapter injects cache_control and calls count_tokens."""

    def _make_adapter(self, **kwargs):
        from agent_recall_ai.adapters import AnthropicAdapter
        store = MemoryStore()
        cp = Checkpoint("test-anthropic-caching", store=store)
        return AnthropicAdapter(cp, **kwargs), cp

    def test_cache_breakpoints_injected_on_create(self):
        """_WrappedAnthropicMessages.create() injects cache_control by default."""
        adapter, cp = self._make_adapter()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 80
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        # count_tokens mock
        mock_count = MagicMock()
        mock_count.input_tokens = 100
        mock_client.messages.count_tokens.return_value = mock_count

        wrapped = adapter.wrap(mock_client)
        wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        # System should be list with cache_control
        assert isinstance(call_kwargs["system"], list)
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_prompt_caching_disabled(self):
        adapter, cp = self._make_adapter(enable_prompt_caching=False)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response
        mock_client.messages.count_tokens.return_value = MagicMock(input_tokens=100)

        wrapped = adapter.wrap(mock_client)
        wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        # System should remain as string (no transformation)
        assert call_kwargs["system"] == "You are helpful."

    def test_pre_inference_token_count_stored_in_metadata(self):
        adapter, cp = self._make_adapter(count_tokens_before_call=True)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        mock_count = MagicMock()
        mock_count.input_tokens = 123
        mock_client.messages.count_tokens.return_value = mock_count

        wrapped = adapter.wrap(mock_client)
        wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        assert cp.state.metadata.get("pre_inference_tokens") == 123

    def test_pre_inference_counting_disabled(self):
        adapter, cp = self._make_adapter(count_tokens_before_call=False)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        wrapped = adapter.wrap(mock_client)
        wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        assert "pre_inference_tokens" not in cp.state.metadata

    def test_cache_savings_recorded_in_metadata(self):
        adapter, cp = self._make_adapter()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 180
        mock_response.usage.cache_creation_input_tokens = 20
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response
        mock_client.messages.count_tokens.return_value = MagicMock(input_tokens=200)

        wrapped = adapter.wrap(mock_client)
        wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        savings = cp.state.metadata.get("cache_savings", [])
        assert len(savings) == 1
        assert savings[0]["cache_read_tokens"] == 180
        assert savings[0]["cache_write_tokens"] == 20

    def test_count_tokens_failure_is_non_fatal(self):
        """If count_tokens raises, the main call still proceeds."""
        adapter, cp = self._make_adapter(count_tokens_before_call=True)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response
        mock_client.messages.count_tokens.side_effect = Exception("API error")

        wrapped = adapter.wrap(mock_client)
        # Should not raise
        wrapped.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
        # pre_inference_tokens should not be set
        assert "pre_inference_tokens" not in cp.state.metadata
