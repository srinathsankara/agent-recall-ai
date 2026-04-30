"""
Tests for Phase 5 privacy components:
  - PIIRedactor (redaction accuracy, sensitivity levels, dry_run, custom rules)
  - VersionedSchema (migration path finding, forward/backward migration, edge cases)
"""
from __future__ import annotations

import json
import pytest

from agent_recall_ai.privacy import (
    PIIRedactor,
    RedactionRule,
    RedactionResult,
    SensitivityLevel,
    VersionedSchema,
    SchemaVersion,
    MigrationError,
)


# ===========================================================================
# PIIRedactor tests
# ===========================================================================

class TestPIIRedactorBasic:
    """Core redaction accuracy tests."""

    def test_openai_key_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text, cats = redactor.redact_text("My key is sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde")
        assert "sk-proj" not in text
        assert "[REDACTED:openai_key]" in text
        assert "openai_api_key" in cats

    def test_anthropic_key_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text, cats = redactor.redact_text("token: sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdef")
        assert "sk-ant" not in text
        assert "anthropic_api_key" in cats

    def test_aws_access_key_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text, cats = redactor.redact_text("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in text
        assert "aws_access_key" in cats

    def test_github_token_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        # GitHub PAT format: gh[pousr]_ followed by 36+ alphanumeric chars
        text, cats = redactor.redact_text("token = ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl")
        assert "ghp_" not in text
        assert "github_token" in cats

    def test_database_url_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text, cats = redactor.redact_text(
            "DATABASE_URL=postgres://myuser:mysecretpassword@localhost:5432/mydb"
        )
        assert "mysecretpassword" not in text
        assert "database_url" in cats

    def test_generic_password_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text, cats = redactor.redact_text('password = "SuperSecret123!"')
        assert "SuperSecret123" not in text
        assert "generic_password" in cats

    def test_private_key_header_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text, cats = redactor.redact_text("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...")
        assert "BEGIN RSA PRIVATE KEY" not in text
        assert "private_key_block" in cats

    def test_email_medium_sensitivity(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.MEDIUM)
        text, cats = redactor.redact_text("Contact: alice@example.com for help")
        assert "alice@example.com" not in text
        assert "email_address" in cats

    def test_email_not_redacted_at_low(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text, cats = redactor.redact_text("Contact: alice@example.com for help")
        assert "alice@example.com" in text
        assert "email_address" not in cats

    def test_phone_number_medium_sensitivity(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.MEDIUM)
        text, cats = redactor.redact_text("Call me at 555-867-5309")
        assert "867-5309" not in text
        assert "phone_number" in cats

    def test_ssn_high_sensitivity(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.HIGH)
        text, cats = redactor.redact_text("SSN: 123-45-6789")
        assert "123-45-6789" not in text
        assert "ssn" in cats

    def test_ssn_not_redacted_at_medium(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.MEDIUM)
        text, cats = redactor.redact_text("SSN: 123-45-6789")
        assert "ssn" not in cats

    def test_credit_card_high_sensitivity(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.HIGH)
        text, cats = redactor.redact_text("Card: 4111111111111111")
        assert "4111111111111111" not in text
        assert "credit_card" in cats

    def test_private_ip_high_sensitivity(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.HIGH)
        text, cats = redactor.redact_text("Server at 192.168.1.100")
        assert "192.168.1.100" not in text
        assert "private_ip_range" in cats

    def test_no_false_positive_on_clean_text(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.HIGH)
        clean = "The function returns True when the input is valid."
        text, cats = redactor.redact_text(clean)
        assert text == clean
        assert cats == []

    def test_multiple_secrets_in_one_string(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.MEDIUM)
        dirty = (
            "API_KEY=sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde "
            "email=bob@example.com "
            "password=Hunter2Password"
        )
        text, cats = redactor.redact_text(dirty)
        assert "sk-proj" not in text
        assert "bob@example.com" not in text
        assert "Hunter2Password" not in text
        assert len(cats) >= 2

    def test_clean_text_unchanged(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.HIGH)
        text = "Just a normal sentence about Python programming."
        result, cats = redactor.redact_text(text)
        assert result == text
        assert cats == []


class TestPIIRedactorDryRun:
    """dry_run=True detects but does NOT modify."""

    def test_dry_run_does_not_modify(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW, dry_run=True)
        original = "key = sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        text, cats = redactor.redact_text(original)
        assert text == original          # unchanged
        assert "openai_api_key" in cats  # still detected

    def test_dry_run_state_unchanged(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW, dry_run=True)
        state = {
            "session_id": "test",
            "context_summary": "API key: sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde",
            "goals": [],
        }
        import copy
        original_state = copy.deepcopy(state)
        result = redactor.redact_state(state, "test")
        assert state == original_state    # state dict is unchanged in dry_run
        assert result.total_redactions > 0  # but detections are reported


class TestPIIRedactorHashMode:
    """hash_redacted=True produces deterministic tokens."""

    def test_hash_placeholder_format(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW, hash_redacted=True)
        text, _ = redactor.redact_text("sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde")
        # Should contain a hash suffix like [REDACTED:openai_key:abcd1234]
        assert "[REDACTED:openai_key:" in text

    def test_same_value_same_hash(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW, hash_redacted=True)
        key = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde"
        text1, _ = redactor.redact_text(key)
        text2, _ = redactor.redact_text(key)
        assert text1 == text2


class TestPIIRedactorStateWalk:
    """redact_state() walks nested dicts/lists."""

    def _build_state(self) -> dict:
        return {
            "session_id": "test-session",
            "context_summary": "Refactoring auth",
            "goals": ["Replace python-jose with PyJWT"],
            "decisions": [
                {
                    "id": "abc",
                    "summary": "Use PyJWT",
                    "reasoning": "API key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde leaked in output",
                    "tags": [],
                }
            ],
            "metadata": {"db": "postgres://admin:SECRETPASS@db.internal:5432/prod"},
            "alerts": [],
            "next_steps": [],
            "token_usage": {"prompt": 1000, "completion": 500, "cached": 0},
            "cost_usd": 0.001,
            "context_utilization": 0.1,
        }

    def test_nested_decision_reasoning_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        state = self._build_state()
        result = redactor.redact_state(state, "test-session")
        assert "sk-proj" not in state["decisions"][0]["reasoning"]
        assert result.total_redactions >= 1

    def test_metadata_db_url_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        state = self._build_state()
        redactor.redact_state(state, "test-session")
        assert "SECRETPASS" not in state["metadata"]["db"]

    def test_list_items_redacted(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.MEDIUM)
        state = {
            "goals": [
                "Contact alice@example.com about the deployment",
                "Normal goal with no PII",
            ]
        }
        redactor.redact_state(state, "test")
        assert "alice@example.com" not in state["goals"][0]

    def test_numeric_fields_untouched(self):
        """Numbers, booleans, None pass through without error."""
        redactor = PIIRedactor(sensitivity=SensitivityLevel.HIGH)
        state = {
            "cost_usd": 1.23,
            "context_utilization": 0.45,
            "checkpoint_seq": 7,
            "active": True,
            "nullable": None,
        }
        import copy
        original = copy.deepcopy(state)
        redactor.redact_state(state, "test")
        assert state["cost_usd"] == original["cost_usd"]
        assert state["context_utilization"] == original["context_utilization"]
        assert state["checkpoint_seq"] == original["checkpoint_seq"]
        assert state["active"] == original["active"]
        assert state["nullable"] == original["nullable"]

    def test_redaction_result_fields_tracked(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        state = {
            "decisions": [{"reasoning": "token sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabc"}],
            "metadata": {"key": "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZdef"},
        }
        result = redactor.redact_state(state, "sid")
        assert result.total_redactions >= 2
        assert result.was_redacted is True
        assert len(result.fields_touched) >= 1

    def test_scan_only_no_modification(self):
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        text = "key=sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabc password=Hunter2!"
        cats = redactor.scan_only(text)
        assert "openai_api_key" in cats
        assert "generic_password" in cats


class TestPIIRedactorCustomRules:
    """Custom rules can be added at init or runtime."""

    def test_custom_rule_at_init(self):
        import re
        custom = RedactionRule(
            name="internal_token",
            pattern=re.compile(r"CORP-[A-Z0-9]{16}"),
            placeholder="[REDACTED:corp_token]",
            severity=SensitivityLevel.LOW,
        )
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW, custom_rules=[custom])
        text, cats = redactor.redact_text("auth: CORP-ABCD1234EFGH5678")
        assert "CORP-ABCD1234EFGH5678" not in text
        assert "internal_token" in cats

    def test_add_rule_runtime(self):
        import re
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)
        redactor.add_rule(RedactionRule(
            name="employee_id",
            pattern=re.compile(r"EMP-\d{6}"),
            placeholder="[REDACTED:employee_id]",
            severity=SensitivityLevel.LOW,
        ))
        text, cats = redactor.redact_text("Employee EMP-123456 submitted the request")
        assert "EMP-123456" not in text
        assert "employee_id" in cats

    def test_custom_rule_not_triggered_on_non_match(self):
        import re
        custom = RedactionRule(
            name="secret_code",
            pattern=re.compile(r"SECRET-\d{4}"),
            placeholder="[REDACTED:secret_code]",
            severity=SensitivityLevel.LOW,
        )
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW, custom_rules=[custom])
        text, cats = redactor.redact_text("Nothing secret here")
        assert "secret_code" not in cats


class TestRedactionResult:
    """RedactionResult restore() method."""

    def test_restore_redacted_value(self):
        result = RedactionResult(session_id="test")
        result.add("openai_api_key", "decisions[0].reasoning", "sk-real-key", "[REDACTED:openai_key]")
        restored = result.restore("[REDACTED:openai_key] was used here")
        assert "sk-real-key" in restored

    def test_was_redacted_false_when_empty(self):
        result = RedactionResult(session_id="test")
        assert result.was_redacted is False

    def test_by_category_counts(self):
        result = RedactionResult(session_id="test")
        result.add("openai_api_key", "field1", "val1", "[REDACTED:openai_key]")
        result.add("openai_api_key", "field2", "val2", "[REDACTED:openai_key]")
        result.add("email_address", "field3", "val3", "[REDACTED:email]")
        assert result.by_category["openai_api_key"] == 2
        assert result.by_category["email_address"] == 1
        assert result.total_redactions == 3


# ===========================================================================
# VersionedSchema tests
# ===========================================================================

class TestVersionedSchemaBasic:
    """Stamp, version detection, basic migration."""

    def test_stamp_adds_version(self):
        vs = VersionedSchema()
        d = {"session_id": "test"}
        stamped = vs.stamp(d)
        assert stamped["schema_version"] == "1.0.0"
        assert stamped is d  # mutates in-place

    def test_get_version_present(self):
        vs = VersionedSchema()
        assert vs.get_version({"schema_version": "1.0.0"}) == "1.0.0"

    def test_get_version_missing_returns_zero(self):
        vs = VersionedSchema()
        assert vs.get_version({}) == "0.0.0"

    def test_needs_migration_outdated(self):
        vs = VersionedSchema()
        assert vs.needs_migration({"schema_version": "0.0.0"}) is True

    def test_needs_migration_current(self):
        vs = VersionedSchema()
        assert vs.needs_migration({"schema_version": "1.0.0"}) is False

    def test_current_version_is_1_0_0(self):
        vs = VersionedSchema()
        assert vs.current_version == "1.0.0"


class TestVersionedSchemaMigrateForward:
    """Forward migration: 0.0.0 → 1.0.0."""

    def _legacy_state(self) -> dict:
        return {
            "session_id": "old-session",
            "task_description": "Refactor auth",   # old field name
            "goals": ["Goal 1"],
            "constraints": [],
            "decisions": [
                {"summary": "Use PyJWT", "reasoning": "Better maintained"},
            ],
            "files_modified": [],
            "next_steps": [],
            "status": "active",
            "cost_usd": 0.0,
            "checkpoint_seq": 0,
            "prompt_tokens": 1000,       # old flat token fields
            "completion_tokens": 500,
            "cached_tokens": 100,
        }

    def test_migrate_renames_task_description(self):
        vs = VersionedSchema()
        state = self._legacy_state()
        migrated = vs.migrate_to_current(state)
        assert "context_summary" in migrated
        assert migrated["context_summary"] == "Refactor auth"
        assert "task_description" not in migrated

    def test_migrate_adds_alerts(self):
        vs = VersionedSchema()
        state = self._legacy_state()
        migrated = vs.migrate_to_current(state)
        assert "alerts" in migrated
        assert isinstance(migrated["alerts"], list)

    def test_migrate_adds_context_utilization(self):
        vs = VersionedSchema()
        state = self._legacy_state()
        migrated = vs.migrate_to_current(state)
        assert "context_utilization" in migrated
        assert migrated["context_utilization"] == 0.0

    def test_migrate_normalizes_token_usage(self):
        vs = VersionedSchema()
        state = self._legacy_state()
        migrated = vs.migrate_to_current(state)
        assert "token_usage" in migrated
        assert migrated["token_usage"]["prompt"] == 1000
        assert migrated["token_usage"]["completion"] == 500
        assert migrated["token_usage"]["cached"] == 100
        assert "prompt_tokens" not in migrated
        assert "completion_tokens" not in migrated

    def test_migrate_normalizes_decision_ids(self):
        vs = VersionedSchema()
        state = self._legacy_state()
        migrated = vs.migrate_to_current(state)
        decision = migrated["decisions"][0]
        assert "id" in decision
        assert "tags" in decision
        assert isinstance(decision["tags"], list)
        assert "alternatives_rejected" in decision

    def test_migrate_adds_metadata(self):
        vs = VersionedSchema()
        state = self._legacy_state()
        migrated = vs.migrate_to_current(state)
        assert "metadata" in migrated
        assert isinstance(migrated["metadata"], dict)

    def test_migrate_stamps_version(self):
        vs = VersionedSchema()
        state = self._legacy_state()
        migrated = vs.migrate_to_current(state)
        assert migrated["schema_version"] == "1.0.0"

    def test_migrate_idempotent_on_current(self):
        vs = VersionedSchema()
        state = {"schema_version": "1.0.0", "session_id": "x", "goals": []}
        result = vs.migrate_to_current(state)
        assert result is state  # same object, no-op

    def test_migrate_no_task_description_uses_empty(self):
        """If neither task_description nor context_summary exists, default to ''."""
        vs = VersionedSchema()
        state = {"session_id": "x", "goals": []}
        migrated = vs.migrate_to_current(state)
        assert migrated.get("context_summary") == ""


class TestVersionedSchemaMigrateBackward:
    """Backward migration: 1.0.0 → 0.0.0."""

    def _current_state(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "session_id": "current",
            "context_summary": "Refactor auth",
            "goals": [],
            "alerts": [],
            "metadata": {"model": "gpt-4o"},
            "context_utilization": 0.5,
            "open_questions": [],
            "tool_calls": [],
        }

    def test_backward_strips_alerts(self):
        vs = VersionedSchema()
        state = self._current_state()
        downgraded = vs.migrate_to(state, "0.0.0")
        assert "alerts" not in downgraded

    def test_backward_strips_metadata(self):
        vs = VersionedSchema()
        state = self._current_state()
        downgraded = vs.migrate_to(state, "0.0.0")
        assert "metadata" not in downgraded

    def test_backward_renames_context_summary(self):
        vs = VersionedSchema()
        state = self._current_state()
        downgraded = vs.migrate_to(state, "0.0.0")
        assert "task_description" in downgraded
        assert downgraded["task_description"] == "Refactor auth"
        assert "context_summary" not in downgraded

    def test_backward_stamps_old_version(self):
        vs = VersionedSchema()
        state = self._current_state()
        downgraded = vs.migrate_to(state, "0.0.0")
        assert downgraded["schema_version"] == "0.0.0"

    def test_backward_noop_if_same_version(self):
        vs = VersionedSchema()
        state = {"schema_version": "0.0.0", "session_id": "x"}
        result = vs.migrate_to(state, "0.0.0")
        assert result is state


class TestVersionedSchemaPathFinding:
    """BFS migration path finding."""

    def test_forward_path_found(self):
        vs = VersionedSchema()
        path = vs._find_migration_path("0.0.0", "1.0.0", forward=True)
        assert len(path) == 1
        assert path[0].from_version == "0.0.0"
        assert path[0].to_version == "1.0.0"

    def test_backward_path_found(self):
        vs = VersionedSchema()
        path = vs._find_migration_path("1.0.0", "0.0.0", forward=False)
        assert len(path) == 1
        assert path[0].from_version == "0.0.0"
        assert path[0].to_version == "1.0.0"

    def test_same_version_returns_empty_path(self):
        vs = VersionedSchema()
        path = vs._find_migration_path("1.0.0", "1.0.0")
        assert path == []

    def test_nonexistent_path_returns_empty_list(self):
        vs = VersionedSchema()
        path = vs._find_migration_path("9.9.9", "8.8.8")
        assert path == []

    def test_multi_hop_forward_path(self):
        """A → B → C migration chain is found correctly."""
        vs = VersionedSchema(current_version="3.0.0")
        vs.register(SchemaVersion(
            from_version="1.0.0",
            to_version="2.0.0",
            migrate_forward=lambda d: {**d, "v2_field": True},
            description="1→2",
        ))
        vs.register(SchemaVersion(
            from_version="2.0.0",
            to_version="3.0.0",
            migrate_forward=lambda d: {**d, "v3_field": True},
            description="2→3",
        ))
        path = vs._find_migration_path("1.0.0", "3.0.0")
        assert len(path) == 2
        assert path[0].from_version == "1.0.0"
        assert path[1].to_version == "3.0.0"


class TestVersionedSchemaMissingPath:
    """Handling of missing migration paths."""

    def test_no_path_non_strict_logs_warning(self, caplog):
        import logging
        vs = VersionedSchema(strict=False)
        state = {"schema_version": "99.0.0", "session_id": "x"}
        with caplog.at_level(logging.WARNING, logger="agent_recall_ai.privacy.versioned_schema"):
            result = vs.migrate_to_current(state)
        assert "No migration path" in caplog.text
        # Returns the dict anyway, bumps to current version
        assert result["schema_version"] == vs.current_version

    def test_no_path_strict_raises(self):
        vs = VersionedSchema(strict=True)
        state = {"schema_version": "99.0.0", "session_id": "x"}
        with pytest.raises(MigrationError, match="No migration path"):
            vs.migrate_to_current(state)

    def test_no_backward_path_non_strict_returns_dict(self):
        vs = VersionedSchema(strict=False)
        state = {"schema_version": "1.0.0", "session_id": "x"}
        result = vs.migrate_to(state, "5.0.0")   # non-existent target
        assert result is state   # returned unchanged


class TestVersionedSchemaListMigrations:
    """list_migrations() returns correct metadata."""

    def test_lists_builtin_migration(self):
        vs = VersionedSchema()
        migrations = vs.list_migrations()
        assert len(migrations) >= 1
        m = migrations[0]
        assert m["from"] == "0.0.0"
        assert m["to"] == "1.0.0"
        assert m["has_backward"] is True

    def test_lists_custom_migration(self):
        vs = VersionedSchema(current_version="2.0.0")
        vs.register(SchemaVersion(
            from_version="1.0.0",
            to_version="2.0.0",
            migrate_forward=lambda d: d,
            description="Add v2 fields",
        ))
        migrations = vs.list_migrations()
        custom = next((m for m in migrations if m["to"] == "2.0.0"), None)
        assert custom is not None
        assert custom["has_backward"] is False


class TestVersionedSchemaMigrationError:
    """MigrationError is raised when a migration function fails."""

    def test_migration_fn_exception_raises_migration_error(self):
        vs = VersionedSchema(current_version="2.0.0")

        def bad_migrate(d: dict) -> dict:
            raise ValueError("Unexpected field layout")

        vs.register(SchemaVersion(
            from_version="1.0.0",
            to_version="2.0.0",
            migrate_forward=bad_migrate,
            description="Broken migration",
        ))
        state = {"schema_version": "1.0.0", "session_id": "x"}
        with pytest.raises(MigrationError, match="1.0.0 → 2.0.0 failed"):
            vs.migrate_to_current(state)


# ===========================================================================
# Integration: PIIRedactor + Checkpoint.save()
# ===========================================================================

class TestCheckpointWithRedactor:
    """Verify that Checkpoint.save() applies redaction before persisting."""

    def test_secrets_not_persisted(self):
        from agent_recall_ai import Checkpoint
        from agent_recall_ai.storage.memory import MemoryStore

        store = MemoryStore()
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)

        with Checkpoint("redact-test", store=store, redactor=redactor) as cp:
            cp.set_goal("Deploy service")
            cp.record_decision(
                "Use PyJWT",
                reasoning="Old key was sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde",
            )

        saved = store.load("redact-test")
        assert saved is not None
        # The saved decision reasoning must not contain the raw API key
        assert "sk-proj" not in saved.decisions[0].reasoning
        assert "[REDACTED" in saved.decisions[0].reasoning

    def test_in_memory_state_retains_original(self):
        """The live Checkpoint._state should keep the original (unredacted) values."""
        from agent_recall_ai import Checkpoint
        from agent_recall_ai.storage.memory import MemoryStore

        store = MemoryStore()
        redactor = PIIRedactor(sensitivity=SensitivityLevel.LOW)

        cp = Checkpoint("live-test", store=store, redactor=redactor)
        cp.set_goal("Test goal")
        cp.record_decision(
            "Use PyJWT",
            reasoning="key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcde is the real one",
        )
        cp.save()

        # In-memory state is NOT redacted
        live_reasoning = cp.state.decisions[0].reasoning
        assert "sk-proj" in live_reasoning


class TestCheckpointWithVersionedSchema:
    """Verify that Checkpoint.save() stamps schema_version."""

    def test_schema_version_stamped_on_save(self):
        import json
        from agent_recall_ai import Checkpoint
        from agent_recall_ai.storage.memory import MemoryStore
        from agent_recall_ai.privacy import VersionedSchema

        store = MemoryStore()
        schema = VersionedSchema()

        with Checkpoint("schema-test", store=store, schema=schema) as cp:
            cp.set_goal("Test schema stamping")

        saved = store.load("schema-test")
        assert saved is not None
        # schema_version should be stamped — read directly from stored JSON
        stored_json = json.loads(saved.model_dump_json())
        assert stored_json.get("schema_version") == "1.0.0"
