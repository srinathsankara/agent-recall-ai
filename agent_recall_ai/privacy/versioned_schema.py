"""
VersionedSchema — forward/backward compatible schema migrations.

Guarantees that a checkpoint created today works with the library
6 months (or 2 major versions) from now, and vice versa.

Design:
  - Every checkpoint carries a `schema_version` field (semver string)
  - The library knows the CURRENT_VERSION it produces
  - A registry of migration functions handles version gaps
  - Loading an old checkpoint: apply forward migrations in sequence
  - Producing a checkpoint for an old consumer: apply backward migrations

Migration functions are pure functions: (dict) -> dict
They must be registered in strict semver order.

Usage:
    from agent_recall_ai.privacy import VersionedSchema

    # Attach to Checkpoint — auto-migrates on load
    with Checkpoint("my-task", schema=VersionedSchema()) as cp:
        ...

    # Manually check/migrate a raw dict
    vs = VersionedSchema()
    migrated = vs.migrate_to_current(raw_dict)
    print(vs.current_version)   # "1.2.0"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# The canonical version this build of the library produces.
# Bump when adding new fields to TaskState.
CURRENT_VERSION = "1.0.0"

MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


class MigrationError(Exception):
    """Raised when a migration cannot be applied cleanly."""


@dataclass
class SchemaVersion:
    """A registered migration step."""
    from_version: str        # e.g. "0.9.0"
    to_version: str          # e.g. "1.0.0"
    migrate_forward: MigrationFn
    migrate_backward: Optional[MigrationFn] = None
    description: str = ""


from typing import Optional


class VersionedSchema:
    """
    Manages schema versioning and migration for checkpoint data.

    Args:
        current_version: Override the library's current version (for testing).
        strict: If True, raise MigrationError when a migration path is missing.
                If False, log a warning and return the dict as-is.
    """

    def __init__(
        self,
        current_version: str = CURRENT_VERSION,
        strict: bool = False,
    ) -> None:
        self.current_version = current_version
        self.strict = strict
        self._migrations: list[SchemaVersion] = []
        self._register_builtin_migrations()

    # ── Public API ───────────────────────────────────────────────────────────

    def register(self, migration: SchemaVersion) -> None:
        """Register a migration step. Must be added in semver order."""
        self._migrations.append(migration)
        # Keep sorted by from_version for deterministic application
        self._migrations.sort(key=lambda m: _parse_version(m.from_version))

    def stamp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Add the current schema_version to a state dict."""
        state_dict["schema_version"] = self.current_version
        return state_dict

    def get_version(self, state_dict: dict[str, Any]) -> str:
        """Extract the schema version from a state dict. Returns '0.0.0' if missing."""
        return state_dict.get("schema_version", "0.0.0")

    def needs_migration(self, state_dict: dict[str, Any]) -> bool:
        """Return True if the state dict is not at the current version."""
        return self.get_version(state_dict) != self.current_version

    def migrate_to_current(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Apply forward migrations until state_dict reaches the current version.

        Idempotent — safe to call on an already-current dict.
        """
        source_version = self.get_version(state_dict)

        if source_version == self.current_version:
            return state_dict

        path = self._find_migration_path(source_version, self.current_version)

        if not path:
            msg = (
                f"No migration path from schema version '{source_version}' "
                f"to '{self.current_version}'. "
                "The checkpoint may still load if new fields have default values."
            )
            if self.strict:
                raise MigrationError(msg)
            logger.warning("VersionedSchema: %s", msg)
            state_dict["schema_version"] = self.current_version
            return state_dict

        for step in path:
            try:
                state_dict = step.migrate_forward(state_dict)
                state_dict["schema_version"] = step.to_version
                logger.debug(
                    "VersionedSchema: migrated %s → %s: %s",
                    step.from_version, step.to_version, step.description,
                )
            except Exception as exc:
                raise MigrationError(
                    f"Migration {step.from_version} → {step.to_version} failed: {exc}"
                ) from exc

        return state_dict

    def migrate_to(
        self, state_dict: dict[str, Any], target_version: str
    ) -> dict[str, Any]:
        """
        Migrate backward to an older version (for compatibility with old consumers).
        """
        source_version = self.get_version(state_dict)
        if source_version == target_version:
            return state_dict

        # Attempt backward migrations
        path = self._find_migration_path(source_version, target_version, forward=False)

        if not path:
            msg = f"No backward migration path from '{source_version}' to '{target_version}'"
            if self.strict:
                raise MigrationError(msg)
            logger.warning("VersionedSchema: %s", msg)
            return state_dict

        for step in path:
            if step.migrate_backward is None:
                raise MigrationError(
                    f"Migration {step.from_version} → {step.to_version} "
                    "has no backward migration defined."
                )
            state_dict = step.migrate_backward(state_dict)
            state_dict["schema_version"] = step.from_version

        return state_dict

    def list_migrations(self) -> list[dict]:
        """Return registered migrations as a list of dicts."""
        return [
            {
                "from": m.from_version,
                "to": m.to_version,
                "description": m.description,
                "has_backward": m.migrate_backward is not None,
            }
            for m in self._migrations
        ]

    # ── Built-in migrations (library history) ────────────────────────────────

    def _register_builtin_migrations(self) -> None:
        """
        Every schema change in the library's history is registered here.
        This is the authoritative migration chain.
        """

        # 0.0.0 → 1.0.0: Initial stable release
        # Added: schema_version, context_utilization, alerts, metadata
        self.register(SchemaVersion(
            from_version="0.0.0",
            to_version="1.0.0",
            description="Initial stable schema — add schema_version, alerts, context_utilization",
            migrate_forward=_migrate_0_0_0_to_1_0_0,
            migrate_backward=_migrate_1_0_0_to_0_0_0,
        ))

        # Future migrations registered here, e.g.:
        # self.register(SchemaVersion(
        #     from_version="1.0.0",
        #     to_version="1.1.0",
        #     description="Add tool_call_graph for multi-agent tracing",
        #     migrate_forward=_migrate_1_0_0_to_1_1_0,
        # ))

    # ── Private helpers ──────────────────────────────────────────────────────

    def _find_migration_path(
        self, source: str, target: str, forward: bool = True
    ) -> list[SchemaVersion]:
        """BFS over the migration graph to find the shortest path."""
        if source == target:
            return []

        from collections import deque
        graph: dict[str, list[SchemaVersion]] = {}

        for m in self._migrations:
            if forward:
                graph.setdefault(m.from_version, []).append(m)
            else:
                # Reverse graph for backward migrations
                graph.setdefault(m.to_version, []).append(m)

        queue: deque[tuple[str, list[SchemaVersion]]] = deque([(source, [])])
        visited: set[str] = {source}

        while queue:
            current, path = queue.popleft()
            for step in graph.get(current, []):
                next_ver = step.to_version if forward else step.from_version
                new_path = path + [step]
                if next_ver == target:
                    return new_path
                if next_ver not in visited:
                    visited.add(next_ver)
                    queue.append((next_ver, new_path))

        return []


# ── Migration functions ───────────────────────────────────────────────────────

def _migrate_0_0_0_to_1_0_0(state: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade a pre-1.0.0 checkpoint to the stable 1.0.0 schema.

    Changes:
    - Add schema_version field
    - Rename 'task_description' to 'context_summary' (old field name)
    - Add context_utilization default (0.0)
    - Add alerts list if missing
    - Add metadata dict if missing
    - Normalise decision structure: add 'id', 'tags' if missing
    """
    import uuid
    out = dict(state)

    # Rename legacy field
    if "task_description" in out and "context_summary" not in out:
        out["context_summary"] = out.pop("task_description")
    out.setdefault("context_summary", "")

    # Add missing fields with safe defaults
    out.setdefault("context_utilization", 0.0)
    out.setdefault("alerts", [])
    out.setdefault("metadata", {})
    out.setdefault("open_questions", [])
    out.setdefault("tool_calls", [])
    out.setdefault("checkpoint_seq", 0)

    # Normalise token_usage
    if "token_usage" not in out:
        # Legacy: flat token fields
        out["token_usage"] = {
            "prompt": out.pop("prompt_tokens", 0),
            "completion": out.pop("completion_tokens", 0),
            "cached": out.pop("cached_tokens", 0),
        }

    # Normalise decisions
    decisions = out.get("decisions", [])
    for d in decisions:
        d.setdefault("id", uuid.uuid4().hex[:8])
        d.setdefault("tags", [])
        d.setdefault("alternatives_rejected", [])
        d.setdefault("reasoning", "")

    return out


def _migrate_1_0_0_to_0_0_0(state: dict[str, Any]) -> dict[str, Any]:
    """Backward migration: strip fields added in 1.0.0."""
    out = dict(state)
    # Restore old field name
    if "context_summary" in out:
        out["task_description"] = out.pop("context_summary")
    for key in ["context_utilization", "alerts", "metadata", "open_questions",
                "tool_calls", "schema_version"]:
        out.pop(key, None)
    return out


def _parse_version(v: str) -> tuple[int, int, int]:
    """Parse '1.2.3' → (1, 2, 3) for sorting."""
    try:
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (0, 0, 0)
