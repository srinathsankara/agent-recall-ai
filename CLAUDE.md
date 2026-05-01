# CLAUDE.md — agent-recall-ai Project Guide

This file ensures any future Claude Code session can resume building this project
without losing context. It is the project's own checkpoint.

---

## Project Summary

**agent-recall-ai** is a framework-agnostic Python library that solves AI agent session death.

**Core problem:** When an LLM context window fills during a long task, all session state is lost.
There is no recovery point. Every framework has this open issue.

**Our solution:** Structured checkpoints that capture: goals, decisions (with reasoning),
files touched, next steps, token usage, and real-time monitor alerts.

---

## Architecture Decisions Made

### Decision 1 — Pydantic v2 for TaskState
**Why:** Type safety, fast JSON serialization via `model_dump_json()`, automatic validation.
**Rejected:** dataclasses (no validation), attrs (less ecosystem support).

### Decision 2 — ContextVar pattern NOT used (unlike agenttest)
**Why:** agent-recall-ai is explicit, not magic. Users call `cp.record_decision()` directly.
**Rejected:** Thread-local/ContextVar injection (too magical, breaks in subprocesses).

### Decision 3 — SQLite as default, Redis as production option
**Why:** SQLite = zero config for dev. Redis = fast hydration for distributed agents.
**Rejected:** PostgreSQL (overkill for single-machine), raw files (no querying).

### Decision 4 — Keyword fallback for SemanticPruner
**Why:** sentence-transformers is optional (300MB download). Must work without it.
**Rejected:** Always requiring embeddings (breaks CI, adds cold start latency).

### Decision 5 — Plugin adapter registry (`@register_adapter`)
**Why:** Clean extensibility. Users can add custom frameworks without forking.
**Rejected:** Inheritance-only pattern (requires subclassing base, harder to discover).

### Decision 6 — Decision Anchors are NEVER pruned
**Why:** The reasoning chain is the irreplaceable part of a session. Token counts can be
reduced, but the "why" must survive compression.
**Keywords:** decided, rejected, architecture, because, must not, never, constraint, chosen,
alternative, trade-off, required, rationale, agreed, principle, avoid, forbidden.

### Decision 7 — Protobuf schema in `proto/schema.proto`
**Why:** Provides a language-neutral contract for multi-agent handoffs and
gRPC-based checkpoint services. Not required for basic use.
**Rejected:** Avro (less Python support), JSON Schema (verbose, no code gen).

---

## Current File Structure

```
agent_recall_ai/
├── __init__.py              Public API exports
├── checkpoint.py            Checkpoint class (primary user-facing API)
├── core/
│   ├── state.py             TaskState (Pydantic), Decision, FileChange, Alert, TokenUsage
│   ├── tracker.py           TokenCostTracker, _MODEL_COSTS pricing table
│   ├── compressor.py        compress_tool_output, compress_decision_log, build_resume_context
│   └── semantic_pruner.py   SemanticPruner, _is_decision_anchor, ScoredMessage
├── storage/
│   ├── disk.py              DiskStore (simple SQLite)
│   └── memory.py            MemoryStore (in-memory, tests only)
├── persistence/
│   ├── sqlite_provider.py   SQLiteProvider (production SQLite with decision_log table)
│   └── redis_provider.py    RedisProvider (Redis with TTL, pub/sub, sorted index)
├── monitors/
│   ├── base.py              BaseMonitor ABC, event hooks
│   ├── cost_monitor.py      CostMonitor, CostBudgetExceeded exception
│   ├── token_monitor.py     TokenMonitor (context pressure alerts)
│   ├── drift_monitor.py     DriftMonitor (constraint violation detection)
│   ├── package_monitor.py   PackageHallucinationMonitor
│   └── tool_bloat_monitor.py ToolBloatMonitor (auto-compression)
├── adapters/
│   ├── base.py              BaseAdapter, register_adapter(), get_adapter() registry
│   ├── anthropic_adapter.py AnthropicAdapter (prompt caching + pre-inference token count)
│   ├── openai_adapter.py    OpenAIAdapter (ConversationRepair + tool recording)
│   ├── langchain_adapter.py LangChainAdapter (CallbackHandler + MessageHistory)
│   ├── langgraph_adapter.py LangGraphAdapter (BaseCheckpointSaver drop-in + thread forking)
│   ├── crewai_adapter.py    CrewAIAdapter (kickoff + task instrumentation)
│   └── smolagents_adapter.py smolagentsAdapter (run + step + log harvesting)
├── exporters/
│   ├── otlp.py              OTLPExporter (OpenTelemetry spans → Datadog/Jaeger/Grafana)
│   └── datadog.py           DatadogExporter (convenience wrapper for Datadog APM)
└── privacy/
    ├── redactor.py          PIIRedactor (14 regex categories, reversible, pre-serialization)
    ├── versioned_schema.py  VersionedSchema (BFS migration graph, stamp + migrate)
    └── presidio_backend.py  PresidioBackend (NER-based PII via Microsoft Presidio, optional)

cli/main.py              Typer CLI (list, inspect, resume, export, delete, status, install-hooks)
proto/schema.proto       Universal AgentState protobuf schema
tests/                   235 passing tests (17 skipped — optional deps)
examples/                Working examples (OpenAI, Anthropic, LangChain, LangGraph)
scripts/benchmark.py     Performance comparison: agent-recall-ai vs summarization vs truncation
```

---

## Architecture Decisions Made

### Decision 8 — Anthropic prompt caching ON by default
**Why:** The spec promises 90% cost/latency reduction. Zero user friction — breakpoints are
injected transparently on system message, tools, and last user message. Disabled via
`AnthropicAdapter(cp, enable_prompt_caching=False)`.
**Rejected:** Opt-in (too many users miss it), per-call (inconsistent cache warming).

### Decision 9 — LangGraphAdapter uses BaseCheckpointSaver
**Why:** Drop-in compatibility means 47K+ LangGraph users can switch with one line change.
`BaseCheckpointSaver` is the stable public interface in LangGraph >= 0.2.
**Rejected:** Custom interface (requires user code changes), LangGraph internals (unstable).

### Decision 10 — Thread forking via JSON round-trip
**Why:** Pydantic's `model_dump_json()` + `model_validate()` is the safest deep copy for
Pydantic v2 models. Python's `copy.deepcopy()` can fail on datetime/enum fields.
**Rejected:** `copy.deepcopy()` (brittle with complex nested models).

### Decision 11 — OTLP export over custom metrics
**Why:** Every observability platform (Datadog, Grafana, Honeycomb, Jaeger) ingests OTLP.
One implementation covers all backends.
**Rejected:** Datadog-only SDK (vendor lock-in), Prometheus (pull-based, wrong model for spans).

### Decision 12 — Presidio as optional backend (not default)
**Why:** Presidio requires spaCy models (200MB+ download). Can't be a required dep.
Current regex covers 14 categories of secrets — enough for the common case.
**Rejected:** Default Presidio (too heavy), Presidio-only (breaks offline/CI environments).

---

## Test Status

```
354 passed, 30 skipped (optional deps not installed), 0 failed
```

Run: `pytest tests/ -v`

Test files:
- `tests/test_state.py`           — TaskState models, TokenUsage
- `tests/test_tracker.py`         — TokenCostTracker, cost calculations
- `tests/test_semantic_pruner.py` — Decision anchor detection, keyword scoring, compression
- `tests/test_persistence.py`     — SQLiteProvider, MemoryStore CRUD
- `tests/test_monitors.py`        — All 5 monitors, edge cases
- `tests/test_checkpoint.py`      — Checkpoint context manager, resume, monitors
- `tests/test_adapters.py`        — Adapter registry, AnthropicAdapter, OpenAIAdapter
- `tests/test_compressor.py`      — Tool output compression, decision log compression
- `tests/test_privacy.py`         — PIIRedactor (14 categories), VersionedSchema, migrations
- `tests/test_new_features.py`    — Prompt caching, thread forking, LangGraph adapter, OTLP
- `tests/test_cli.py`             — CLI commands E2E: list, inspect, resume, export, delete, status, auto-save, install-hooks
- `tests/test_e2e.py`             — Production-grade E2E scenarios: long sessions, forking, PII, concurrency, decorator, schema versioning
- `tests/test_regression.py`     — One test class per bug fixed (prevents regressions across all 11 API fixes)
- `tests/test_application.py`    — End-to-end workflow scenarios: full task lifecycle, adapters, monitors, privacy, fork isolation, async, persistence, export, compression, decorator

---

## Known Issues / Next Work Items

1. **LangGraph list()** — Full checkpoint history listing requires store-level prefix scan.
   Current implementation returns only the latest checkpoint.  Fix: add `list_prefix()` to
   DiskStore/RedisProvider and use it in `LangGraphAdapter.list()`.
2. **gRPC service** — `proto/schema.proto` defines the service but no Python implementation exists.
3. **Dashboard (AgentPrism)** — Spec calls for Timeline + Sequence Diagram React views.
   Current `dashboard/index.html` is HTML-only.  Next step: port to React + D3.
4. **MCP server** — Model Context Protocol support (spec Q2 2026).
5. **Model cost table** — `core/tracker.py` has `_MODEL_COSTS` — update quarterly as pricing changes.
6. **ASAP pruning** — First-token surprisal I(x) = -log P(x) scoring for SemanticPruner.
   More principled than keyword matching but requires logprobs from the LLM API.

---

## Open Alternatives Considered and Rejected

| Alternative | Why Rejected |
|---|---|
| LangChain Memory | Framework-locked, no cost tracking, no decision anchors |
| Semantic Kernel | C#-first, Python SDK is secondary |
| Zep | External SaaS dependency, no self-hosted OSS option |
| Mem0 | Focuses on facts/memories, not reasoning/decision chains |
| LangGraph persistence | LangGraph-only, not framework-agnostic |

---

## Environment Setup

```bash
git clone https://github.com/srinathsankara/agent-recall-ai
cd agent-recall-ai
pip install -e ".[dev]"

# With all optional deps:
pip install -e ".[all]"

# Run tests:
pytest tests/ -v

# Lint:
ruff check agent_recall_ai/

# Type check:
mypy agent_recall_ai/
```

---

## Companion Project

**agenttest** (`../agenttest/`) — behavioral consistency testing for agents.
Checkpoints from this project export directly as agenttest fixtures:

```bash
agent-recall-ai export my-session --format agenttest > test_session.py
agenttest run ./test_session.py
```

---

*This file is the project's own checkpoint. Update it when adding new modules,
making architectural decisions, or resolving open issues.*
