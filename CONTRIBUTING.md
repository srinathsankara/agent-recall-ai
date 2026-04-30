# Contributing to agent-recall-ai

Thanks for helping solve AI agent session death. Every contribution matters.

---

## The fastest path to a merged PR

1. **Check open issues first** — someone may be working on the same thing.
2. **For new features**, open an issue with `[RFC]` in the title to discuss before writing code.
3. **For bug fixes**, just open a PR — no prior issue needed for obvious bugs.
4. **Tests are required.** Every new feature needs tests. Every bug fix needs a regression test.

---

## Setup

```bash
git clone https://github.com/srinathsankara/agent-recall-ai
cd agent-recall-ai
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check agent_recall_ai/

# Type check
mypy agent_recall_ai/
```

---

## What we want

### High-impact contributions

- **New framework adapters** — PydanticAI, AutoGen, Dify, n8n, LlamaIndex, Haystack, Agno
- **Async storage backends** — async DiskStore, async RedisProvider
- **New monitors** — latency monitor, hallucination rate monitor, tool retry counter
- **Export formats** — OTLP trace export, Datadog integration, OpenTelemetry spans
- **Migration scripts** — schema migrations as the library evolves

### Good first issues

- Add a new PII detection rule to `PIIRedactor`
- Add a new model to `_MODEL_COSTS` in `core/tracker.py`
- Improve the `dashboard/index.html` visualization
- Add a usage example to `examples/`
- Fix a typo or improve docs

---

## Adding a new framework adapter

1. Create `agent_recall_ai/adapters/<framework>_adapter.py`
2. Subclass `BaseAdapter` and decorate with `@register_adapter("<name>")`
3. Implement `wrap(client, **kwargs)` — return the wrapped client
4. Use `self.on_llm_start/end`, `self.on_tool_start/end`, `self.on_error` to record state
5. Add the import to `agent_recall_ai/adapters/__init__.py`
6. Add `<framework>` to `[project.optional-dependencies]` in `pyproject.toml`
7. Write tests in `tests/test_adapters.py` — mock the framework with `unittest.mock`

See `adapters/crewai_adapter.py` for a clean example.

---

## Writing tests

```bash
# Run just your new tests
pytest tests/test_your_module.py -v

# Run the full suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=agent_recall_ai --cov-report=term-missing
```

Rules:
- Use `MemoryStore` in tests — never write to disk
- Mock external SDK calls with `unittest.mock.patch`
- One test per behavior, not one test per function
- Test the sad path (exceptions, empty input, edge cases) not just the happy path

---

## Code style

- Python 3.10+ syntax throughout
- Type hints on every function signature
- Docstrings for public classes and methods (business outcome, not implementation detail)
- `ruff` for linting — run before every commit: `ruff check agent_recall_ai/`
- No `datetime.utcnow()` — use `datetime.now(timezone.utc).replace(tzinfo=None)`
- No bare `except:` — catch specific exceptions

---

## Commit messages

```
feat: add AutoGen adapter for multi-agent workflows
fix: repair orphaned tool_call IDs in long sessions
docs: add CrewAI example to README
test: add regression for cost monitor double-fire bug
refactor: extract conversation repair into standalone function
```

---

## Architecture decisions already made

Before proposing a change to core architecture, read `CLAUDE.md` — it documents
every architectural decision and why alternatives were rejected. Reopening a
closed decision needs strong new evidence.

---

## Pull request checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] No ruff errors: `ruff check agent_recall_ai/`
- [ ] New public API has docstrings
- [ ] `CLAUDE.md` updated if an architectural decision was made
- [ ] `CHANGELOG.md` entry added (Unreleased section)

---

## Community

- **Bugs**: [Open an issue](https://github.com/srinathsankara/agent-recall-ai/issues/new?template=bug_report.md)
- **Feature requests**: [Open an issue](https://github.com/srinathsankara/agent-recall-ai/issues/new?template=feature_request.md)
- **Questions**: [GitHub Discussions](https://github.com/srinathsankara/agent-recall-ai/discussions)

---

*By contributing, you agree that your contributions will be licensed under the MIT License.*
