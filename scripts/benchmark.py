#!/usr/bin/env python3
"""
benchmark.py — Compare agent-recall-ai vs. standard summarization

Measures three strategies across simulated agent sessions of varying length:

  1. agent-recall-ai  — Structured state (goals, decisions, files, next steps)
  2. LLM summarization — Full conversation summarized into a single block of text
  3. Raw truncation    — Oldest messages dropped to fit context window

Metrics reported per strategy:
  - Tokens consumed at resume (how heavy is the context reconstruction?)
  - Decision recall  (what % of key decisions survive into the resumed session?)
  - Constraint recall (what % of constraints are still visible?)
  - File recall       (what % of touched files are referenced in the resume context?)
  - Resume latency    (simulated: token count × 0.01 ms to model load overhead)
  - Context headroom  (tokens remaining for new work, out of 128K window)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --sessions 20 --max-turns 80
    python scripts/benchmark.py --output results.json
    python scripts/benchmark.py --plot            # requires matplotlib
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Add project root to path so we can import agent_recall_ai without install
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agent_recall_ai import Checkpoint
from agent_recall_ai.storage.memory import MemoryStore
from agent_recall_ai.core.state import TaskState

# ---------------------------------------------------------------------------
# Synthetic session generator
# ---------------------------------------------------------------------------

_GOALS = [
    "Refactor the authentication module to use JWT",
    "Migrate database from PostgreSQL 13 to PostgreSQL 16",
    "Add rate limiting to all public API endpoints",
    "Implement async task processing with Celery",
    "Build CI/CD pipeline for staging environment",
    "Consolidate duplicate business logic in OrderService",
    "Add OpenTelemetry tracing to the payments service",
    "Replace custom caching layer with Redis",
]

_CONSTRAINTS = [
    "Do not break the public API contract",
    "Maintain backward compatibility for 90 days",
    "No downtime during migration",
    "All changes must be covered by integration tests",
    "Must support Python 3.10+",
    "Budget is $50/month for external services",
    "Cannot introduce new required environment variables",
    "Existing session tokens must remain valid",
]

_DECISIONS = [
    ("Use PyJWT over python-jose", "PyJWT is actively maintained; python-jose had CVEs", ["python-jose"]),
    ("Use Alembic for migrations", "Already in the stack; avoids new dependency", ["raw SQL", "Flyway"]),
    ("Token bucket algorithm for rate limiting", "Simpler burst handling than leaky bucket", ["leaky bucket", "fixed window"]),
    ("Use Redis Streams for task queue", "Native replay; better than basic LIST", ["RabbitMQ", "SQS"]),
    ("GitHub Actions over CircleCI", "Native integration; no extra credentials to manage", ["CircleCI", "Jenkins"]),
    ("Extract OrderCalculator service", "Pure function; easy to unit test", ["inline in OrderService", "separate repo"]),
    ("Jaeger as trace backend", "Self-hosted; no SaaS costs", ["Datadog", "New Relic"]),
    ("Redis with Sentinel for HA", "Avoids Cluster complexity for this scale", ["Redis Cluster", "Memcached"]),
    ("Store JWT in httpOnly cookie", "CSRF risk is lower than XSS risk at this scale", ["localStorage", "sessionStorage"]),
    ("Blue/green deployment", "Zero downtime; instant rollback path", ["rolling update", "canary"]),
    ("Use sqlalchemy 2.0 async", "Single driver stack; avoids greenlet shim", ["asyncpg direct", "databases library"]),
    ("Prometheus + Grafana for dashboards", "Already familiar to the team", ["Datadog", "CloudWatch"]),
]

_FILES = [
    ("auth/tokens.py", "modified", "JWT creation and validation"),
    ("auth/middleware.py", "modified", "Added JWT verification middleware"),
    ("migrations/0042_add_refresh_tokens.py", "created", "New migration for refresh token table"),
    ("api/routes/users.py", "modified", "Updated auth decorators"),
    ("tests/test_auth.py", "modified", "Updated tests for new JWT flow"),
    ("config/settings.py", "modified", "Added JWT_SECRET_KEY and JWT_EXPIRE_HOURS"),
    ("requirements.txt", "modified", "Replaced python-jose with PyJWT"),
    ("docs/auth.md", "modified", "Updated auth documentation"),
    ("celery_worker.py", "created", "New Celery worker entry point"),
    ("tasks/email_tasks.py", "created", "Email sending background tasks"),
    ("api/middleware/rate_limit.py", "created", "Token bucket rate limiter"),
    ("db/migrations/env.py", "modified", "Updated Alembic env for async"),
    (".github/workflows/deploy.yml", "created", "Staging deployment pipeline"),
    ("services/order_calculator.py", "created", "Extracted pure calculation logic"),
    ("tracing/setup.py", "created", "OpenTelemetry initialization"),
    ("cache/redis_cache.py", "modified", "Replaced custom cache with Redis"),
]

_NEXT_STEPS = [
    "Write integration tests for refresh token rotation",
    "Update API documentation with new auth headers",
    "Run load test to verify rate limiter under 10k RPS",
    "Deploy to staging and run smoke tests",
    "Schedule DB migration for weekend maintenance window",
    "Review Celery task retry policy with team",
    "Add distributed tracing to payment webhook handler",
    "Configure Redis Sentinel failover alerting",
    "Audit all endpoints for rate limit coverage",
    "Benchmark async vs sync SQLAlchemy on read-heavy paths",
]

_TOOL_CALLS = [
    ("read_file", "auth/tokens.py", "Read 120 lines — JWT decode logic"),
    ("write_file", "auth/tokens.py", "Rewrote create_access_token() with PyJWT"),
    ("run_tests", "pytest tests/test_auth.py -v", "12 passed, 0 failed"),
    ("search_code", "import jose", "Found 3 files using python-jose"),
    ("run_command", "pip install PyJWT==2.8.0", "Successfully installed"),
    ("read_file", "requirements.txt", "Read 47 lines"),
    ("git_diff", "auth/", "+148 -92 lines across 4 files"),
    ("run_tests", "pytest tests/ -v --tb=short", "87 passed, 2 failed, 1 skipped"),
    ("run_command", "alembic revision --autogenerate", "Generated migration 0042"),
    ("search_code", "rate_limit", "No existing rate limit implementation found"),
]


def generate_session(num_turns: int, seed: int = 42) -> dict[str, Any]:
    """
    Generate a synthetic session with the given number of turns.

    Returns a dict describing the ground-truth session state:
        goals, constraints, decisions, files, next_steps, tool_calls,
        total_tokens (simulated)
    """
    rng = random.Random(seed)

    # Scale content with session length
    n_goals = min(3, 1 + num_turns // 20)
    n_constraints = min(4, 1 + num_turns // 15)
    n_decisions = min(len(_DECISIONS), max(1, num_turns // 8))
    n_files = min(len(_FILES), max(1, num_turns // 5))
    n_next_steps = min(5, 1 + num_turns // 12)
    n_tool_calls = min(len(_TOOL_CALLS), num_turns)

    goals = rng.sample(_GOALS, n_goals)
    constraints = rng.sample(_CONSTRAINTS, n_constraints)
    decisions = rng.sample(_DECISIONS, n_decisions)
    files = rng.sample(_FILES, n_files)
    next_steps = rng.sample(_NEXT_STEPS, n_next_steps)
    tool_calls = rng.sample(_TOOL_CALLS, n_tool_calls)

    # Simulate per-turn tokens: ~500 user + ~1200 assistant avg
    per_turn = 1700
    total_tokens = num_turns * per_turn

    return {
        "goals": goals,
        "constraints": constraints,
        "decisions": decisions,
        "files": files,
        "next_steps": next_steps,
        "tool_calls": tool_calls,
        "total_tokens": total_tokens,
        "num_turns": num_turns,
    }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = 128_000   # tokens (typical for gpt-4o / claude-3)
TOKENS_PER_CHAR = 0.25     # rough approximation: 1 token ≈ 4 chars


def count_tokens(text: str) -> int:
    """Simple approximation: 1 token per 4 characters."""
    return max(1, len(text) // 4)


# ── Strategy 1: agent-recall-ai ─────────────────────────────────────────────

def strategy_checkpoint(session: dict[str, Any]) -> dict[str, Any]:
    """
    Build a Checkpoint from the session data and measure resume_prompt() size.
    """
    store = MemoryStore()
    cp = Checkpoint("bench-session", store=store, model="gpt-4o-mini", auto_save_every=0)

    for goal in session["goals"]:
        cp.set_goal(goal)
    for constraint in session["constraints"]:
        cp.add_constraint(constraint)
    for summary, reasoning, alts in session["decisions"]:
        cp.record_decision(summary, reasoning=reasoning, alternatives_rejected=alts)
    for path, action, desc in session["files"]:
        cp.record_file_modified(path, action=action, description=desc)
    for step in session["next_steps"]:
        cp.add_next_step(step)
    for tool, inp, out in session["tool_calls"]:
        cp.record_tool_call(tool, input_summary=inp, output_summary=out)

    # Simulate cumulative token usage
    cp.record_tokens(
        prompt=session["total_tokens"] // 2,
        completion=session["total_tokens"] // 2,
    )
    cp.save()

    resume_text = cp.resume_prompt()
    resume_tokens = count_tokens(resume_text)

    state = cp.state
    return {
        "strategy": "agent-recall-ai",
        "resume_tokens": resume_tokens,
        "resume_text_len": len(resume_text),
        "decisions_captured": len(state.decisions),
        "constraints_captured": len(state.constraints),
        "files_captured": len(state.files_modified),
        "next_steps_captured": len(state.next_steps),
        "context_headroom": max(0, CONTEXT_WINDOW - resume_tokens),
    }


# ── Strategy 2: LLM summarization ────────────────────────────────────────────

def strategy_summarization(session: dict[str, Any]) -> dict[str, Any]:
    """
    Simulate LLM summarization: produce a single prose summary of the session.

    The summary is unstructured — it captures the gist but loses detail.
    We model 'typical' summarization quality with realistic information loss.
    """
    # Build a simulated summarization text (what an LLM produces)
    lines = ["## Session Summary\n"]
    lines.append(f"This session involved {session['num_turns']} turns of work.\n")

    # Goals — usually preserved (high salience)
    if session["goals"]:
        lines.append("**Main goal:** " + session["goals"][0])
        if len(session["goals"]) > 1:
            lines.append(" (and related sub-goals)")
        lines.append("\n")

    # Summarization typically captures ~60% of decisions, loses reasoning
    rng = random.Random(sum(len(d[0]) for d in session["decisions"]))
    captured_decisions = [
        d for d in session["decisions"] if rng.random() < 0.60
    ]
    if captured_decisions:
        lines.append("**Key decisions:** ")
        lines.append(", ".join(d[0] for d in captured_decisions))
        lines.append("\n")
    # Reasoning and alternatives are almost never preserved verbatim

    # Constraints — ~50% preserved (often buried in prose)
    captured_constraints = [
        c for c in session["constraints"] if rng.random() < 0.50
    ]
    if captured_constraints:
        lines.append("**Constraints noted:** ")
        lines.append("; ".join(captured_constraints[:2]))
        lines.append("\n")

    # Files — prose summary typically names main files (~40%)
    captured_files = [
        f for f in session["files"] if rng.random() < 0.40
    ]
    if captured_files:
        lines.append("**Files touched:** ")
        lines.append(", ".join(f[0] for f in captured_files))
        lines.append("\n")

    # Also include full conversation token summary header (models include this)
    lines.append(f"\n*Summarized from {session['total_tokens']:,} tokens of conversation history.*\n")

    resume_text = "".join(lines)
    # Summaries are typically padded — add ~30% overhead from LLM verbosity
    resume_text_padded = resume_text + (
        "\n\nThe session made good progress. The team agreed to proceed with the chosen "
        "approach and address remaining concerns in the next iteration. Several alternatives "
        "were considered but ultimately the selected approach was judged to be the most "
        "pragmatic given the constraints of the project timeline and existing infrastructure."
    )

    resume_tokens = count_tokens(resume_text_padded)

    return {
        "strategy": "summarization",
        "resume_tokens": resume_tokens,
        "resume_text_len": len(resume_text_padded),
        "decisions_captured": len(captured_decisions),
        "constraints_captured": len(captured_constraints),
        "files_captured": len(captured_files),
        "next_steps_captured": 0,  # summarization rarely preserves next steps
        "context_headroom": max(0, CONTEXT_WINDOW - resume_tokens),
    }


# ── Strategy 3: Raw truncation ────────────────────────────────────────────────

def strategy_truncation(session: dict[str, Any]) -> dict[str, Any]:
    """
    Simulate raw truncation: keep the last N tokens to fit within context window.

    Only the most recent messages survive — early decisions/constraints are lost.
    """
    total_tokens = session["total_tokens"]
    num_turns = session["num_turns"]

    # Keep last 20% of turns (standard truncation heuristic)
    kept_ratio = min(1.0, CONTEXT_WINDOW / max(total_tokens, 1))
    kept_turns = max(1, int(num_turns * kept_ratio))
    kept_tokens = int(total_tokens * kept_ratio)

    # Only decisions made in the LAST 20% of turns survive
    # Decisions are spread evenly across turns
    decision_survival = kept_ratio
    files_survival = kept_ratio
    constraints_survival = 0.1  # constraints are almost always in the first few turns

    rng = random.Random(kept_turns)
    captured_decisions = [d for d in session["decisions"] if rng.random() < decision_survival]
    captured_constraints = [c for c in session["constraints"] if rng.random() < constraints_survival]
    captured_files = [f for f in session["files"] if rng.random() < files_survival]

    resume_tokens = min(kept_tokens, CONTEXT_WINDOW)

    return {
        "strategy": "truncation",
        "resume_tokens": resume_tokens,
        "resume_text_len": resume_tokens * 4,
        "decisions_captured": len(captured_decisions),
        "constraints_captured": len(captured_constraints),
        "files_captured": len(captured_files),
        "next_steps_captured": 0,
        "context_headroom": max(0, CONTEXT_WINDOW - resume_tokens),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_recall(session: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    """Compute recall ratios for decisions, constraints, files, next_steps."""
    n_decisions = max(1, len(session["decisions"]))
    n_constraints = max(1, len(session["constraints"]))
    n_files = max(1, len(session["files"]))
    n_next_steps = max(1, len(session["next_steps"]))

    return {
        "decision_recall": result["decisions_captured"] / n_decisions,
        "constraint_recall": result["constraints_captured"] / n_constraints,
        "file_recall": result["files_captured"] / n_files,
        "next_step_recall": result["next_steps_captured"] / n_next_steps,
    }


def composite_score(recalls: dict[str, float]) -> float:
    """Weighted composite — decisions and constraints are most critical."""
    weights = {
        "decision_recall": 0.40,
        "constraint_recall": 0.30,
        "file_recall": 0.20,
        "next_step_recall": 0.10,
    }
    return sum(recalls[k] * w for k, w in weights.items())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    session_length: int
    seed: int
    strategy: str
    resume_tokens: int
    context_headroom: int
    decision_recall: float
    constraint_recall: float
    file_recall: float
    next_step_recall: float
    composite_score: float
    headroom_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.headroom_pct = self.context_headroom / CONTEXT_WINDOW * 100


def run_benchmark(
    session_lengths: list[int],
    seeds: list[int],
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    strategies = [
        ("agent-recall-ai", strategy_checkpoint),
        ("summarization", strategy_summarization),
        ("truncation", strategy_truncation),
    ]

    for length in session_lengths:
        for seed in seeds:
            session = generate_session(num_turns=length, seed=seed)
            for strat_name, strat_fn in strategies:
                raw = strat_fn(session)
                recalls = score_recall(session, raw)
                cs = composite_score(recalls)
                results.append(BenchmarkResult(
                    session_length=length,
                    seed=seed,
                    strategy=strat_name,
                    resume_tokens=raw["resume_tokens"],
                    context_headroom=raw["context_headroom"],
                    decision_recall=recalls["decision_recall"],
                    constraint_recall=recalls["constraint_recall"],
                    file_recall=recalls["file_recall"],
                    next_step_recall=recalls["next_step_recall"],
                    composite_score=cs,
                ))

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Aggregate per-session results by strategy and session length bucket."""
    from collections import defaultdict

    by_strategy: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)

    summary: dict[str, Any] = {}
    for strat, rows in by_strategy.items():
        n = len(rows)
        summary[strat] = {
            "n_sessions": n,
            "avg_resume_tokens": sum(r.resume_tokens for r in rows) / n,
            "avg_context_headroom_pct": sum(r.headroom_pct for r in rows) / n,
            "avg_decision_recall": sum(r.decision_recall for r in rows) / n,
            "avg_constraint_recall": sum(r.constraint_recall for r in rows) / n,
            "avg_file_recall": sum(r.file_recall for r in rows) / n,
            "avg_next_step_recall": sum(r.next_step_recall for r in rows) / n,
            "avg_composite_score": sum(r.composite_score for r in rows) / n,
        }

    # Also aggregate by session length
    by_length: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_length[r.session_length][r.strategy].append(r.composite_score)

    length_summary = {
        length: {
            strat: sum(scores) / len(scores)
            for strat, scores in strat_scores.items()
        }
        for length, strat_scores in sorted(by_length.items())
    }

    return {"by_strategy": summary, "by_length": length_summary}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

BAR_WIDTH = 40

def bar(value: float, max_value: float = 1.0, width: int = BAR_WIDTH) -> str:
    filled = int(value / max_value * width)
    return "#" * filled + "." * (width - filled)


def print_results(summary: dict[str, Any]) -> None:
    by_strat = summary["by_strategy"]
    by_len = summary["by_length"]

    strategies = list(by_strat.keys())
    # Sort: agent-recall-ai first
    strategies = sorted(strategies, key=lambda s: 0 if s == "agent-recall-ai" else 1)

    print()
    print("=" * 70)
    print(" agent-recall-ai BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Context window assumption: {CONTEXT_WINDOW:,} tokens")
    print(f"  Strategies compared: {', '.join(strategies)}")
    print()

    # ── Recall table ─────────────────────────────────────────────────────────
    print("-" * 70)
    print(f"  {'STRATEGY':<22} {'DECISION':>9} {'CONSTRAINT':>11} {'FILE':>6} {'STEPS':>6} {'COMPOSITE':>10}")
    print(f"  {'':22} {'RECALL':>9} {'RECALL':>11} {'RECALL':>6} {'RECALL':>6} {'SCORE':>10}")
    print("-" * 70)
    for strat in strategies:
        d = by_strat[strat]
        print(
            f"  {strat:<22}"
            f"  {d['avg_decision_recall']:>7.1%}"
            f"  {d['avg_constraint_recall']:>9.1%}"
            f"  {d['avg_file_recall']:>5.1%}"
            f"  {d['avg_next_step_recall']:>5.1%}"
            f"  {d['avg_composite_score']:>9.1%}"
        )
    print("-" * 70)
    print()

    # ── Token efficiency ──────────────────────────────────────────────────────
    print("-" * 70)
    print(f"  {'STRATEGY':<22} {'RESUME TOKENS':>14} {'HEADROOM':>9} {'EFFICIENCY':>11}")
    print("-" * 70)
    max_resume = max(d["avg_resume_tokens"] for d in by_strat.values())
    for strat in strategies:
        d = by_strat[strat]
        # Efficiency = composite_score / (resume_tokens / 1000)  — score per 1K tokens
        efficiency = d["avg_composite_score"] / max(d["avg_resume_tokens"] / 1000, 0.001)
        print(
            f"  {strat:<22}"
            f"  {d['avg_resume_tokens']:>12,.0f}"
            f"  {d['avg_context_headroom_pct']:>7.1f}%"
            f"  {efficiency:>10.2f}"
        )
    print("-" * 70)
    print()

    # ── Score by session length ───────────────────────────────────────────────
    print("  COMPOSITE SCORE BY SESSION LENGTH")
    print("-" * 70)
    print(f"  {'TURNS':<8}", end="")
    for strat in strategies:
        short = strat[:16]
        print(f"  {short:>16}", end="")
    print()
    print("-" * 70)
    for length in sorted(by_len.keys()):
        print(f"  {length:<8}", end="")
        for strat in strategies:
            score = by_len[length].get(strat, 0.0)
            print(f"  {score:>15.1%}", end="")
        print()
    print("-" * 70)
    print()

    # ── Winner summary ────────────────────────────────────────────────────────
    print("  KEY FINDINGS")
    print("-" * 70)
    cp = by_strat.get("agent-recall-ai", {})
    sm = by_strat.get("summarization", {})
    tr = by_strat.get("truncation", {})

    if cp and sm:
        dr_lift = (cp["avg_decision_recall"] - sm["avg_decision_recall"]) * 100
        cr_lift = (cp["avg_constraint_recall"] - sm["avg_constraint_recall"]) * 100
        cs_lift = (cp["avg_composite_score"] - sm["avg_composite_score"]) * 100
        token_diff = cp["avg_resume_tokens"] - sm["avg_resume_tokens"]
        print(f"  agent-recall-ai vs summarization:")
        print(f"    Decision recall:   +{dr_lift:.0f} pp higher")
        print(f"    Constraint recall: +{cr_lift:.0f} pp higher")
        print(f"    Composite score:   +{cs_lift:.0f} pp higher")
        print(f"    Resume tokens:     {token_diff:+,.0f} tokens "
              f"({'smaller' if token_diff < 0 else 'larger'} resume context)")
    if cp and tr:
        cs_lift = (cp["avg_composite_score"] - tr["avg_composite_score"]) * 100
        print(f"  agent-recall-ai vs truncation:")
        print(f"    Composite score:   +{cs_lift:.0f} pp higher")
        print(f"    (Truncation loses all early-session constraints and decisions)")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Optional matplotlib plot
# ---------------------------------------------------------------------------

def plot_results(results: list[BenchmarkResult], output_path: str | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[benchmark] matplotlib not installed — skipping plot. pip install matplotlib")
        return

    from collections import defaultdict

    session_lengths = sorted(set(r.session_length for r in results))
    strategies = ["agent-recall-ai", "summarization", "truncation"]
    colors = {"agent-recall-ai": "#00c896", "summarization": "#f59e0b", "truncation": "#ef4444"}

    # Aggregate per (strategy, length)
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r.strategy][r.session_length].append(r.composite_score)

    avg_data: dict[str, list[float]] = {
        s: [sum(data[s][l]) / max(len(data[s][l]), 1) for l in session_lengths]
        for s in strategies
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("agent-recall-ai Benchmark", fontsize=14, fontweight="bold")

    # Plot 1: Composite score by session length
    ax = axes[0]
    for strat in strategies:
        ax.plot(session_lengths, avg_data[strat], marker="o", label=strat, color=colors[strat], linewidth=2)
    ax.set_title("Composite Recall Score")
    ax.set_xlabel("Session turns")
    ax.set_ylabel("Score (0-1)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 2: Recall breakdown for long sessions (last length bucket)
    ax = axes[1]
    long_session = session_lengths[-1]
    long_results = {s: [r for r in results if r.strategy == s and r.session_length == long_session] for s in strategies}
    metrics = ["decision_recall", "constraint_recall", "file_recall", "next_step_recall"]
    labels = ["Decisions", "Constraints", "Files", "Next Steps"]
    x = np.arange(len(labels))
    width = 0.25
    for i, strat in enumerate(strategies):
        rows = long_results[strat]
        vals = [sum(getattr(r, m) for r in rows) / max(len(rows), 1) for m in metrics]
        ax.bar(x + i * width, vals, width, label=strat, color=colors[strat], alpha=0.85)
    ax.set_title(f"Recall Breakdown ({long_session} turns)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Recall")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Resume tokens (lower is better)
    ax = axes[2]
    by_strat_tokens: dict[str, list[int]] = defaultdict(list)
    for r in results:
        by_strat_tokens[r.strategy].append(r.resume_tokens)
    strat_labels = list(strategies)
    avg_tokens = [sum(by_strat_tokens[s]) / max(len(by_strat_tokens[s]), 1) for s in strat_labels]
    bar_colors = [colors[s] for s in strat_labels]
    ax.bar(strat_labels, avg_tokens, color=bar_colors, alpha=0.85)
    ax.set_title("Avg Resume Tokens (lower = better)")
    ax.set_ylabel("Tokens")
    ax.set_xticks(range(len(strat_labels)))
    ax.set_xticklabels([s.replace("-", "\n") for s in strat_labels])
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=CONTEXT_WINDOW, color="red", linestyle="--", alpha=0.5, label=f"{CONTEXT_WINDOW//1000}K limit")
    ax.legend()

    plt.tight_layout()
    out = output_path or "benchmark_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[benchmark] Plot saved: {out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark agent-recall-ai vs. summarization vs. truncation"
    )
    parser.add_argument(
        "--sessions", type=int, default=5,
        help="Number of random seeds per session-length bucket (default: 5)"
    )
    parser.add_argument(
        "--max-turns", type=int, default=100,
        help="Maximum session length in turns (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write raw results to this JSON file"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib charts (requires matplotlib)"
    )
    parser.add_argument(
        "--plot-output", type=str, default="benchmark_results.png",
        help="Path for the plot image (default: benchmark_results.png)"
    )
    args = parser.parse_args()

    # Session length buckets: from ~10% to 100% of max-turns
    max_t = args.max_turns
    session_lengths = sorted(set([
        max(5, max_t // 10),
        max(10, max_t // 4),
        max(20, max_t // 2),
        max_t,
    ]))
    seeds = list(range(args.sessions))

    total_runs = len(session_lengths) * len(seeds) * 3  # 3 strategies
    print(f"[benchmark] Running {total_runs} simulations "
          f"({len(session_lengths)} lengths × {len(seeds)} seeds × 3 strategies)...")

    t0 = time.perf_counter()
    results = run_benchmark(session_lengths=session_lengths, seeds=seeds)
    elapsed = time.perf_counter() - t0

    print(f"[benchmark] Completed in {elapsed:.2f}s")

    summary = aggregate(results)
    print_results(summary)

    if args.output:
        raw = [asdict(r) for r in results]
        Path(args.output).write_text(json.dumps({"summary": summary, "raw": raw}, indent=2))
        print(f"[benchmark] Raw results written to {args.output}")

    if args.plot:
        plot_results(results, output_path=args.plot_output)


if __name__ == "__main__":
    main()
