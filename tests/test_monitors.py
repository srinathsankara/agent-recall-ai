"""Tests for all real-time monitors."""
from __future__ import annotations

import pytest

from agent_recall_ai.core.state import AlertType, SessionStatus, TaskState, TokenUsage
from agent_recall_ai.monitors.cost_monitor import CostMonitor, CostBudgetExceeded
from agent_recall_ai.monitors.token_monitor import TokenMonitor
from agent_recall_ai.monitors.drift_monitor import DriftMonitor
from agent_recall_ai.monitors.package_monitor import (
    PackageHallucinationMonitor,
    _normalize_package,
    _is_suspicious,
)
from agent_recall_ai.monitors.tool_bloat_monitor import ToolBloatMonitor


def make_state(cost: float = 0.0, tokens: int = 0, utilization: float = 0.0) -> TaskState:
    state = TaskState(session_id="test")
    state.cost_usd = cost
    state.token_usage.add(prompt=tokens)
    state.context_utilization = utilization
    return state


# ── CostMonitor ───────────────────────────────────────────────────────────────

class TestCostMonitor:
    def test_no_alert_below_warn_threshold(self):
        m = CostMonitor(budget_usd=5.00, warn_at=0.80)
        state = make_state(cost=3.00)
        alerts = m.check(state)
        assert alerts == []

    def test_warns_at_threshold(self):
        m = CostMonitor(budget_usd=5.00, warn_at=0.80)
        state = make_state(cost=4.10)
        alerts = m.check(state)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == AlertType.COST_WARNING

    def test_warn_fires_only_once(self):
        m = CostMonitor(budget_usd=5.00, warn_at=0.80)
        state = make_state(cost=4.10)
        m.check(state)
        state.cost_usd = 4.50
        alerts = m.check(state)
        assert alerts == []   # already warned

    def test_raises_on_exceed_when_configured(self):
        m = CostMonitor(budget_usd=2.00, raise_on_exceed=True)
        state = make_state(cost=2.50)
        with pytest.raises(CostBudgetExceeded) as exc_info:
            m.check(state)
        assert exc_info.value.cost_usd == 2.50

    def test_no_raise_when_disabled(self):
        m = CostMonitor(budget_usd=2.00, raise_on_exceed=False)
        state = make_state(cost=2.50)
        alerts = m.check(state)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == AlertType.COST_EXCEEDED

    def test_exceed_fires_only_once(self):
        m = CostMonitor(budget_usd=2.00, raise_on_exceed=False)
        state = make_state(cost=2.50)
        m.check(state)
        state.cost_usd = 3.00
        alerts = m.check(state)
        assert alerts == []


# ── TokenMonitor ──────────────────────────────────────────────────────────────

class TestTokenMonitor:
    def test_no_alert_below_warn_threshold(self):
        m = TokenMonitor(model="gpt-4o-mini", warn_at=0.75)
        state = make_state(utilization=0.50)
        alerts = m.check(state)
        assert alerts == []

    def test_warn_alert_at_threshold(self):
        m = TokenMonitor(model="gpt-4o-mini", warn_at=0.75, compress_at=0.90)
        state = make_state(utilization=0.80)
        alerts = m.check(state)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == AlertType.TOKEN_PRESSURE

    def test_critical_alert_at_compress_threshold(self):
        m = TokenMonitor(model="gpt-4o-mini", warn_at=0.75, compress_at=0.90)
        state = make_state(utilization=0.92)
        alerts = m.check(state)
        # Should fire critical (not warn — jumps straight to compress)
        types = [a["alert_type"] for a in alerts]
        assert AlertType.TOKEN_CRITICAL in types

    def test_fires_only_once_per_level(self):
        m = TokenMonitor(model="gpt-4o-mini", warn_at=0.75, compress_at=0.90)
        state = make_state(utilization=0.80)
        m.check(state)
        state.context_utilization = 0.85
        alerts = m.check(state)
        assert alerts == []


# ── DriftMonitor ──────────────────────────────────────────────────────────────

class TestDriftMonitor:
    def test_no_alerts_without_constraints(self):
        m = DriftMonitor()
        state = TaskState(session_id="s1")
        state.add_decision("Deployed to production directly")
        alerts = m.check(state)
        assert alerts == []

    def test_detects_production_deploy_violation(self):
        m = DriftMonitor(sensitivity="medium")
        state = TaskState(session_id="s1")
        state.constraints = ["Do not deploy to production without approval"]
        state.add_decision("Deployed to production environment directly")
        alerts = m.check(state)
        assert len(alerts) >= 1
        assert alerts[0]["alert_type"] == AlertType.BEHAVIORAL_DRIFT

    def test_no_duplicate_alerts_for_same_decision(self):
        m = DriftMonitor(sensitivity="medium")
        state = TaskState(session_id="s1")
        state.constraints = ["Do not deploy to production without approval"]
        state.add_decision("Deployed to production environment directly")
        alerts1 = m.check(state)
        alerts2 = m.check(state)  # same decision, should not re-fire
        assert len(alerts2) == 0


# ── PackageHallucinationMonitor ───────────────────────────────────────────────

class TestPackageHallucinationMonitor:
    def test_known_package_not_flagged(self):
        m = PackageHallucinationMonitor()
        state = TaskState(session_id="s1")
        state.add_tool_call(
            "bash",
            input_summary="pip install requests",
            output_summary="Successfully installed requests-2.32.0",
        )
        alerts = m.on_tool_call(state)
        assert alerts == []

    def test_suspicious_package_flagged(self):
        m = PackageHallucinationMonitor()
        state = TaskState(session_id="s1")
        state.add_tool_call(
            "bash",
            input_summary="pip install ai-langchain-helper-pro-max",
            output_summary="",
        )
        alerts = m.on_tool_call(state)
        assert len(alerts) >= 1
        assert alerts[0]["alert_type"] == AlertType.PACKAGE_HALLUCINATION

    def test_normalize_package(self):
        assert _normalize_package("requests==2.31.0") == "requests"
        assert _normalize_package("PyYAML[extras]>=6.0") == "pyyaml"
        assert _normalize_package("scikit_learn") == "scikit-learn"

    def test_suspicious_double_hyphen(self):
        assert _is_suspicious("ai--helper") is not None

    def test_real_packages_not_suspicious(self):
        for pkg in ["requests", "numpy", "fastapi", "pydantic"]:
            assert _is_suspicious(pkg) is None

    def test_no_duplicate_alerts(self):
        m = PackageHallucinationMonitor()
        state = TaskState(session_id="s1")
        state.add_tool_call("bash", input_summary="pip install fake-ai-wrapper-ultra")
        m.on_tool_call(state)
        state.add_tool_call("bash", input_summary="pip install fake-ai-wrapper-ultra")
        alerts = m.on_tool_call(state)
        assert alerts == []   # already alerted for this package


# ── ToolBloatMonitor ──────────────────────────────────────────────────────────

class TestToolBloatMonitor:
    def test_small_output_no_alert(self):
        m = ToolBloatMonitor(max_output_tokens=1000)
        state = TaskState(session_id="s1")
        state.add_tool_call("bash", output_summary="ok", output_tokens=50)
        alerts = m.on_tool_call(state)
        assert alerts == []

    def test_large_output_triggers_alert(self):
        m = ToolBloatMonitor(max_output_tokens=500, auto_compress=False)
        state = TaskState(session_id="s1")
        state.add_tool_call("ls", output_summary="x" * 2000, output_tokens=600)
        alerts = m.on_tool_call(state)
        assert len(alerts) >= 1
        assert alerts[0]["alert_type"] == AlertType.TOOL_BLOAT

    def test_auto_compress_modifies_output(self):
        m = ToolBloatMonitor(max_output_tokens=100, auto_compress=True)
        state = TaskState(session_id="s1")
        long_output = "line\n" * 500
        state.add_tool_call("cat", output_summary=long_output, output_tokens=len(long_output) // 4)
        m.on_tool_call(state)
        # The last tool call should have been compressed
        assert state.tool_calls[-1].compressed is True
