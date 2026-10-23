"""Tests for comparator.decide — the production accept/rollback decision."""

from __future__ import annotations

from dynamic_prompt_core.application.use_cases.compare_versions.comparator import (
    VersionComparatorConfig,
    decide,
)


def _metrics(macro: float, minority: float) -> dict:
    return {"f1": {"macro": macro, "minority": minority}}


_CFG = VersionComparatorConfig(
    decision_metric="macro_f1",
    tie_breaker_metric="minority_f1",
)


def test_accept_on_primary_improvement():
    decision, _ = decide(_metrics(0.85, 0.60), _metrics(0.80, 0.65), _CFG)
    assert decision == "accept"


def test_accept_on_tie_with_tie_breaker_improvement():
    """Primary tied, tie-breaker strictly higher -> accept."""
    decision, reason = decide(_metrics(0.80, 0.70), _metrics(0.80, 0.65), _CFG)
    assert decision == "accept"
    assert "tied" in reason


def test_rollback_on_tie_without_tie_breaker_improvement():
    decision, reason = decide(_metrics(0.80, 0.60), _metrics(0.80, 0.65), _CFG)
    assert decision == "rollback"
    assert "did not improve" in reason


def test_rollback_on_primary_degradation():
    decision, _ = decide(_metrics(0.75, 0.70), _metrics(0.80, 0.65), _CFG)
    assert decision == "rollback"


def test_rollback_when_new_metrics_f1_is_none():
    decision, reason = decide({"f1": None}, _metrics(0.80, 0.65), _CFG)
    assert decision == "rollback"
    assert "unavailable" in reason


def test_rollback_when_active_metrics_empty():
    decision, reason = decide(_metrics(0.85, 0.60), {}, _CFG)
    assert decision == "rollback"
    assert "unavailable" in reason
