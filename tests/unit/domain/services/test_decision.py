"""Tests for the domain service decide() function."""

from __future__ import annotations

from dynamic_prompt_core.domain.services.decision import decide


def _metrics(macro: float, minority: float) -> dict:
    return {"f1": {"macro": macro, "minority": minority}}


def test_accept_on_primary_improvement():
    """New primary strictly higher -> accept."""
    metrics_new = _metrics(0.85, 0.60)
    metrics_active = _metrics(0.80, 0.65)
    decision, reason = decide(metrics_new, metrics_active, "macro_f1", "minority_f1")
    assert decision == "accept"
    assert "improved" in reason


def test_accept_on_tie_with_tie_breaker_improvement():
    """Primary tied, tie-breaker strictly higher -> accept."""
    metrics_new = _metrics(0.80, 0.70)
    metrics_active = _metrics(0.80, 0.65)
    decision, reason = decide(metrics_new, metrics_active, "macro_f1", "minority_f1")
    assert decision == "accept"
    assert "tied" in reason
    assert "improved" in reason


def test_rollback_on_tie_without_tie_breaker_improvement():
    """Primary tied, tie-breaker equal -> rollback."""
    metrics_new = _metrics(0.80, 0.60)
    metrics_active = _metrics(0.80, 0.65)
    decision, reason = decide(metrics_new, metrics_active, "macro_f1", "minority_f1")
    assert decision == "rollback"
    assert "tied" in reason
    assert "did not improve" in reason


def test_rollback_on_tie_with_tie_breaker_equal():
    """Primary tied, tie-breaker also equal -> rollback."""
    metrics_new = _metrics(0.80, 0.65)
    metrics_active = _metrics(0.80, 0.65)
    decision, reason = decide(metrics_new, metrics_active, "macro_f1", "minority_f1")
    assert decision == "rollback"
    assert "tied" in reason
    assert "did not improve" in reason


def test_rollback_on_primary_degradation():
    """New primary strictly lower -> rollback."""
    metrics_new = _metrics(0.75, 0.70)
    metrics_active = _metrics(0.80, 0.65)
    decision, reason = decide(metrics_new, metrics_active, "macro_f1", "minority_f1")
    assert decision == "rollback"
    assert "degraded" in reason
