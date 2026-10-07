"""Unit tests for the plateau detection criterion."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.criteria import (
    _check_plateau,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
    StopCriteriaConfig,
)

from tests.unit.evaluate_stop_criteria.conftest import make_context, metrics


def test_plateau_detected():
    ctx = make_context(metric_history=metrics(0.50, 0.505, 0.508, 0.509))
    assert _check_plateau(ctx) == "plateau_detected"


def test_improvement_resets_window():
    ctx = make_context(metric_history=metrics(0.50, 0.525, 0.530, 0.531))
    assert _check_plateau(ctx) is None


def test_window_not_reached():
    ctx = make_context(metric_history=metrics(0.50, 0.505, 0.508))
    assert _check_plateau(ctx) is None


def test_custom_plateau_window():
    cfg = StopCriteriaConfig(plateau_window=2, min_improvement=0.01)
    ctx = make_context(
        metric_history=metrics(0.50, 0.505, 0.508),
        config=cfg,
    )
    assert _check_plateau(ctx) == "plateau_detected"


def test_custom_decision_metric():
    cfg = StopCriteriaConfig(decision_metric="accuracy", plateau_window=3)
    ctx = make_context(
        metric_history=metrics(0.50, 0.50, 0.50, 0.50, metric="accuracy"),
        config=cfg,
    )
    assert _check_plateau(ctx) == "plateau_detected"


def test_improvement_above_threshold_resets():
    cfg = StopCriteriaConfig(plateau_window=3, min_improvement=0.01)
    ctx = make_context(
        metric_history=metrics(0.50, 0.53, 0.535, 0.536),
        config=cfg,
    )
    assert _check_plateau(ctx) is None
