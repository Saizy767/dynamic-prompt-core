"""Unit tests for the metric degradation detection criterion."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
    StopCriteriaConfig,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.criteria import (
    _check_degradation,
)
from tests.unit.evaluate_stop_criteria.conftest import make_context, metrics


def test_degradation_detected():
    ctx = make_context(metric_history=metrics(0.50, 0.49, 0.48, 0.47))
    assert _check_degradation(ctx) == "metric_degradation"


def test_single_drop_not_degradation():
    ctx = make_context(metric_history=metrics(0.50, 0.49, 0.50, 0.51))
    assert _check_degradation(ctx) is None


def test_exactly_window_decreases():
    cfg = StopCriteriaConfig(degradation_window=3)
    ctx = make_context(
        metric_history=metrics(0.50, 0.49, 0.48, 0.47),
        config=cfg,
    )
    assert _check_degradation(ctx) == "metric_degradation"


def test_window_not_reached():
    ctx = make_context(metric_history=metrics(0.50, 0.49, 0.48))
    assert _check_degradation(ctx) is None


def test_custom_degradation_window():
    cfg = StopCriteriaConfig(degradation_window=2)
    ctx = make_context(
        metric_history=metrics(0.50, 0.49, 0.48),
        config=cfg,
    )
    assert _check_degradation(ctx) == "metric_degradation"
