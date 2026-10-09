"""Unit tests for the rule stagnation criterion."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
    StopCriteriaConfig,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.criteria import (
    _check_stagnation,
)
from tests.unit.evaluate_stop_criteria.conftest import make_context

A = frozenset({"r1", "r2", "r3"})
B = frozenset({"r1", "r2", "r4"})


def test_stagnation_detected():
    ctx = make_context(rule_set_history=[A, A, A, A])
    assert _check_stagnation(ctx) == "rule_stagnation"


def test_rule_added():
    ctx = make_context(rule_set_history=[A, A, A, B])
    assert _check_stagnation(ctx) is None


def test_rule_removed():
    ctx = make_context(rule_set_history=[A, A, B, B])
    assert _check_stagnation(ctx) is None


def test_window_not_reached():
    ctx = make_context(rule_set_history=[A, A, A])
    assert _check_stagnation(ctx) is None


def test_custom_stagnation_window():
    cfg = StopCriteriaConfig(stagnation_window=2)
    ctx = make_context(rule_set_history=[A, A, A], config=cfg)
    assert _check_stagnation(ctx) == "rule_stagnation"
