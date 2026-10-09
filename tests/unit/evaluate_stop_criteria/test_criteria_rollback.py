"""Unit tests for the rollback streak criterion."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.criteria import (
    _check_rollback_streak,
)
from tests.unit.evaluate_stop_criteria.conftest import make_context


def test_rollback_streak():
    ctx = make_context(rollback_counter=2, max_consecutive_rollbacks=2)
    assert _check_rollback_streak(ctx) == "rollback_streak"


def test_below_limit():
    ctx = make_context(rollback_counter=1, max_consecutive_rollbacks=2)
    assert _check_rollback_streak(ctx) is None


def test_exceeds_limit():
    ctx = make_context(rollback_counter=5, max_consecutive_rollbacks=3)
    assert _check_rollback_streak(ctx) == "rollback_streak"
