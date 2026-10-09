"""Unit tests for the budget exhaustion criterion."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
    StopCriteriaConfig,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.criteria import (
    _check_budget,
)
from tests.unit.evaluate_stop_criteria.conftest import make_context


def test_round_budget_exceeded():
    cfg = StopCriteriaConfig(max_total_rounds=5)
    ctx = make_context(round_counter=6, config=cfg)
    assert _check_budget(ctx) == "budget_exhausted"


def test_token_budget_exceeded():
    cfg = StopCriteriaConfig(max_total_tokens=10000)
    ctx = make_context(total_teacher_tokens=10001, config=cfg)
    assert _check_budget(ctx) == "budget_exhausted"


def test_max_tokens_none_skips_token_check():
    cfg = StopCriteriaConfig(max_total_tokens=None)
    ctx = make_context(total_teacher_tokens=999999, round_counter=0, config=cfg)
    assert _check_budget(ctx) is None


def test_budget_not_exceeded():
    cfg = StopCriteriaConfig(max_total_rounds=10, max_total_tokens=10000)
    ctx = make_context(round_counter=5, total_teacher_tokens=5000, config=cfg)
    assert _check_budget(ctx) is None
