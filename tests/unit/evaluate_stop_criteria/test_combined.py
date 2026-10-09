"""Unit tests for the combined stop-criteria decision and check order."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
    StopCriteriaConfig,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.deps import (
    EvaluateStopCriteriaDeps,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.evaluate import (
    evaluate_stop_criteria,
)
from tests.unit.evaluate_stop_criteria.conftest import make_context, metrics


class _MockRunRepository:
    def save_results(self, results, run_id, path):
        return path

    def load_results(self, path):
        return []


def _deps() -> EvaluateStopCriteriaDeps:
    return EvaluateStopCriteriaDeps(run_repository=_MockRunRepository())


async def test_budget_before_plateau():
    cfg = StopCriteriaConfig(max_total_rounds=2, plateau_window=3)
    ctx = make_context(
        round_counter=3,
        metric_history=metrics(0.50, 0.50, 0.50, 0.50),
        config=cfg,
    )
    decision = await evaluate_stop_criteria(_deps(), ctx)
    assert decision.should_stop is True
    assert decision.reason == "budget_exhausted"
    assert "plateau_detected" in decision.triggered_criteria
    assert decision.triggered_criteria[0] == "budget_exhausted"


async def test_plateau_before_rollback():
    cfg = StopCriteriaConfig(plateau_window=3)
    ctx = make_context(
        round_counter=4,
        rollback_counter=2,
        max_consecutive_rollbacks=2,
        candidate_queue_size=1,
        metric_history=metrics(0.50, 0.50, 0.50, 0.50),
        config=cfg,
    )
    decision = await evaluate_stop_criteria(_deps(), ctx)
    assert decision.should_stop is True
    assert decision.reason == "plateau_detected"
    assert "rollback_streak" in decision.triggered_criteria


async def test_no_criteria_continue():
    cfg = StopCriteriaConfig(plateau_window=3)
    ctx = make_context(
        round_counter=1,
        candidate_queue_size=1,
        metric_history=metrics(0.50, 0.55),
        config=cfg,
    )
    decision = await evaluate_stop_criteria(_deps(), ctx)
    assert decision.should_stop is False
    assert decision.reason is None
    assert decision.triggered_criteria == []


async def test_enabled_criteria_filters_budget():
    cfg = StopCriteriaConfig(
        enabled_criteria=["plateau_detected"],
        max_total_rounds=2,
        plateau_window=3,
    )
    ctx = make_context(
        round_counter=3,
        candidate_queue_size=1,
        metric_history=metrics(0.50, 0.50, 0.50, 0.50),
        config=cfg,
    )
    decision = await evaluate_stop_criteria(_deps(), ctx)
    assert decision.should_stop is True
    assert decision.reason == "plateau_detected"
    assert "budget_exhausted" not in decision.triggered_criteria
