"""Shared fixtures and helpers for evaluate_stop_criteria unit tests."""
from __future__ import annotations

from typing import Any

import pytest

from dynamic_prompt_core.application.ports.outbound.stop_criteria import (
    StopEvaluationContext,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
    StopCriteriaConfig,
)


def make_context(
    *,
    round_counter: int = 0,
    rollback_counter: int = 0,
    max_consecutive_rollbacks: int = 2,
    candidate_queue_size: int = 0,
    new_candidates_found: int = 0,
    metric_history: list[dict[str, float]] | None = None,
    decision_history: list[str] | None = None,
    rule_set_history: list[frozenset[str]] | None = None,
    total_teacher_tokens: int = 0,
    run_id: str = "test-run",
    config: StopCriteriaConfig | None = None,
) -> StopEvaluationContext:
    return StopEvaluationContext(
        round_counter=round_counter,
        rollback_counter=rollback_counter,
        max_consecutive_rollbacks=max_consecutive_rollbacks,
        candidate_queue_size=candidate_queue_size,
        new_candidates_found=new_candidates_found,
        metric_history=metric_history or [],
        decision_history=decision_history or [],
        rule_set_history=rule_set_history or [],
        total_teacher_tokens=total_teacher_tokens,
        run_id=run_id,
        config=config or StopCriteriaConfig(),
    )


def metrics(*values: float, metric: str = "macro_f1") -> list[dict[str, float]]:
    """Build a metric history from a sequence of macro_f1 values."""
    return [{metric: v, "accuracy": v, "minority_f1": v} for v in values]


@pytest.fixture
def make_ctx() -> Any:
    return make_context
