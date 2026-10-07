"""Pure stop-criterion functions for the evaluate_stop_criteria use case.

Each function takes a ``StopEvaluationContext`` and returns the reason string
when the criterion triggers, or ``None`` when it does not.  The functions are
pure: they have no side effects and depend only on the context and config.
"""
from __future__ import annotations

from typing import Callable

from dynamic_prompt_core.application.ports.outbound.stop_criteria import (
    StopEvaluationContext,
)

__all__ = ["CHECK_ORDER"]

_REASON_BUDGET = "budget_exhausted"
_REASON_NO_CANDIDATES = "no_candidates_available"
_REASON_PLATEAU = "plateau_detected"
_REASON_DEGRADATION = "metric_degradation"
_REASON_STAGNATION = "rule_stagnation"
_REASON_ROLLBACK = "rollback_streak"


def _metric_values(ctx: StopEvaluationContext) -> list[float]:
    """Extract the configured decision metric from each round's metric dict."""
    metric_name = "macro_f1"
    if ctx.config is not None:
        metric_name = getattr(ctx.config, "decision_metric", "macro_f1")
    values: list[float] = []
    for m in ctx.metric_history:
        if metric_name in m:
            values.append(float(m[metric_name]))
        else:
            values.append(0.0)
    return values


def _check_budget(ctx: StopEvaluationContext) -> str | None:
    """Trigger when the round or token budget is exceeded."""
    if ctx.config is None:
        return None
    max_rounds = getattr(ctx.config, "max_total_rounds", 10)
    if ctx.round_counter > max_rounds:
        return _REASON_BUDGET
    max_tokens = getattr(ctx.config, "max_total_tokens", None)
    if max_tokens is not None and ctx.total_teacher_tokens > max_tokens:
        return _REASON_BUDGET
    return None


def _check_no_candidates(ctx: StopEvaluationContext) -> str | None:
    """Trigger when the queue is empty and no new candidates were found."""
    if ctx.candidate_queue_size == 0 and ctx.new_candidates_found == 0:
        return _REASON_NO_CANDIDATES
    return None


def _check_plateau(ctx: StopEvaluationContext) -> str | None:
    """Trigger when the metric does not improve for ``plateau_window`` rounds."""
    if ctx.config is None:
        return None
    window = getattr(ctx.config, "plateau_window", 3)
    min_improvement = getattr(ctx.config, "min_improvement", 0.01)

    values = _metric_values(ctx)
    if len(values) < window + 1:
        return None

    recent = values[-(window + 1):]
    for i in range(1, len(recent)):
        if recent[i] - recent[i - 1] > min_improvement:
            return None
    return _REASON_PLATEAU


def _check_degradation(ctx: StopEvaluationContext) -> str | None:
    """Trigger when the metric decreases for ``degradation_window`` rounds."""
    if ctx.config is None:
        return None
    window = getattr(ctx.config, "degradation_window", 3)

    values = _metric_values(ctx)
    if len(values) < window + 1:
        return None

    recent = values[-(window + 1):]
    for i in range(1, len(recent)):
        if recent[i] >= recent[i - 1]:
            return None
    return _REASON_DEGRADATION


def _check_stagnation(ctx: StopEvaluationContext) -> str | None:
    """Trigger when the rule set does not change for ``stagnation_window`` rounds."""
    if ctx.config is None:
        return None
    window = getattr(ctx.config, "stagnation_window", 3)

    history = ctx.rule_set_history
    if len(history) < window + 1:
        return None

    recent = history[-(window + 1):]
    first = recent[0]
    for entry in recent[1:]:
        if entry != first:
            return None
    return _REASON_STAGNATION


def _check_rollback_streak(ctx: StopEvaluationContext) -> str | None:
    """Trigger when the rollback counter reaches the limit."""
    if ctx.rollback_counter >= ctx.max_consecutive_rollbacks:
        return _REASON_ROLLBACK
    return None


CHECK_ORDER: tuple[Callable[[StopEvaluationContext], str | None], ...] = (
    _check_budget,
    _check_no_candidates,
    _check_plateau,
    _check_degradation,
    _check_stagnation,
    _check_rollback_streak,
)

CRITERION_REASONS: dict[str, str] = {
    "_check_budget": _REASON_BUDGET,
    "_check_no_candidates": _REASON_NO_CANDIDATES,
    "_check_plateau": _REASON_PLATEAU,
    "_check_degradation": _REASON_DEGRADATION,
    "_check_stagnation": _REASON_STAGNATION,
    "_check_rollback_streak": _REASON_ROLLBACK,
}
