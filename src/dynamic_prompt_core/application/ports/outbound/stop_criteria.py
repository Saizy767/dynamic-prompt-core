"""Outbound port: contract for content-based cycle stop-criteria evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "StopCriteria",
    "StopDecision",
    "StopEvaluationContext",
]


@dataclass(frozen=True)
class StopEvaluationContext:
    """All cycle state the stop-criteria evaluator needs.

    ``metric_history`` is a list of per-round metric dicts with keys
    ``accuracy``, ``macro_f1``, ``minority_f1``.  ``decision_history`` is a
    list of ``"accept"`` / ``"rollback"`` strings.  ``rule_set_history`` is a
    list of ``frozenset[str]`` of rule ids in the active version per round.
    """

    round_counter: int
    rollback_counter: int
    max_consecutive_rollbacks: int
    candidate_queue_size: int
    new_candidates_found: int
    metric_history: list[dict[str, float]]
    decision_history: list[str]
    rule_set_history: list[frozenset[str]]
    total_teacher_tokens: int
    run_id: str
    log_path: str = ""
    config: Any = None


@dataclass(frozen=True)
class StopDecision:
    """Result of a stop-criteria evaluation."""

    should_stop: bool
    reason: str | None
    triggered_criteria: list[str]
    round_number: int
    metric_snapshot: list[dict[str, float]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    run_id: str = ""


@runtime_checkable
class StopCriteria(Protocol):
    """Outbound port: contract for evaluating cycle stop criteria."""

    async def evaluate(
        self,
        context: StopEvaluationContext,
    ) -> StopDecision:
        """Evaluate the cycle state and return a continue/stop decision."""
        ...
