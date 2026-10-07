"""evaluate_stop_criteria: content-based cycle stop-criteria evaluation use case.

Evaluates seven criteria in a deterministic order and returns a continue/stop
decision.  When the decision is ``stop``, persists a ``stop_decision`` artifact
via ``run_repository`` and appends an event to the cycle log.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.application.ports.outbound.stop_criteria import (
    StopDecision,
    StopEvaluationContext,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.artifact import (
    write_stop_decision_artifact,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.criteria import (
    CHECK_ORDER,
    CRITERION_REASONS,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.deps import (
    EvaluateStopCriteriaDeps,
)

log = logging.getLogger(__name__)

__all__ = ["evaluate_stop_criteria"]


def _log_event(
    event: str,
    round_number: int,
    details: dict[str, object],
    log_path: str,
) -> None:
    """Append a JSON event to the cycle log when ``log_path`` is set."""
    if not log_path:
        return
    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round": round_number,
        "event": event,
        "details": details,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def evaluate_stop_criteria(
    deps: EvaluateStopCriteriaDeps,
    context: StopEvaluationContext,
    output_dir: str = "",
) -> StopDecision:
    """Evaluate all enabled stop criteria and return a continue/stop decision.

    Criteria are checked in ``CHECK_ORDER`` (critical → qualitative →
    rollback).  The first triggered criterion is the ``reason``; all triggered
    criteria are recorded in ``triggered_criteria``.  When the decision is
    ``stop``, a ``stop_decision`` artifact is written via ``run_repository``.
    """
    if not context.metric_history:
        decision = StopDecision(
            should_stop=False,
            reason=None,
            triggered_criteria=[],
            round_number=context.round_counter,
        )
        log.debug("empty metric history → continue")
        return decision

    enabled = None
    if context.config is not None:
        enabled = getattr(context.config, "enabled_criteria", None)

    triggered: list[str] = []
    for func in CHECK_ORDER:
        reason_name = CRITERION_REASONS.get(func.__name__, "")
        if enabled is not None and reason_name not in enabled:
            continue
        result = func(context)
        if result is not None:
            triggered.append(result)

    should_stop = len(triggered) > 0
    stop_reason = triggered[0] if triggered else None

    plateau_window = 3
    if context.config is not None:
        plateau_window = getattr(context.config, "plateau_window", 3)
    snapshot = list(context.metric_history[-plateau_window:])

    counters: dict[str, int] = {
        "rounds": context.round_counter,
        "rollbacks": context.rollback_counter,
        "tokens": context.total_teacher_tokens,
        "candidates": context.candidate_queue_size,
    }

    decision = StopDecision(
        should_stop=should_stop,
        reason=stop_reason,
        triggered_criteria=triggered,
        round_number=context.round_counter,
        metric_snapshot=snapshot,
        counters=counters,
        run_id=context.run_id,
    )

    _log_event(
        "stop_evaluation",
        context.round_counter,
        {
            "triggered_criteria": triggered,
            "decision": "stop" if should_stop else "continue",
            "reason": stop_reason,
        },
        context.log_path,
    )

    if should_stop and output_dir:
        write_stop_decision_artifact(decision, deps.run_repository, output_dir)

    return decision
