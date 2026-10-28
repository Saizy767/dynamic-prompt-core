"""Per-round reports, final summary, and append-only cycle log.

All artifacts are written via ``run_repository`` (JSONL) or direct JSON files
in ``output_dir``.  The module uses ``logging.getLogger(__name__)`` for runtime
logging and never calls ``logging.basicConfig``.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, cast

from dynamic_prompt_core.application.ports.outbound.run_repository import (
    RunRepository,
)
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState
from dynamic_prompt_core.infrastructure.storage.run_repository import _atomic_write

log = logging.getLogger(__name__)

_write_counter = itertools.count()

VALID_STOP_REASONS = frozenset(
    {
        "max_cycles_reached",
        "no_improvement",
        "unrecoverable_error",
        "plateau_detected",
        "budget_exhausted",
        "no_candidates_available",
        "metric_degradation",
        "rule_stagnation",
        "rollback_streak",
    }
)


class CycleOrchestratorError(ValueError):
    """Raised when the cycle orchestrator encounters an unrecoverable error."""


# --------------------------------------------------------------------------- #
#  Per-round report
# --------------------------------------------------------------------------- #
def write_report(
    round_number: int,
    active_version_at_start: str,
    new_version: str,
    decision: str,
    metrics_dev_active: dict[str, Any],
    metrics_dev_new: dict[str, Any],
    metrics_holdout_active: dict[str, Any],
    metrics_holdout_new: dict[str, Any],
    rollback_counter: int,
    changed_decisions: list[dict[str, Any]],
    next_action: str,
    run_id: str,
    run_repository: RunRepository,
    output_dir: str,
) -> str:
    """Serialize a per-cycle report to ``report_cycle/report_{run_id}_cycle{N}_{ts}.json``."""
    output_dir = os.path.join(output_dir, "report_cycle")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"report_{run_id}_cycle{round_number}_{timestamp}.json"
    path = os.path.join(output_dir, name)

    report = {
        "cycle": round_number,
        "active_version_at_start": active_version_at_start,
        "new_version": new_version,
        "decision": decision,
        "metrics_dev_active": metrics_dev_active,
        "metrics_dev_new": metrics_dev_new,
        "metrics_holdout_active": metrics_holdout_active,
        "metrics_holdout_new": metrics_holdout_new,
        "rollback_counter": rollback_counter,
        "changed_decisions": changed_decisions,
        "next_action": next_action,
        "run_id": run_id,
        "timestamp": timestamp,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    _atomic_write(path, payload, _write_counter)
    return path


def load_report(path: str) -> dict[str, Any]:
    """Restore a per-round report without recomputation."""
    with open(path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


# --------------------------------------------------------------------------- #
#  Final summary
# --------------------------------------------------------------------------- #
def write_summary(
    state: CycleState,
    final_dev_metrics: dict[str, Any],
    final_holdout_metrics: dict[str, Any],
    stop_reason: str,
    run_id: str,
    run_repository: RunRepository,
    output_dir: str,
) -> str:
    """Serialize a final summary to ``report_cycle/summary_{run_id}_{ts}.json``."""
    if stop_reason not in VALID_STOP_REASONS:
        raise CycleOrchestratorError(
            f"invalid stop_reason {stop_reason!r}; expected one of {sorted(VALID_STOP_REASONS)}"
        )
    output_dir = os.path.join(output_dir, "report_cycle")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"summary_{run_id}_{timestamp}.json"
    path = os.path.join(output_dir, name)

    summary = {
        "total_cycles": state.cycle_counter,
        "stop_reason": stop_reason,
        "final_active_version": state.active_version.version,
        "final_dev_metrics": final_dev_metrics,
        "final_holdout_metrics": final_holdout_metrics,
        "accepted_history": state.accepted_history,
        "rollback_history": state.rollback_history,
        "run_id": run_id,
        "timestamp": timestamp,
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    _atomic_write(path, payload, _write_counter)
    return path


# --------------------------------------------------------------------------- #
#  Cycle log (append-only JSONL)
# --------------------------------------------------------------------------- #
def log_event(
    event: str,
    round: int,
    details: dict[str, Any],
    log_path: str,
) -> None:
    """Append one JSON object to the cycle log.

    Format: ``{"timestamp": ..., "round": ..., "event": ..., "details": ...}``
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round": round,
        "event": event,
        "details": details,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.debug("cycle event: %s (round %d)", event, round)


def read_cycle_log(path: str, round: int | None = None) -> list[dict[str, Any]]:
    """Read the JSONL log and return all events, or only events for a round."""
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if round is None or entry.get("round") == round:
                events.append(entry)
    return events
