"""Stop-decision artifact I/O for the evaluate_stop_criteria use case.

Writes a single JSON file ``stop_decision_{run_id}_{timestamp}.json`` when the
decision is ``stop``.  The artifact is self-contained and analyzable without
recomputation.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.application.ports.outbound.run_repository import (
    RunRepository,
)
from dynamic_prompt_core.application.ports.outbound.stop_criteria import (
    StopDecision,
)

log = logging.getLogger(__name__)

__all__ = ["write_stop_decision_artifact"]


def write_stop_decision_artifact(
    decision: StopDecision,
    run_repository: RunRepository,
    output_dir: str,
) -> str:
    """Serialize a stop decision to ``stop_decision_{run_id}_{ts}.json``.

    Returns the path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = decision.run_id
    name = f"stop_decision_{run_id}_{timestamp}.json" if run_id else f"stop_decision_{timestamp}.json"
    path = os.path.join(output_dir, name)

    artifact: dict[str, Any] = {
        "round": decision.round_number,
        "reason": decision.reason,
        "triggered_criteria": decision.triggered_criteria,
        "metric_snapshot": decision.metric_snapshot,
        "counters": decision.counters,
        "run_id": run_id,
        "timestamp": timestamp,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    log.debug("stop_decision artifact written to %s", path)
    return path
