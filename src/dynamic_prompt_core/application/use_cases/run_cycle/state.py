"""Cycle state: the mutable snapshot threaded across rounds.

``CycleState`` holds everything the orchestrator needs to resume an interrupted
cycle: the round counter, the active prompt version, the thesis collection and
cluster artifact paths, the candidate queue, the rollback counter, and the
accepted/rollback histories.  ``dump_state`` / ``load_state`` serialize to a
single JSON file per round via ``run_repository``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.application.ports.outbound.run_repository import (
    RunRepository,
)
from dynamic_prompt_core.domain.prompts.base import PromptArtifact
from dynamic_prompt_core.domain.prompts.fixed import CLASSIFICATION_PROMPT_V0


@dataclass
class CycleState:
    """Mutable cycle state threaded across rounds."""

    round_counter: int = 0
    active_version: PromptArtifact = field(default_factory=lambda: CLASSIFICATION_PROMPT_V0)
    active_version_path: str = ""
    thesis_bank_path: str = ""
    clusters_path: str = ""
    candidate_queue: list[dict[str, Any]] = field(default_factory=list)
    rollback_counter: int = 0
    latest_report_path: str = ""
    accepted_history: list[str] = field(default_factory=list)
    rollback_history: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = ""
    last_round_new_candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        d = asdict(self)
        d["active_version"] = {
            "version": self.active_version.version,
            "text": self.active_version.text,
            "sha256": self.active_version.sha256,
        }
        return d


def dump_state(
    state: CycleState,
    run_repository: RunRepository,
    output_dir: str,
) -> str:
    """Serialize ``state`` to ``state_{run_id}_{timestamp}.json``.

    Returns the path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"state_{state.run_id}_{timestamp}.json"
    path = os.path.join(output_dir, name)
    payload = [state.to_dict()]
    return run_repository.save_results(payload, state.run_id, path)


def load_state(path: str) -> CycleState:
    """Restore a ``CycleState`` from a dumped JSON file.

    The dump is a single-element list written by ``dump_state`` via
    ``run_repository.save_results`` (JSONL format).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    version_dict = data.get("active_version", {})
    active_version = PromptArtifact(
        version=version_dict.get("version", "classify-v0"),
        layers=None,
        text=version_dict.get("text", CLASSIFICATION_PROMPT_V0.text),
        sha256=version_dict.get("sha256", ""),
    )
    return CycleState(
        round_counter=int(data.get("round_counter", 0)),
        active_version=active_version,
        active_version_path=data.get("active_version_path", ""),
        thesis_bank_path=data.get("thesis_bank_path", ""),
        clusters_path=data.get("clusters_path", ""),
        candidate_queue=list(data.get("candidate_queue", [])),
        rollback_counter=int(data.get("rollback_counter", 0)),
        latest_report_path=data.get("latest_report_path", ""),
        accepted_history=list(data.get("accepted_history", [])),
        rollback_history=list(data.get("rollback_history", [])),
        run_id=data.get("run_id", ""),
    )


def update_in_prompt_flags(
    thesis_bank_path: str,
    accepted_cluster_ids: set[str],
) -> None:
    """Set ``in_prompt`` on theses in the thesis bank and persist.

    Theses whose ``cluster_id`` is in ``accepted_cluster_ids`` get
    ``in_prompt=true``; all others get ``false``.
    """
    with open(thesis_bank_path, encoding="utf-8") as f:
        bank = json.load(f)
    theses = bank.get("theses", {})
    for _norm, entry in theses.items():
        cid = str(entry.get("cluster_id", ""))
        entry["in_prompt"] = cid in accepted_cluster_ids
    with open(thesis_bank_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
