"""Unit tests for the evaluate_stop_criteria use case."""
from __future__ import annotations

import json
import os
from typing import Any

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
    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    def save_results(self, results: list[dict[str, Any]], run_id: str, path: str) -> str:
        self.saved_paths.append(path)
        return path

    def load_results(self, path: str) -> list[dict[str, Any]]:
        return []


def _deps(repo: _MockRunRepository | None = None) -> EvaluateStopCriteriaDeps:
    return EvaluateStopCriteriaDeps(run_repository=repo or _MockRunRepository())


async def test_empty_metric_history_continues():
    repo = _MockRunRepository()
    ctx = make_context(metric_history=[], config=StopCriteriaConfig())
    decision = await evaluate_stop_criteria(_deps(repo), ctx, output_dir="/tmp")
    assert decision.should_stop is False
    assert decision.reason is None
    assert repo.saved_paths == []


async def test_stop_writes_artifact(tmp_path):
    repo = _MockRunRepository()
    cfg = StopCriteriaConfig(plateau_window=3)
    ctx = make_context(
        round_counter=4,
        metric_history=metrics(0.50, 0.50, 0.50, 0.50),
        run_id="test-run",
        config=cfg,
    )
    output_dir = str(tmp_path / "results")
    decision = await evaluate_stop_criteria(_deps(repo), ctx, output_dir=output_dir)
    assert decision.should_stop is True
    assert decision.reason is not None

    files = os.listdir(output_dir)
    artifact_files = [f for f in files if f.startswith("stop_decision_")]
    assert len(artifact_files) == 1
    with open(os.path.join(output_dir, artifact_files[0]), encoding="utf-8") as f:
        artifact = json.load(f)
    assert artifact["round"] == 4
    assert artifact["reason"] == decision.reason
    assert "triggered_criteria" in artifact
    assert "metric_snapshot" in artifact
    assert "counters" in artifact


async def test_continue_writes_no_artifact(tmp_path):
    repo = _MockRunRepository()
    cfg = StopCriteriaConfig(plateau_window=3)
    ctx = make_context(
        round_counter=1,
        candidate_queue_size=1,
        metric_history=metrics(0.50, 0.55),
        config=cfg,
    )
    output_dir = str(tmp_path / "results")
    decision = await evaluate_stop_criteria(_deps(repo), ctx, output_dir=output_dir)
    assert decision.should_stop is False
    assert repo.saved_paths == []
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        artifact_files = [f for f in files if f.startswith("stop_decision_")]
        assert artifact_files == []


async def test_decision_fields_populated():
    cfg = StopCriteriaConfig(plateau_window=3)
    ctx = make_context(
        round_counter=5,
        rollback_counter=1,
        candidate_queue_size=2,
        total_teacher_tokens=500,
        metric_history=metrics(0.50, 0.50, 0.50, 0.50),
        run_id="run-abc",
        config=cfg,
    )
    decision = await evaluate_stop_criteria(_deps(), ctx)
    assert decision.round_number == 5
    assert len(decision.metric_snapshot) == 3
    assert decision.counters["rounds"] == 5
    assert decision.counters["rollbacks"] == 1
    assert decision.counters["tokens"] == 500
    assert decision.counters["candidates"] == 2
