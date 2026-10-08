"""Tests for the nine-step round sequence and step functions."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState
from dynamic_prompt_core.domain.prompts.fixed import CLASSIFICATION_PROMPT_V0

STEPS = "dynamic_prompt_core.application.use_cases.run_cycle.steps"


def test_compute_dev_metrics_returns_expected_keys(tmp_path):
    """Step 2: compute_dev_metrics returns accuracy and f1."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        compute_dev_metrics,
    )

    results_path = tmp_path / "results_test_v0_dev_20260101T000000Z.jsonl"
    rows = [
        {
            "id": 1,
            "text": "a",
            "true_label": 1,
            "predicted_decision": 1,
            "classify_status": "ok",
            "theses_norm": ["t1"],
        },
        {
            "id": 2,
            "text": "b",
            "true_label": 0,
            "predicted_decision": 0,
            "classify_status": "ok",
            "theses_norm": ["t2"],
        },
    ]
    with open(results_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg_path = "dynamic_prompt_core.application.services.metrics"
    with patch(f"{cfg_path}.metrics.MetricsConfig.from_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            output_dir=str(tmp_path),
            bias_threshold=0.15,
            group_short_max=10,
            group_long_min=30,
        )
        metrics = compute_dev_metrics(str(results_path), "config.toml")

    assert "accuracy" in metrics
    assert "f1" in metrics


def test_cycle_state_round_counter_starts_at_zero():
    """CycleState starts with round_counter=0."""
    state = CycleState(run_id="test")
    assert state.round_counter == 0
    assert state.rollback_counter == 0
    assert state.candidate_queue == []


def test_cycle_state_active_version_defaults_to_v0():
    """CycleState defaults to CLASSIFICATION_PROMPT_V0."""
    state = CycleState(run_id="test")
    assert state.active_version.version == CLASSIFICATION_PROMPT_V0.version


def test_run_round_executes_steps_and_writes_report(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """run_round chains all nine steps and writes a report."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_round

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection", return_value=("thesis.json", "clusters.json")
            )
        )
        stack.enter_context(
            patch(f"{STEPS}.select_candidates", return_value=("cands.json", [{"cluster_id": 1}]))
        )
        stack.enter_context(
            patch(
                f"{STEPS}.compose_new_version",
                new_callable=AsyncMock,
                return_value=(CLASSIFICATION_PROMPT_V0, "prompt.json", [1]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.run_new_on_dev", new_callable=AsyncMock, return_value="new_results.jsonl"
            )
        )
        stack.enter_context(
            patch(f"{STEPS}.decide_and_update", return_value=("accept", "improved", []))
        )
        stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))

        updated = asyncio.run(run_round(state, deps, cycle_config, "config.toml", log_path))

    assert updated.latest_report_path == "report.json"
