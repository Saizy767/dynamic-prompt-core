"""Regression: the summary's final_dev_metrics must reflect the final version.

When the last cycle accepts, the accepted version never ran on dev during the
cycle (metrics_dev_new is always {}), so the summary must measure it instead of
falling back to the replaced version's metrics_dev_active.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from dynamic_prompt_core.application.ports.inbound.run_cycle_input import (
    RunCycleInput,
)
from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import run_cycle
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState

RC = "dynamic_prompt_core.application.use_cases.run_cycle.run_cycle"


def _write_report(tmp_path, decision: str) -> str:
    report = {
        "cycle": 1,
        "active_version_at_start": "classify-v0",
        "new_version": "v1",
        "decision": decision,
        "metrics_dev_active": {"accuracy": 0.4},
        "metrics_dev_new": {},
        "metrics_holdout_active": {"accuracy": 0.5},
        "metrics_holdout_new": {"accuracy": 0.9},
        "rollback_counter": 0,
        "changed_decisions": [],
        "next_action": "continue" if decision == "accept" else "rollback",
        "run_id": "test-run",
        "timestamp": "ts",
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def _run_one_cycle(build_mock_deps, tmp_path, decision: str) -> tuple[MagicMock, MagicMock]:
    """Run run_cycle for exactly one (patched) cycle; return write_summary and
    run_active_on_dev mocks."""
    report_path = _write_report(tmp_path, decision)
    state = CycleState(cycle_counter=0, run_id="test-run", latest_report_path=report_path)
    config = CycleConfig(
        max_cycles=3,
        run_id="test-run",
        output_dir=str(tmp_path),
        dump_state_after_each_cycle=False,
    )
    deps = build_mock_deps()

    with (
        patch(f"{RC}.load_initial_state", return_value=state),
        patch(
            f"{RC}.check_stop",
            side_effect=[(False, None), (True, "max_cycles_reached")],
        ),
        patch(f"{RC}.run_cycle_step", AsyncMock(return_value=state)),
        patch(f"{RC}.run_active_on_dev", AsyncMock(return_value="final_dev.jsonl")) as run_dev,
        patch(f"{RC}.compute_dev_metrics", return_value={"accuracy": 0.9}),
        patch(f"{RC}.write_summary", return_value="summary.json") as write_summary,
    ):
        asyncio.run(run_cycle(deps, RunCycleInput(config_path="config.toml", config=config)))

    return write_summary, run_dev


def test_accepting_cycle_measures_final_version_on_dev(
    build_mock_deps, tmp_path
):
    """After an accepting cycle the summary must carry the final version's dev
    metrics, not the replaced version's."""
    write_summary, run_dev = _run_one_cycle(build_mock_deps, tmp_path, "accept")

    assert run_dev.await_count == 1
    final_dev_metrics = write_summary.call_args.kwargs.get(
        "final_dev_metrics", write_summary.call_args.args[1]
    )
    assert final_dev_metrics == {"accuracy": 0.9}


def test_rolling_back_cycle_reuses_report_dev_metrics(build_mock_deps, tmp_path):
    """After a rollback the final version IS the old active, so the report's
    metrics_dev_active is correct and no extra dev run happens."""
    write_summary, run_dev = _run_one_cycle(build_mock_deps, tmp_path, "rollback")

    assert run_dev.await_count == 0
    final_dev_metrics = write_summary.call_args.kwargs.get(
        "final_dev_metrics", write_summary.call_args.args[1]
    )
    assert final_dev_metrics == {"accuracy": 0.4}
