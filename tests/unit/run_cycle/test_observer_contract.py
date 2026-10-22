"""Test the cycle->observer artifact contract.

The observer must discover and read reports that write_report actually
produces.  This guards against filename-pattern or report-key drift.
"""

from __future__ import annotations

from dynamic_prompt_core.application.use_cases.observe_cycle.observer import (
    CycleObserverConfig,
    build_round_summaries,
    discover_artifacts,
    load_cycle_artifacts,
)
from dynamic_prompt_core.application.use_cases.run_cycle.report import (
    write_report,
    write_summary,
)
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState


def test_observer_finds_report_and_reads_cycle_number(tmp_path, mock_run_repository):
    """discover_artifacts finds a write_report file; build_round_summaries
    reads the 'cycle' key as the round number."""
    run_id = "test-run"
    output_dir = str(tmp_path)

    write_report(
        round_number=1,
        active_version_at_start="classify-v0",
        new_version="v1",
        decision="accept",
        metrics_dev_active={"accuracy": 0.5},
        metrics_dev_new={"accuracy": 0.6},
        metrics_holdout_active={},
        metrics_holdout_new={},
        rollback_counter=0,
        changed_decisions=[],
        next_action="continue",
        run_id=run_id,
        run_repository=mock_run_repository,
        output_dir=output_dir,
    )

    state = CycleState(cycle_counter=1, run_id=run_id)
    write_summary(
        state=state,
        final_dev_metrics={"accuracy": 0.6},
        final_holdout_metrics={},
        stop_reason="max_cycles_reached",
        all_changed_decisions=[],
        run_id=run_id,
        run_repository=mock_run_repository,
        output_dir=output_dir,
    )

    config = CycleObserverConfig(output_dir=output_dir)
    discovered = discover_artifacts(run_id, config)
    assert len(discovered["reports"]) == 1, discovered["reports"]

    bundle = load_cycle_artifacts(run_id, config)
    summaries = build_round_summaries(bundle, config)
    assert len(summaries) == 1
    assert summaries[0].round_number == 1
