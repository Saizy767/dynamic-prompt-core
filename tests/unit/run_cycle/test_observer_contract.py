"""Test the cycle->observer artifact contract.

The observer must discover and read reports that write_report actually
produces.  This guards against filename-pattern or report-key drift.
"""

from __future__ import annotations

from dynamic_prompt_core.application.use_cases.observe_cycle.observer import (
    CycleObserverConfig,
    CycleObserverError,
    build_metric_trends,
    build_round_summaries,
    discover_artifacts,
    load_cycle_artifacts,
    resolve_run_id,
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


def _nested_metrics(macro: float, minority: float, failed: int) -> dict:
    """The shape compute_metrics actually writes (f1/counts sub-dicts)."""
    return {
        "accuracy": 0.5,
        "f1": {"macro": macro, "minority": minority},
        "counts": {"total": 50, "ok": 50 - failed, "failed": failed},
    }


def test_observer_reads_nested_metrics_and_holdout_fallback(tmp_path, mock_run_repository):
    """With metrics_dev_new empty (as the pipeline writes) and nested
    compute_metrics shape, trend and summary cells must show numbers, not N/A."""
    run_id = "nested-run"
    output_dir = str(tmp_path)

    write_report(
        round_number=1,
        active_version_at_start="classify-v0",
        new_version="v1",
        decision="accept",
        metrics_dev_active=_nested_metrics(0.30, 0.60, 0),
        metrics_dev_new={},
        metrics_holdout_active={},
        metrics_holdout_new=_nested_metrics(0.40, 0.70, 2),
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
        final_dev_metrics=_nested_metrics(0.40, 0.70, 2),
        final_holdout_metrics={},
        stop_reason="max_cycles_reached",
        run_id=run_id,
        run_repository=mock_run_repository,
        output_dir=output_dir,
    )

    config = CycleObserverConfig(output_dir=output_dir)
    bundle = load_cycle_artifacts(run_id, config)

    trends = build_metric_trends(bundle, config)
    assert len(trends.rows) == 1
    row = trends.rows[0]
    assert row.macro_f1 == 0.40
    assert row.minority_f1 == 0.70
    assert row.parse_failures == 2
    assert row.holdout_macro_f1 == 0.40

    summaries = build_round_summaries(bundle, config)
    assert summaries[0].metrics_dev_after["f1"]["macro"] == 0.40
    assert summaries[0].minority_f1_change == 0.70 - 0.60


def test_resolve_run_id_latest_picks_newest_summary(tmp_path):
    """--run-id latest must resolve to the run_id of the newest summary file."""

    summary_dir = tmp_path / "report_cycle"
    summary_dir.mkdir()
    for name in (
        "summary_older-run_20261027T000000Z.json",
        "summary_newer-run_20261028T000000Z.json",
    ):
        (summary_dir / name).touch()

    config = CycleObserverConfig(output_dir=str(tmp_path))
    assert resolve_run_id("latest", config) == "newer-run"
    assert resolve_run_id("explicit-run", config) == "explicit-run"

    empty_config = CycleObserverConfig(output_dir=str(tmp_path / "nowhere"))
    try:
        resolve_run_id("latest", empty_config)
    except CycleObserverError as exc:
        assert "latest" in str(exc)
    else:
        raise AssertionError("expected CycleObserverError for no summaries")
