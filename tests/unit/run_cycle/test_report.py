"""Tests for per-round report, final summary, and cycle log."""

from __future__ import annotations

import json

import pytest

from dynamic_prompt_core.application.use_cases.run_cycle.report import (
    CycleOrchestratorError,
    load_report,
    log_event,
    read_cycle_log,
    write_report,
    write_summary,
)
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState


def test_write_report_produces_valid_json(tmp_path, mock_run_repository):
    """write_report produces a valid JSON file with all fields."""
    path = write_report(
        round_number=1,
        active_version_at_start="classify-v0",
        new_version="classify-v1",
        decision="accept",
        metrics_dev_active={"accuracy": 0.5},
        metrics_dev_new={"accuracy": 0.6},
        metrics_holdout_active={"accuracy": 0.5},
        metrics_holdout_new={"accuracy": 0.6},
        rollback_counter=0,
        changed_decisions=[],
        next_action="continue",
        run_id="test-run",
        run_repository=mock_run_repository,
        output_dir=str(tmp_path),
    )
    assert path.endswith(".json")
    assert "test-run" in path
    assert "round1" in path
    report = json.loads(open(path).read())
    assert report["round"] == 1
    assert report["decision"] == "accept"
    assert report["metrics_dev_new"]["accuracy"] == 0.6


def test_load_report_round_trips(tmp_path, mock_run_repository):
    """load_report reproduces every field written by write_report."""
    path = write_report(
        round_number=2,
        active_version_at_start="v0",
        new_version="v1",
        decision="rollback",
        metrics_dev_active={"accuracy": 0.7},
        metrics_dev_new={"accuracy": 0.6},
        metrics_holdout_active={},
        metrics_holdout_new={},
        rollback_counter=1,
        changed_decisions=[{"id": 1, "direction": "broke"}],
        next_action="rollback",
        run_id="r",
        run_repository=mock_run_repository,
        output_dir=str(tmp_path),
    )
    loaded = load_report(path)
    assert loaded["round"] == 2
    assert loaded["decision"] == "rollback"
    assert loaded["rollback_counter"] == 1
    assert loaded["changed_decisions"] == [{"id": 1, "direction": "broke"}]


def test_write_summary_valid_stop_reason(tmp_path, mock_run_repository):
    """write_summary produces a valid summary with an enumerated stop reason."""
    state = CycleState(round_counter=3, run_id="r")
    path = write_summary(
        state=state,
        final_dev_metrics={"accuracy": 0.8},
        final_holdout_metrics={"accuracy": 0.7},
        stop_reason="max_rounds_reached",
        all_changed_decisions=[],
        run_id="r",
        run_repository=mock_run_repository,
        output_dir=str(tmp_path),
    )
    summary = json.loads(open(path).read())
    assert summary["total_rounds"] == 3
    assert summary["stop_reason"] == "max_rounds_reached"


def test_write_summary_rejects_invalid_stop_reason(tmp_path, mock_run_repository):
    """write_summary rejects an invalid stop reason."""
    state = CycleState(run_id="r")
    with pytest.raises(CycleOrchestratorError):
        write_summary(
            state=state,
            final_dev_metrics={},
            final_holdout_metrics={},
            stop_reason="bogus_reason",
            all_changed_decisions=[],
            run_id="r",
            run_repository=mock_run_repository,
            output_dir=str(tmp_path),
        )


def test_log_event_is_append_only(tmp_path):
    """log_event appends one JSON line per call."""
    log_path = str(tmp_path / "cycle_log.jsonl")
    log_event("round_start", 1, {"active": "v0"}, log_path)
    log_event("round_end", 1, {"decision": "accept"}, log_path)
    lines = open(log_path).read().strip().split("\n")
    assert len(lines) == 2
    entry0 = json.loads(lines[0])
    assert entry0["event"] == "round_start"
    assert entry0["round"] == 1
    assert "timestamp" in entry0
    entry1 = json.loads(lines[1])
    assert entry1["event"] == "round_end"


def test_read_cycle_log_filters_by_round(tmp_path):
    """read_cycle_log returns only events for the given round."""
    log_path = str(tmp_path / "cycle_log.jsonl")
    log_event("round_start", 1, {}, log_path)
    log_event("round_end", 1, {}, log_path)
    log_event("round_start", 2, {}, log_path)
    log_event("round_end", 2, {}, log_path)

    all_events = read_cycle_log(log_path)
    assert len(all_events) == 4

    round1 = read_cycle_log(log_path, round=1)
    assert len(round1) == 2
    assert all(e["round"] == 1 for e in round1)

    round2 = read_cycle_log(log_path, round=2)
    assert len(round2) == 2
    assert all(e["round"] == 2 for e in round2)
