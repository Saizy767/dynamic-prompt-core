"""Tests for round counter, rollback counter, and check_stop."""

from __future__ import annotations

from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import check_stop
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState


def test_check_stop_healthy_state_returns_false():
    """A healthy state should not stop."""
    state = CycleState(round_counter=0, rollback_counter=0, run_id="test")
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=2)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False
    assert reason is None


def test_check_stop_max_rounds_reached():
    """Stop when round_counter reaches max_rounds."""
    state = CycleState(round_counter=5, rollback_counter=0, run_id="test")
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=2)
    should_stop, reason = check_stop(state, config)
    assert should_stop is True
    assert reason == "max_rounds_reached"


def test_check_stop_max_rollbacks_reached():
    """Stop when rollback_counter reaches max_consecutive_rollbacks."""
    state = CycleState(round_counter=1, rollback_counter=2, run_id="test")
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=2)
    should_stop, reason = check_stop(state, config)
    assert should_stop is True
    assert reason == "max_rollbacks_reached"


def test_check_stop_candidate_queue_exhausted():
    """Stop when queue is empty and rollback_counter > 0."""
    state = CycleState(round_counter=1, rollback_counter=1, candidate_queue=[], run_id="test")
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=3)
    should_stop, reason = check_stop(state, config)
    assert should_stop is True
    assert reason == "candidate_queue_exhausted"


def test_check_stop_does_not_trigger_on_empty_queue_without_rollback():
    """Empty queue alone (rollback_counter=0) should not stop."""
    state = CycleState(round_counter=0, rollback_counter=0, candidate_queue=[], run_id="test")
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=2)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False


def test_round_counter_increments():
    """Round counter increments by one."""
    state = CycleState(round_counter=2, run_id="test")
    state.round_counter += 1
    assert state.round_counter == 3


def test_rollback_counter_resets_on_accept():
    """Rollback counter resets to 0 on accept."""
    from dynamic_prompt_core.application.use_cases.compare_versions.comparator import (
        update_rollback_count,
    )

    assert update_rollback_count(2, "accept") == 0


def test_rollback_counter_increments_on_rollback():
    """Rollback counter increments on rollback."""
    from dynamic_prompt_core.application.use_cases.compare_versions.comparator import (
        update_rollback_count,
    )

    assert update_rollback_count(1, "rollback") == 2
