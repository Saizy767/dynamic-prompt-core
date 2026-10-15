"""Tests for cycle counter and check_stop."""

from __future__ import annotations

from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import check_stop
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState


def test_check_stop_healthy_state_returns_false():
    """A healthy state should not stop."""
    state = CycleState(cycle_counter=0, any_accepted=False, run_id="test")
    config = CycleConfig(max_cycles=5)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False
    assert reason is None


def test_check_stop_max_cycles_reached():
    """Stop when cycle_counter reaches max_cycles."""
    state = CycleState(cycle_counter=5, run_id="test")
    config = CycleConfig(max_cycles=5)
    should_stop, reason = check_stop(state, config)
    assert should_stop is True
    assert reason == "max_cycles_reached"


def test_check_stop_no_improvement():
    """Stop when no candidate was accepted and cycle_counter > 0."""
    state = CycleState(cycle_counter=1, any_accepted=False, run_id="test")
    config = CycleConfig(max_cycles=5)
    should_stop, reason = check_stop(state, config)
    assert should_stop is True
    assert reason == "no_improvement"


def test_check_stop_does_not_trigger_on_empty_queue_without_rollback():
    """Empty queue alone (cycle_counter=0) should not stop."""
    state = CycleState(cycle_counter=0, any_accepted=False, candidate_queue=[], run_id="test")
    config = CycleConfig(max_cycles=5)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False


def test_cycle_counter_increments():
    """Cycle counter increments by one."""
    state = CycleState(cycle_counter=2, run_id="test")
    state.cycle_counter += 1
    assert state.cycle_counter == 3
