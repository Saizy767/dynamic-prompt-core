"""Tests for the candidate queue behavior."""

from __future__ import annotations

from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState


def test_queue_consumed_in_order():
    """Candidates are consumed in FIFO order."""
    state = CycleState(run_id="test")
    state.candidate_queue = [
        {"cluster_id": 1, "rank": 0},
        {"cluster_id": 2, "rank": 1},
        {"cluster_id": 3, "rank": 2},
    ]
    first = state.candidate_queue.pop(0)
    assert first["cluster_id"] == 1
    second = state.candidate_queue.pop(0)
    assert second["cluster_id"] == 2
    third = state.candidate_queue.pop(0)
    assert third["cluster_id"] == 3


def test_queue_refilled_replaces_contents():
    """Refilling the queue replaces, not appends."""
    state = CycleState(run_id="test")
    state.candidate_queue = [{"cluster_id": 99}]
    new_candidates = [{"cluster_id": 1}, {"cluster_id": 2}]
    state.candidate_queue = list(new_candidates)
    assert len(state.candidate_queue) == 2
    assert state.candidate_queue[0]["cluster_id"] == 1


def test_queue_exhausted_on_rollback():
    """When the queue is empty and a rollback occurs, the cycle should stop."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import check_stop

    state = CycleState(round_counter=1, rollback_counter=1, candidate_queue=[], run_id="test")
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=3)
    should_stop, reason = check_stop(state, config)
    assert should_stop is True
    assert reason == "candidate_queue_exhausted"


def test_queue_not_exhausted_when_has_candidates():
    """When the queue has candidates, the cycle should not stop."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import check_stop

    state = CycleState(
        round_counter=1,
        rollback_counter=1,
        candidate_queue=[{"cluster_id": 1}],
        run_id="test",
    )
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=3)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False
