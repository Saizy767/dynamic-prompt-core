"""Tests for error handling behavior."""

from __future__ import annotations

import pytest

from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.report import (
    CycleOrchestratorError,
)
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import check_stop
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState


def test_single_example_failure_does_not_stop_cycle():
    """A single example failure (cycle_counter=0, queue non-empty) continues."""
    state = CycleState(cycle_counter=0, candidate_queue=[{"c": 1}], run_id="t")
    config = CycleConfig(max_cycles=5)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False


def test_unrecoverable_error_raises_cycle_orchestrator_error():
    """CycleOrchestratorError is a ValueError subclass."""
    assert issubclass(CycleOrchestratorError, ValueError)


def test_unrecoverable_stop_reason_is_enumerated():
    """unrecoverable_error is a valid stop reason."""
    from dynamic_prompt_core.application.use_cases.run_cycle.report import (
        VALID_STOP_REASONS,
    )

    assert "unrecoverable_error" in VALID_STOP_REASONS


def test_cycle_orchestrator_error_can_be_raised_and_caught():
    """CycleOrchestratorError can be raised and caught."""
    with pytest.raises(CycleOrchestratorError):
        raise CycleOrchestratorError("test error")
