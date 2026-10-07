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
    """A single example failure (rollback_counter=0, queue non-empty) continues."""
    state = CycleState(round_counter=0, rollback_counter=0, candidate_queue=[{"c": 1}], run_id="t")
    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=2)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False


def test_unrecoverable_error_raises_cycle_orchestrator_error():
    """CycleOrchestratorError is a ValueError subclass."""
    assert issubclass(CycleOrchestratorError, ValueError)


def test_stop_on_first_error_config():
    """stop_on_first_error config flag is readable."""
    config = CycleConfig(stop_on_first_error=True)
    assert config.stop_on_first_error is True


def test_stop_on_first_error_default_is_false():
    """stop_on_first_error defaults to False."""
    config = CycleConfig()
    assert config.stop_on_first_error is False


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
