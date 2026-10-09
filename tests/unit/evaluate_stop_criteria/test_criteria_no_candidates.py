"""Unit tests for the candidate exhaustion criterion."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.criteria import (
    _check_no_candidates,
)
from tests.unit.evaluate_stop_criteria.conftest import make_context


def test_no_candidates():
    ctx = make_context(candidate_queue_size=0, new_candidates_found=0)
    assert _check_no_candidates(ctx) == "no_candidates_available"


def test_new_candidates_found():
    ctx = make_context(candidate_queue_size=0, new_candidates_found=3)
    assert _check_no_candidates(ctx) is None


def test_queue_non_empty():
    ctx = make_context(candidate_queue_size=2, new_candidates_found=0)
    assert _check_no_candidates(ctx) is None
