"""Unit tests for per-stage scoring instrumentation."""
from __future__ import annotations

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.instrumentation import (
    COUNTER_FIELDS,
    TIMING_FIELDS,
    ScoringProfile,
    timed_stage,
)


class TestScoringProfileSchema:
    def test_all_timing_fields_present(self) -> None:
        profile = ScoringProfile()
        for name in TIMING_FIELDS:
            assert hasattr(profile, name)
            assert getattr(profile, name) == 0.0

    def test_all_counter_fields_present(self) -> None:
        profile = ScoringProfile()
        for name in COUNTER_FIELDS:
            assert hasattr(profile, name)
            assert getattr(profile, name) == 0

    def test_as_dict_has_all_fields(self) -> None:
        d = ScoringProfile().as_dict()
        assert set(d.keys()) == set(TIMING_FIELDS) | set(COUNTER_FIELDS)


class TestTimedStage:
    def test_records_non_negative_ms(self) -> None:
        profile = ScoringProfile()
        with timed_stage(profile, "eval_ms"):
            pass
        assert profile.eval_ms >= 0.0

    def test_accumulates_across_uses(self) -> None:
        profile = ScoringProfile()
        with timed_stage(profile, "reset_ms"):
            pass
        first = profile.reset_ms
        with timed_stage(profile, "reset_ms"):
            pass
        assert profile.reset_ms >= first

    def test_records_even_on_exception(self) -> None:
        profile = ScoringProfile()
        with pytest.raises(RuntimeError):
            with timed_stage(profile, "eval_ms"):
                raise RuntimeError("boom")
        assert profile.eval_ms >= 0.0
