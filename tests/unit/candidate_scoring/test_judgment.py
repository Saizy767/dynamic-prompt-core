"""Tests for the Judgment domain model."""
from __future__ import annotations

import dataclasses
import math

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment


class TestJudgmentAssociation:
    def test_associates_candidate_and_score(self) -> None:
        candidate = Candidate("a")
        judgment = Judgment(candidate, 0.5)
        assert judgment.candidate == candidate
        assert judgment.score == 0.5

    def test_candidate_accessible(self) -> None:
        judgment = Judgment(Candidate("x"), 1.0)
        assert judgment.candidate.value == "x"


class TestJudgmentEquality:
    def test_equal_when_candidate_and_score_match(self) -> None:
        assert Judgment(Candidate("a"), 0.5) == Judgment(Candidate("a"), 0.5)

    def test_unequal_when_candidate_differs(self) -> None:
        assert Judgment(Candidate("a"), 0.5) != Judgment(Candidate("b"), 0.5)

    def test_unequal_when_score_differs(self) -> None:
        assert Judgment(Candidate("a"), 0.5) != Judgment(Candidate("a"), 0.6)


class TestJudgmentImmutability:
    def test_candidate_cannot_be_mutated(self) -> None:
        judgment = Judgment(Candidate("a"), 0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            judgment.candidate = Candidate("b")  # type: ignore[misc]

    def test_score_cannot_be_mutated(self) -> None:
        judgment = Judgment(Candidate("a"), 0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            judgment.score = 0.9  # type: ignore[misc]


class TestScoreValidity:
    def test_finite_score_within_unit_interval_accepted(self) -> None:
        judgment = Judgment(Candidate("a"), 0.0)
        assert judgment.score == 0.0

    def test_negative_score_outside_unit_interval_accepted(self) -> None:
        judgment = Judgment(Candidate("a"), -3.2)
        assert judgment.score == -3.2

    def test_large_positive_score_outside_unit_interval_accepted(self) -> None:
        judgment = Judgment(Candidate("a"), 42.0)
        assert judgment.score == 42.0

    def test_nan_score_rejected(self) -> None:
        with pytest.raises(ValueError):
            Judgment(Candidate("a"), float("nan"))

    def test_positive_infinity_rejected(self) -> None:
        with pytest.raises(ValueError):
            Judgment(Candidate("a"), math.inf)

    def test_negative_infinity_rejected(self) -> None:
        with pytest.raises(ValueError):
            Judgment(Candidate("a"), -math.inf)

    def test_out_of_range_score_not_clamped(self) -> None:
        judgment = Judgment(Candidate("a"), 42.0)
        assert judgment.score == 42.0

    def test_negative_score_not_clamped(self) -> None:
        judgment = Judgment(Candidate("a"), -3.2)
        assert judgment.score == -3.2
