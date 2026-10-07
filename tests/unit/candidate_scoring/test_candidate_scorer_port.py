"""Tests for the CandidateScorer port and duplicate validation helper."""
from __future__ import annotations

import typing

import pytest

from dynamic_prompt_core.application.ports.outbound.candidate_scorer import (
    CandidateScorer,
    validate_unique_candidates,
)
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment


class TestCandidateScorerProtocol:
    def test_is_a_typing_protocol(self) -> None:
        assert typing.is_protocol(CandidateScorer)

    def test_is_runtime_checkable(self) -> None:
        assert getattr(CandidateScorer, "_is_runtime_protocol", False)

    async def test_matching_class_satisfies_isinstance(self) -> None:
        class StubScorer:
            async def score(
                self,
                text: str,
                candidates: list[Candidate],
            ) -> list[Judgment]:
                return [Judgment(c, 0.0) for c in candidates]

        assert isinstance(StubScorer(), CandidateScorer)


class TestValidateUniqueCandidates:
    def test_unique_candidates_accepted(self) -> None:
        validate_unique_candidates([Candidate("a"), Candidate("b"), Candidate("c")])

    def test_single_candidate_accepted(self) -> None:
        validate_unique_candidates([Candidate("a")])

    def test_empty_list_accepted(self) -> None:
        validate_unique_candidates([])

    def test_duplicate_candidates_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_unique_candidates([Candidate("a"), Candidate("a")])

    def test_repeated_identical_candidates_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_unique_candidates(
                [Candidate("a"), Candidate("b"), Candidate("a")]
            )

    def test_duplicate_at_end_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_unique_candidates(
                [Candidate("a"), Candidate("b"), Candidate("b")]
            )
