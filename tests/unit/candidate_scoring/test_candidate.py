"""Tests for the Candidate domain model."""
from __future__ import annotations

import dataclasses

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate


class TestCandidateCreation:
    def test_valid_candidate_creation(self) -> None:
        candidate = Candidate("positive")
        assert candidate.value == "positive"

    def test_empty_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            Candidate("")

    def test_blank_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            Candidate("   ")


class TestCandidateEquality:
    def test_equal_candidates_same_value(self) -> None:
        assert Candidate("a") == Candidate("a")

    def test_unequal_candidates_different_value(self) -> None:
        assert Candidate("a") != Candidate("b")

    def test_hash_equal_for_equal_candidates(self) -> None:
        assert hash(Candidate("a")) == hash(Candidate("a"))

    def test_can_be_used_as_dict_key(self) -> None:
        mapping = {Candidate("a"): 1, Candidate("b"): 2}
        assert mapping[Candidate("a")] == 1
        assert mapping[Candidate("b")] == 2

    def test_can_be_used_in_set(self) -> None:
        assert len({Candidate("a"), Candidate("a"), Candidate("b")}) == 2


class TestCandidateImmutability:
    def test_value_cannot_be_mutated(self) -> None:
        candidate = Candidate("a")
        with pytest.raises(dataclasses.FrozenInstanceError):
            candidate.value = "b"  # type: ignore[misc]


class TestCandidateNoAutoNormalization:
    def test_value_stored_verbatim_with_whitespace(self) -> None:
        assert Candidate("  X  ").value == "  X  "

    def test_no_implicit_stripping(self) -> None:
        assert Candidate("  x  ").value == "  x  "

    def test_no_implicit_case_folding(self) -> None:
        assert Candidate("A") != Candidate("a")

    def test_no_implicit_casefold(self) -> None:
        assert Candidate("Hello") != Candidate("hello")
