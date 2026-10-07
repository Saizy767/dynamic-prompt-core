"""Tests for the Classification domain model."""
from __future__ import annotations

import dataclasses

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification import Classification
from dynamic_prompt_core.domain.models.judgment import Judgment


def _judgments() -> list[Judgment]:
    return [Judgment(Candidate("a"), 0.2), Judgment(Candidate("b"), 0.8)]


class TestClassificationCreation:
    def test_valid_creation(self) -> None:
        judgments = _judgments()
        classification = Classification(Candidate("b"), judgments)
        assert classification.selected == Candidate("b")
        assert classification.judgments == tuple(judgments)

    def test_judgments_stored_as_tuple(self) -> None:
        classification = Classification(Candidate("b"), _judgments())
        assert isinstance(classification.judgments, tuple)

    def test_selected_preserved(self) -> None:
        classification = Classification(Candidate("b"), _judgments())
        assert classification.selected.value == "b"

    def test_judgments_preserved(self) -> None:
        judgments = _judgments()
        classification = Classification(Candidate("b"), judgments)
        assert classification.judgments[0].candidate == Candidate("a")
        assert classification.judgments[1].candidate == Candidate("b")

    def test_accepts_tuple_directly(self) -> None:
        judgments = tuple(_judgments())
        classification = Classification(Candidate("b"), judgments)
        assert classification.judgments == judgments


class TestClassificationImmutability:
    def test_selected_cannot_be_mutated(self) -> None:
        classification = Classification(Candidate("b"), _judgments())
        with pytest.raises(dataclasses.FrozenInstanceError):
            classification.selected = Candidate("a")  # type: ignore[misc]

    def test_judgments_cannot_be_reassigned(self) -> None:
        classification = Classification(Candidate("b"), _judgments())
        with pytest.raises(dataclasses.FrozenInstanceError):
            classification.judgments = ()  # type: ignore[misc]

    def test_judgments_tuple_rejects_append(self) -> None:
        classification = Classification(Candidate("b"), _judgments())
        with pytest.raises(AttributeError):
            classification.judgments.append(Judgment(Candidate("c"), 0.5))  # type: ignore[attr-defined]

    def test_judgments_tuple_rejects_setitem(self) -> None:
        classification = Classification(Candidate("b"), _judgments())
        with pytest.raises(TypeError):
            classification.judgments[0] = Judgment(Candidate("c"), 0.5)  # type: ignore[index]


class TestClassificationValidation:
    def test_empty_judgments_rejected(self) -> None:
        with pytest.raises(ValueError):
            Classification(Candidate("a"), [])

    def test_selected_not_in_judgments_rejected(self) -> None:
        with pytest.raises(ValueError):
            Classification(Candidate("c"), _judgments())
