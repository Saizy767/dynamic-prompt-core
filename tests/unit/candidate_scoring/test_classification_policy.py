"""Tests for the ClassificationPolicy protocol and ArgmaxClassificationPolicy."""
from __future__ import annotations

import typing

import pytest

from dynamic_prompt_core.application.services.classification_policy import (
    ArgmaxClassificationPolicy,
    ClassificationPolicy,
)
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment


def _judgments(*pairs: tuple[str, float]) -> list[Judgment]:
    return [Judgment(Candidate(name), score) for name, score in pairs]


class TestClassificationPolicyProtocol:
    def test_is_a_typing_protocol(self) -> None:
        assert typing.is_protocol(ClassificationPolicy)

    def test_is_runtime_checkable(self) -> None:
        assert getattr(ClassificationPolicy, "_is_runtime_protocol", False)

    def test_argmax_satisfies_isinstance(self) -> None:
        assert isinstance(ArgmaxClassificationPolicy(), ClassificationPolicy)

    def test_module_does_not_import_infrastructure(self) -> None:
        import ast

        import dynamic_prompt_core.application.services.classification_policy as mod

        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "dynamic_prompt_core.infrastructure"
                    ), f"module imports infrastructure: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith(
                    "dynamic_prompt_core.infrastructure"
                ), f"module imports from infrastructure: {node.module}"


class TestArgmaxClassificationPolicy:
    def setup_method(self) -> None:
        self.policy = ArgmaxClassificationPolicy()

    def test_selects_highest_score(self) -> None:
        judgments = _judgments(("a", 0.2), ("b", 0.8), ("c", 0.5))
        result = self.policy.classify(judgments)
        assert result.selected == Candidate("b")

    def test_single_candidate(self) -> None:
        judgments = _judgments(("a", 0.5))
        result = self.policy.classify(judgments)
        assert result.selected == Candidate("a")

    def test_tie_returns_first_in_order(self) -> None:
        judgments = _judgments(("a", 0.8), ("b", 0.8), ("c", 0.5))
        result = self.policy.classify(judgments)
        assert result.selected == Candidate("a")

    def test_tie_at_end_returns_first_in_order(self) -> None:
        judgments = _judgments(("a", 0.5), ("b", 0.8), ("c", 0.8))
        result = self.policy.classify(judgments)
        assert result.selected == Candidate("b")

    def test_empty_judgments_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            self.policy.classify([])

    def test_judgments_preserved_as_tuple_in_input_order(self) -> None:
        judgments = _judgments(("a", 0.2), ("b", 0.8), ("c", 0.5))
        result = self.policy.classify(judgments)
        assert isinstance(result.judgments, tuple)
        assert [j.candidate.value for j in result.judgments] == ["a", "b", "c"]

    def test_does_not_calibrate_or_normalize(self) -> None:
        judgments = _judgments(("a", -3.2), ("b", 42.0))
        result = self.policy.classify(judgments)
        assert result.selected == Candidate("b")
        assert result.judgments[0].score == -3.2
        assert result.judgments[1].score == 42.0
