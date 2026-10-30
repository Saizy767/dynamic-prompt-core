"""Unit tests for the ClassificationRules domain model."""

from __future__ import annotations

import dataclasses

import pytest

from dynamic_prompt_core.domain.models.classification_rules import (
    ClassificationRules,
    classification_rules_from_artifact,
)
from dynamic_prompt_core.domain.prompts.base import (
    PromptArtifact,
    PromptLayer,
)


class TestClassificationRules:
    def test_immutable(self) -> None:
        rules = ClassificationRules(rules=("a", "b"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            rules.rules = ("c",)  # type: ignore[misc]

    def test_empty_rules_are_valid(self) -> None:
        rules = ClassificationRules(rules=())
        assert len(rules.rules) == 0

    def test_order_preserved(self) -> None:
        rules = ClassificationRules(rules=("rule C", "rule A", "rule B"))
        assert rules.rules == ("rule C", "rule A", "rule B")

    def test_text_preserved(self) -> None:
        rules = ClassificationRules(rules=("  spaced  ", "MixedCase"))
        assert rules.rules == ("  spaced  ", "MixedCase")


class TestClassificationRulesFromArtifact:
    def _make_artifact(self, layers: PromptLayer | None) -> PromptArtifact:
        return PromptArtifact(version="test", layers=layers, text="test")

    def test_valid_rules(self) -> None:
        layers = PromptLayer(
            role="r", task="t", rules=["a", "b"], output_contract="o", fallback="f"
        )
        rules = classification_rules_from_artifact(self._make_artifact(layers))
        assert rules.rules == ("a", "b")

    def test_none_layers_produces_empty(self) -> None:
        rules = classification_rules_from_artifact(self._make_artifact(None))
        assert rules.rules == ()

    def test_empty_rules_list_produces_empty(self) -> None:
        layers = PromptLayer(role="r", task="t", rules=[], output_contract="o", fallback="f")
        rules = classification_rules_from_artifact(self._make_artifact(layers))
        assert rules.rules == ()

    def test_non_string_element_raises(self) -> None:
        layers = PromptLayer(
            role="r",
            task="t",
            rules=["a", 42],
            output_contract="o",
            fallback="f",  # type: ignore[list-item]
        )
        with pytest.raises(TypeError):
            classification_rules_from_artifact(self._make_artifact(layers))

    def test_order_and_text_preserved(self) -> None:
        layers = PromptLayer(
            role="r",
            task="t",
            rules=["rule C", "rule A", "rule B"],
            output_contract="o",
            fallback="f",
        )
        rules = classification_rules_from_artifact(self._make_artifact(layers))
        assert rules.rules == ("rule C", "rule A", "rule B")

    def test_empty_rules_bypass_35_constraint(self) -> None:
        rules = classification_rules_from_artifact(self._make_artifact(None))
        assert len(rules.rules) == 0
