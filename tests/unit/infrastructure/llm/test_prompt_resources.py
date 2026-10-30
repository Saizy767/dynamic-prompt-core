"""Unit tests for the system prompt resource loader."""

from __future__ import annotations

import pytest

from dynamic_prompt_core.infrastructure.llm.prompts import load_prompt


class TestLoadPrompt:
    def test_classifier_prompt_nonempty(self) -> None:
        text = load_prompt("classifier")
        assert len(text) > 0

    def test_extractor_prompt_nonempty(self) -> None:
        text = load_prompt("extractor")
        assert len(text) > 0

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_prompt("nonexistent")

    def test_classifier_and_extractor_are_distinct(self) -> None:
        assert load_prompt("classifier") != load_prompt("extractor")
