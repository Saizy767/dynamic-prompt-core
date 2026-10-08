"""Tests for the infrastructure scoring seams."""
from __future__ import annotations

import typing

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    CausalLanguageModel,
    SequenceLogits,
)
from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
    ScoringPromptBuilder,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    TokenizerAdapter,
)


class TestScoringPromptBuilder:
    def setup_method(self) -> None:
        self.builder = ScoringPromptBuilder()

    def test_prefix_ends_with_candidate_label(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert prefix.endswith("Candidate: ")

    def test_prefix_contains_input(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "some input" in prefix

    def test_candidate_value_verbatim(self) -> None:
        assert self.builder.build_candidate(Candidate("sports")) == "sports"

    def test_candidate_value_not_stripped(self) -> None:
        assert self.builder.build_candidate(Candidate("  sports  ")) == "  sports  "

    def test_candidate_value_not_lowercased(self) -> None:
        assert self.builder.build_candidate(Candidate("Sports")) == "Sports"

    def test_full_prompt_is_prefix_plus_candidate(self) -> None:
        candidate = Candidate("sports")
        full = self.builder.build("input text", candidate)
        assert full == self.builder.build_prefix("input text") + "sports"

    def test_prefix_contains_semantic_criteria(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "Evaluate how well the candidate" in prefix

    def test_prefix_contains_relevance_criteria(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "content of the text" in prefix

    def test_no_json_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "JSON" not in prefix

    def test_no_confidence_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "confidence" not in prefix

    def test_no_decision_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "decision" not in prefix

    def test_no_reply_with_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "Reply with" not in prefix

    def test_no_json_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "JSON" not in full

    def test_no_confidence_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "confidence" not in full

    def test_no_decision_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "decision" not in full

    def test_no_reply_with_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "Reply with" not in full

    def test_candidate_with_spaces_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("  sports  "))
        assert full.endswith("  sports  ")

    def test_candidate_with_mixed_case_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("Sports"))
        assert full.endswith("Sports")

    def test_candidate_with_punctuation_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("billing / payments"))
        assert full.endswith("billing / payments")

    def test_candidate_with_unicode_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("分类"))
        assert full.endswith("分类")

    def test_prefix_candidate_boundary_maintained(self) -> None:
        candidate = Candidate("sports")
        text = "input text"
        full = self.builder.build(text, candidate)
        assert full == self.builder.build_prefix(text) + self.builder.build_candidate(candidate)

    def test_candidate_is_final_textual_component(self) -> None:
        candidate = Candidate("sports")
        full = self.builder.build("input text", candidate)
        assert full.endswith("sports")
        prefix = self.builder.build_prefix("input text")
        assert prefix.endswith("Candidate: ")
        assert full == prefix + "sports"


class TestJudgmentPromptSemantics:
    """Verify semantic requirements from CLASSIFICATION_PROMPT_V0 are represented."""

    def setup_method(self) -> None:
        self.builder = ScoringPromptBuilder()

    def test_prompt_contains_task_framing_for_candidate_evaluation(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "evaluate" in prefix.lower()
        assert "candidate" in prefix.lower()
        assert "matches" in prefix.lower()

    def test_prompt_contains_relevance_criteria_based_on_content(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "content of the text" in prefix
        assert "hashtags" in prefix

    def test_prompt_contains_metaphorical_handling(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "metaphorically" in prefix

    def test_prompt_contains_ambiguity_handling(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "ambiguous" in prefix

    def test_prompt_contains_factual_match_criteria(self) -> None:
        prefix = self.builder.build_prefix("some input")
        assert "Factual statements" in prefix

    def test_no_json_output_phrase(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "Reply with a single JSON object" not in full

    def test_no_decision_field(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "decision" not in full

    def test_no_confidence_field(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "confidence" not in full

    def test_no_no_markdown_phrase(self) -> None:
        full = self.builder.build("input text", Candidate("sports"))
        assert "No markdown" not in full

    def test_candidate_specific_prompts_are_distinct(self) -> None:
        text = "The user wants to cancel their subscription."
        candidates = [
            Candidate("subscription cancellation"),
            Candidate("password reset"),
            Candidate("billing inquiry"),
        ]
        prompts = [self.builder.build(text, c) for c in candidates]
        assert len(set(prompts)) == 3

    def test_each_prompt_contains_exactly_one_candidate(self) -> None:
        text = "The user wants to cancel their subscription."
        candidates = [Candidate("alpha"), Candidate("beta"), Candidate("gamma")]
        prompts = [self.builder.build(text, c) for c in candidates]
        for i, prompt in enumerate(prompts):
            assert candidates[i].value in prompt
            for j, other in enumerate(candidates):
                if i != j:
                    assert other.value not in prompt


class TestTokenizerAdapterProtocol:
    def test_is_a_typing_protocol(self) -> None:
        assert typing.is_protocol(TokenizerAdapter)

    def test_is_runtime_checkable(self) -> None:
        assert getattr(TokenizerAdapter, "_is_runtime_protocol", False)


class StubTokenizer:
    """Deterministic tokenizer double: maps characters to ord values."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tid) for tid in token_ids)


class TestStubTokenizerRoundTrip:
    def test_encode_decode_round_trip(self) -> None:
        tokenizer = StubTokenizer()
        text = "hello world"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_encode_returns_list_of_ints(self) -> None:
        tokenizer = StubTokenizer()
        result = tokenizer.encode("abc")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_stub_satisfies_protocol(self) -> None:
        assert isinstance(StubTokenizer(), TokenizerAdapter)


class StubModel:
    """Deterministic model double returning fixed logits."""

    def __init__(self, logits: SequenceLogits) -> None:
        self._logits = logits
        self.forward_calls: list[list[int]] = []

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        self.forward_calls.append(list(input_ids))
        return self._logits


class TestCausalLanguageModelProtocol:
    def test_is_a_typing_protocol(self) -> None:
        assert typing.is_protocol(CausalLanguageModel)

    def test_is_runtime_checkable(self) -> None:
        assert getattr(CausalLanguageModel, "_is_runtime_protocol", False)

    def test_stub_satisfies_protocol(self) -> None:
        logits = SequenceLogits(logits=((1.0, 2.0), (3.0, 4.0)))
        assert isinstance(StubModel(logits), CausalLanguageModel)

    def test_stub_returns_deterministic_logits(self) -> None:
        logits = SequenceLogits(logits=((1.0, 2.0), (3.0, 4.0)))
        model = StubModel(logits)
        result1 = model.forward([1, 2])
        result2 = model.forward([1, 2])
        assert result1.logits == result2.logits


class TestLLMScoringError:
    def test_is_exception_subclass(self) -> None:
        assert issubclass(LLMScoringError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LLMScoringError, match="test error"):
            raise LLMScoringError("test error")
