"""Tests for the infrastructure scoring seams."""
from __future__ import annotations

import typing
from collections.abc import Sequence

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.prompts import CLASSIFICATION_PROMPT_V0
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

_DEFAULT_RULES: Sequence[str] = ["rule a", "rule b", "rule c"]


class TestScoringPromptBuilder:
    def setup_method(self) -> None:
        self.builder = ScoringPromptBuilder()

    def test_prefix_ends_with_candidate_label(self) -> None:
        prefix = self.builder.build_prefix("some input", _DEFAULT_RULES)
        assert prefix.endswith("Candidate: ")

    def test_prefix_contains_input(self) -> None:
        prefix = self.builder.build_prefix("some input", _DEFAULT_RULES)
        assert "some input" in prefix

    def test_candidate_value_verbatim(self) -> None:
        assert self.builder.build_candidate(Candidate("sports")) == "sports"

    def test_candidate_value_not_stripped(self) -> None:
        assert self.builder.build_candidate(Candidate("  sports  ")) == "  sports  "

    def test_candidate_value_not_lowercased(self) -> None:
        assert self.builder.build_candidate(Candidate("Sports")) == "Sports"

    def test_full_prompt_is_prefix_plus_candidate(self) -> None:
        candidate = Candidate("sports")
        full = self.builder.build("input text", candidate, _DEFAULT_RULES)
        assert full == self.builder.build_prefix("input text", _DEFAULT_RULES) + "sports"

    def test_prefix_contains_semantic_criteria(self) -> None:
        prefix = self.builder.build_prefix("some input", _DEFAULT_RULES)
        assert "Evaluate how well the candidate" in prefix

    def test_no_json_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input", _DEFAULT_RULES)
        assert "JSON" not in prefix

    def test_no_confidence_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input", _DEFAULT_RULES)
        assert "confidence" not in prefix

    def test_no_decision_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input", _DEFAULT_RULES)
        assert "decision" not in prefix

    def test_no_reply_with_instruction_in_prefix(self) -> None:
        prefix = self.builder.build_prefix("some input", _DEFAULT_RULES)
        assert "Reply with" not in prefix

    def test_no_json_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"), _DEFAULT_RULES)
        assert "JSON" not in full

    def test_no_confidence_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"), _DEFAULT_RULES)
        assert "confidence" not in full

    def test_no_decision_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"), _DEFAULT_RULES)
        assert "decision" not in full

    def test_no_reply_with_instruction_in_full_prompt(self) -> None:
        full = self.builder.build("input text", Candidate("sports"), _DEFAULT_RULES)
        assert "Reply with" not in full

    def test_candidate_with_spaces_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("  sports  "), _DEFAULT_RULES)
        assert full.endswith("  sports  ")

    def test_candidate_with_mixed_case_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("Sports"), _DEFAULT_RULES)
        assert full.endswith("Sports")

    def test_candidate_with_punctuation_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("billing / payments"), _DEFAULT_RULES)
        assert full.endswith("billing / payments")

    def test_candidate_with_unicode_preserved_in_build(self) -> None:
        full = self.builder.build("input text", Candidate("分类"), _DEFAULT_RULES)
        assert full.endswith("分类")

    def test_prefix_candidate_boundary_maintained(self) -> None:
        candidate = Candidate("sports")
        text = "input text"
        full = self.builder.build(text, candidate, _DEFAULT_RULES)
        assert (
            full
            == self.builder.build_prefix(text, _DEFAULT_RULES)
            + self.builder.build_candidate(candidate)
        )

    def test_candidate_is_final_textual_component(self) -> None:
        candidate = Candidate("sports")
        full = self.builder.build("input text", candidate, _DEFAULT_RULES)
        assert full.endswith("sports")
        prefix = self.builder.build_prefix("input text", _DEFAULT_RULES)
        assert prefix.endswith("Candidate: ")
        assert full == prefix + "sports"


class TestScoringPromptBuilderRules:
    """Unit tests for active-rule incorporation in ScoringPromptBuilder."""

    def setup_method(self) -> None:
        self.builder = ScoringPromptBuilder()

    def test_all_supplied_rules_appear_in_prompt(self) -> None:
        rules = ["first rule", "second rule", "third rule"]
        prefix = self.builder.build_prefix("text", rules)
        for rule in rules:
            assert rule in prefix

    def test_rule_order_preserved(self) -> None:
        rules = ["alpha", "beta", "gamma"]
        prefix = self.builder.build_prefix("text", rules)
        pos_alpha = prefix.index("alpha")
        pos_beta = prefix.index("beta")
        pos_gamma = prefix.index("gamma")
        assert pos_alpha < pos_beta < pos_gamma

    def test_rules_appear_before_input_text_and_candidate(self) -> None:
        rules = ["my rule"]
        text = "zzz_unique_input_zzz"
        candidate = Candidate("zzz_unique_candidate_zzz")
        full = self.builder.build(text, candidate, rules)
        pos_rule = full.index("my rule")
        pos_text = full.index(text)
        pos_candidate = full.index(candidate.value)
        assert pos_rule < pos_text < pos_candidate

    def test_rules_appear_after_fixed_instructions(self) -> None:
        rules = ["my rule"]
        prefix = self.builder.build_prefix("text", rules)
        pos_instructions = prefix.index("Evaluate how well the candidate")
        pos_rule = prefix.index("my rule")
        assert pos_instructions < pos_rule

    def test_unicode_rules_handled(self) -> None:
        rules = ["правило один", "規則二", "分类规则"]
        prefix = self.builder.build_prefix("text", rules)
        for rule in rules:
            assert rule in prefix

    def test_multiline_rules_handled(self) -> None:
        rules = ["first line\nsecond line", "another rule"]
        prefix = self.builder.build_prefix("text", rules)
        assert "first line\nsecond line" in prefix
        assert "another rule" in prefix

    def test_empty_rules_do_not_produce_default_rules(self) -> None:
        prefix_empty = self.builder.build_prefix("text", [])
        prefix_with_rules = self.builder.build_prefix("text", ["some rule"])
        assert "Classification rules:" not in prefix_empty
        assert "some rule" not in prefix_empty
        assert "Classification rules:" in prefix_with_rules

    def test_identical_inputs_produce_identical_output(self) -> None:
        rules = ["rule a", "rule b"]
        text = "same text"
        prefix1 = self.builder.build_prefix(text, rules)
        prefix2 = self.builder.build_prefix(text, rules)
        assert prefix1 == prefix2

    def test_different_rules_produce_different_prompts(self) -> None:
        prefix_a = self.builder.build_prefix("text", ["rule a"])
        prefix_b = self.builder.build_prefix("text", ["rule b"])
        assert prefix_a != prefix_b

    def test_rules_formatted_as_bullets(self) -> None:
        rules = ["rule one", "rule two"]
        prefix = self.builder.build_prefix("text", rules)
        assert "- rule one" in prefix
        assert "- rule two" in prefix


class TestJudgmentPromptSemantics:
    """Verify the scoring prompt structure with active rules."""

    def setup_method(self) -> None:
        self.builder = ScoringPromptBuilder()
        layers = CLASSIFICATION_PROMPT_V0.layers
        self.rules = list(layers.rules) if layers else []

    def test_prompt_contains_task_framing_for_candidate_evaluation(self) -> None:
        prefix = self.builder.build_prefix("some input", self.rules)
        assert "evaluate" in prefix.lower()
        assert "candidate" in prefix.lower()
        assert "matches" in prefix.lower()

    def test_prompt_contains_active_rules(self) -> None:
        prefix = self.builder.build_prefix("some input", self.rules)
        for rule in self.rules:
            assert rule in prefix

    def test_no_json_output_phrase(self) -> None:
        full = self.builder.build("input text", Candidate("sports"), self.rules)
        assert "Reply with a single JSON object" not in full

    def test_no_json_field_descriptions(self) -> None:
        full = self.builder.build("input text", Candidate("sports"), self.rules)
        assert '"decision" (integer 0 or 1)' not in full
        assert '"confidence" (integer 0-100)' not in full

    def test_no_no_markdown_phrase(self) -> None:
        full = self.builder.build("input text", Candidate("sports"), self.rules)
        assert "No markdown, no extra text" not in full

    def test_default_prompt_rules_produce_expected_structure(self) -> None:
        prefix = self.builder.build_prefix("some input", self.rules)
        assert "Classification rules:" in prefix
        for rule in self.rules:
            assert f"- {rule}" in prefix

    def test_candidate_specific_prompts_are_distinct(self) -> None:
        text = "The user wants to cancel their subscription."
        candidates = [
            Candidate("subscription cancellation"),
            Candidate("password reset"),
            Candidate("billing inquiry"),
        ]
        prompts = [self.builder.build(text, c, self.rules) for c in candidates]
        assert len(set(prompts)) == 3

    def test_each_prompt_contains_exactly_one_candidate(self) -> None:
        text = "The user wants to cancel their subscription."
        candidates = [Candidate("alpha"), Candidate("beta"), Candidate("gamma")]
        prompts = [self.builder.build(text, c, self.rules) for c in candidates]
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
