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
