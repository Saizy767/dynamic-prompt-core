"""Unit tests for the GGUF tokenizer adapter (``LlamaCppBatchTokenizerAdapter``)."""
from __future__ import annotations

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import GGUFProviderError
from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchTokenizerAdapter,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenizerAdapter,
)

from ._gguf_mocks import FakeLlama, MergingFakeLlama


def _adapter(llama: object | None = None) -> LlamaCppBatchTokenizerAdapter:
    return LlamaCppBatchTokenizerAdapter(llama or FakeLlama(n_vocab=8))


class TestProtocolSatisfaction:
    def test_satisfies_batch_protocol(self) -> None:
        assert isinstance(_adapter(), BatchTokenizerAdapter)


class TestPrefixPrefixBoundary:
    def test_boundary_correct_when_prefix_is_exact_prefix(self) -> None:
        adapter = _adapter()
        result = adapter.encode_batch(
            prefixes=["hello "],
            candidates=["world"],
        )
        assert result.prefix_token_counts[0] == len(b"hello ")
        assert result.candidate_token_ids[0] == list(b"world")

    def test_candidate_ids_match_continuation_positions(self) -> None:
        adapter = _adapter()
        result = adapter.encode_batch(["ab"], ["xy"])
        prefix_count = result.prefix_token_counts[0]
        candidate_ids = result.candidate_token_ids[0]
        full = result.input_ids[0]
        unpadded_len = prefix_count + len(candidate_ids)
        assert full[prefix_count:unpadded_len] == candidate_ids

    def test_multiple_items_boundaries_independent(self) -> None:
        adapter = _adapter()
        result = adapter.encode_batch(["ab", "abc"], ["x", "yz"])
        assert result.prefix_token_counts == [2, 3]
        assert result.candidate_token_ids[0] == list(b"x")
        assert result.candidate_token_ids[1] == list(b"yz")


class TestBoundaryMismatch:
    def test_cross_boundary_merge_rejects(self) -> None:
        adapter = LlamaCppBatchTokenizerAdapter(MergingFakeLlama())
        with pytest.raises(GGUFProviderError, match="boundary mismatch"):
            adapter.encode_batch(["xa"], ["by"])

    def test_no_silent_different_sequence(self) -> None:
        adapter = LlamaCppBatchTokenizerAdapter(MergingFakeLlama())
        with pytest.raises(GGUFProviderError):
            adapter.encode_batch(["xa"], ["by"])


class TestEmptyCandidate:
    def test_empty_candidate_no_double_bos(self) -> None:
        adapter = LlamaCppBatchTokenizerAdapter(FakeLlama(n_vocab=8), add_bos=True)
        result = adapter.encode_batch(["prefix"], [""])
        assert result.candidate_token_ids[0] == []
        assert result.prefix_token_counts[0] == len(b"prefix") + 1  # BOS + prefix

    def test_empty_candidate_without_bos(self) -> None:
        adapter = _adapter()
        result = adapter.encode_batch(["prefix"], [""])
        assert result.candidate_token_ids[0] == []
        assert result.prefix_token_counts[0] == len(b"prefix")


class TestPadding:
    def test_right_padding_applied(self) -> None:
        adapter = _adapter()
        result = adapter.encode_batch(["ab", "abcde"], ["x", "y"])
        assert len(result.input_ids[0]) == len(result.input_ids[1])

    def test_attention_mask_covers_non_padding(self) -> None:
        adapter = _adapter()
        result = adapter.encode_batch(["ab", "abcde"], ["x", "y"])
        for ids, mask in zip(result.input_ids, result.attention_mask, strict=True):
            assert len(ids) == len(mask)
            for _j, m in enumerate(mask):
                assert m in (0, 1)


class TestValidation:
    def test_unequal_length_raises(self) -> None:
        adapter = _adapter()
        with pytest.raises(GGUFProviderError, match="equal length"):
            adapter.encode_batch(["a", "b"], ["x"])
