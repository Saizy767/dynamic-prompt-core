"""Tests for shared-prefix KV-cache reuse: detection, probe, fallback, isolation."""
from __future__ import annotations

from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
    _probe_kv_cache_reuse,
    shared_prefix_length,
)
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer

from ._gguf_mocks import FakeLlama

VOCAB = 8


def _adapter(model: FakeLlama | None = None) -> LlamaCppBatchedCausalLMAdapter:
    return LlamaCppBatchedCausalLMAdapter(
        model_path="unused.gguf",
        n_ctx=64,
        _llama=model or FakeLlama(n_vocab=VOCAB),
    )


class TestSharedPrefixDetection:
    def test_identical_prefixes(self) -> None:
        input_ids = [[1, 2, 3, 10], [1, 2, 3, 20]]
        prefix_counts = [3, 3]
        assert shared_prefix_length(input_ids, prefix_counts) == 3

    def test_partial_prefix_overlap(self) -> None:
        input_ids = [[1, 2, 3, 10], [1, 2, 4, 20]]
        prefix_counts = [3, 3]
        assert shared_prefix_length(input_ids, prefix_counts) == 2

    def test_no_overlap(self) -> None:
        input_ids = [[1, 2, 3], [4, 5, 6]]
        prefix_counts = [3, 3]
        assert shared_prefix_length(input_ids, prefix_counts) == 0

    def test_empty_batch(self) -> None:
        assert shared_prefix_length([], []) == 0

    def test_single_item(self) -> None:
        input_ids = [[1, 2, 3, 4]]
        prefix_counts = [3]
        assert shared_prefix_length(input_ids, prefix_counts) == 3

    def test_capped_at_min_prefix_count(self) -> None:
        input_ids = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]
        prefix_counts = [5, 3]
        assert shared_prefix_length(input_ids, prefix_counts) == 3

    def test_token_level_not_text_level(self) -> None:
        input_ids = [[10, 20, 30], [10, 20, 40]]
        prefix_counts = [2, 2]
        result = shared_prefix_length(input_ids, prefix_counts)
        assert result == 2


class TestKVCacheReuseProbe:
    def test_probe_returns_false_for_mock(self) -> None:
        assert _probe_kv_cache_reuse(FakeLlama(n_vocab=VOCAB)) is False

    def test_adapter_reports_reuse_unsupported(self) -> None:
        adapter = _adapter()
        assert adapter.kv_cache_reuse_supported is False

    def test_describe_includes_reuse_flag(self) -> None:
        adapter = _adapter()
        meta = adapter.describe()
        assert meta["kv_cache_reuse_supported"] is False


class TestPrefixReuseFallback:
    def test_fallback_uses_per_candidate_reset(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model)
        adapter.forward_batch([[1, 2, 3], [1, 2, 4]], [[1, 1, 1], [1, 1, 1]])
        assert model.reset_count == 2
        assert len(model.eval_calls) == 2

    def test_fallback_scores_match_baseline(self) -> None:
        base = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        model = FakeLlama(n_vocab=VOCAB, logits_per_position=base)
        adapter = _adapter(model)
        result = adapter.forward_batch(
            [[1, 2, 3, 4], [1, 2, 3, 5]], [[1, 1, 1, 1], [1, 1, 1, 1]]
        )
        scorer = LogitScorer()
        s0 = scorer.score(2, [0, 1], result.items[0])
        s1 = scorer.score(2, [0, 1], result.items[1])
        assert isinstance(s0, float)
        assert isinstance(s1, float)


class TestCandidateIsolation:
    def test_no_cross_contamination_between_candidates(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model)
        adapter.forward_batch([[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [1, 1, 1]])
        assert model.eval_calls[0] == [1, 2, 3]
        assert model.eval_calls[1] == [4, 5, 6]
        assert model.reset_count == 2

    def test_shared_prefix_candidates_isolated(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model)
        shared = [1, 2, 3]
        adapter.forward_batch(
            [shared + [10], shared + [20]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
        )
        assert model.eval_calls[0] == [1, 2, 3, 10]
        assert model.eval_calls[1] == [1, 2, 3, 20]
        assert model.reset_count == 2
