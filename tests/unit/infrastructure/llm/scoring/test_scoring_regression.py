"""Regression tests: optimized scorer vs. baseline reference on a golden set.

The golden set is a fixed collection of inputs with known logits and
expected scores computed from a reference log-softmax.  These tests verify
per-token log-probabilities, final candidate scores, ranking consistency,
and edge cases (special tokens, EOS, non-finite, empty vectors, fallback).
The optimized path (selective extraction) is disabled by the capability
probe, so both paths use the full-read fallback — these tests pin the
scoring contract so any future optimization can be validated against them.
"""

from __future__ import annotations

import math

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
    LogitScorer,
    _log_probability,
)
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    SequenceLogits,
)

TOLERANCE = 1e-4
scorer = LogitScorer()


def _ref_log_prob(logits: list[float], token_id: int) -> float:
    max_val = max(logits)
    shifted = [v - max_val for v in logits]
    lse = max_val + math.log(sum(math.exp(v) for v in shifted))
    return logits[token_id] - lse


def _seq(*positions: list[float]) -> SequenceLogits:
    return SequenceLogits(logits=tuple(tuple(p) for p in positions))


def _logits(vocab: int, peak: int, strength: float = 2.0) -> list[float]:
    base = [0.0] * vocab
    base[peak] = strength
    return base


class TestGoldenSetPerTokenScores:
    def test_single_token_candidate(self) -> None:
        logits = _logits(vocab=4, peak=2, strength=3.0)
        seq = _seq(logits)
        score = scorer.score(1, [2], seq)
        expected = _ref_log_prob(logits, 2)
        assert score == pytest.approx(expected, abs=TOLERANCE)

    def test_two_token_candidate(self) -> None:
        p0 = _logits(vocab=4, peak=1, strength=2.0)
        p1 = _logits(vocab=4, peak=3, strength=2.0)
        seq = _seq(p0, p1)
        score = scorer.score(1, [1, 3], seq)
        expected = (_ref_log_prob(p0, 1) + _ref_log_prob(p1, 3)) / 2
        assert score == pytest.approx(expected, abs=TOLERANCE)

    def test_three_token_candidate(self) -> None:
        positions = [_logits(vocab=4, peak=i, strength=2.0) for i in range(3)]
        seq = _seq(*positions)
        score = scorer.score(1, [0, 1, 2], seq)
        expected = sum(_ref_log_prob(positions[i], i) for i in range(3)) / 3
        assert score == pytest.approx(expected, abs=TOLERANCE)

    def test_per_token_match_reference(self) -> None:
        positions = [_logits(vocab=5, peak=(i + 1) % 5, strength=1.5) for i in range(4)]
        for k in range(4):
            tid = (k + 1) % 5
            actual = _log_probability(positions[k], tid)
            expected = _ref_log_prob(positions[k], tid)
            assert actual == pytest.approx(expected, abs=TOLERANCE)


class TestRankingConsistency:
    def test_stronger_candidate_scores_higher(self) -> None:
        strong = _logits(vocab=4, peak=0, strength=5.0)
        weak = _logits(vocab=4, peak=0, strength=0.5)
        seq_strong = _seq(strong)
        seq_weak = _seq(weak)
        s_strong = scorer.score(1, [0], seq_strong)
        s_weak = scorer.score(1, [0], seq_weak)
        assert s_strong > s_weak

    def test_ranking_preserved_across_lengths(self) -> None:
        strong = _logits(vocab=4, peak=0, strength=5.0)
        weak = _logits(vocab=4, peak=0, strength=0.5)
        s_strong = scorer.score(1, [0, 0], _seq(strong, strong))
        s_weak = scorer.score(1, [0, 0], _seq(weak, weak))
        assert s_strong > s_weak


class TestCandidateLengths:
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_various_lengths(self, n: int) -> None:
        positions = [_logits(vocab=4, peak=0, strength=2.0) for _ in range(n)]
        seq = _seq(*positions)
        score = scorer.score(1, [0] * n, seq)
        expected = _ref_log_prob(positions[0], 0)
        assert score == pytest.approx(expected, abs=TOLERANCE)


class TestSpecialTokensAndEOS:
    def test_eos_token_scored_correctly(self) -> None:
        eos_id = 3
        logits = _logits(vocab=4, peak=eos_id, strength=4.0)
        seq = _seq(logits)
        score = scorer.score(1, [eos_id], seq)
        expected = _ref_log_prob(logits, eos_id)
        assert score == pytest.approx(expected, abs=TOLERANCE)

    def test_special_token_at_boundary(self) -> None:
        special = 0
        p0 = _logits(vocab=4, peak=1, strength=2.0)
        p1 = _logits(vocab=4, peak=special, strength=2.0)
        seq = _seq(p0, p1)
        score = scorer.score(1, [1, special], seq)
        expected = (_ref_log_prob(p0, 1) + _ref_log_prob(p1, special)) / 2
        assert score == pytest.approx(expected, abs=TOLERANCE)


class TestNonFiniteAndEmpty:
    def test_nan_logits_rejected(self) -> None:
        bad = [float("nan"), 0.0, 0.0]
        seq = _seq(bad)
        with pytest.raises(LLMScoringError):
            scorer.score(1, [0], seq)

    def test_inf_logits_rejected(self) -> None:
        bad = [float("-inf"), 0.0, 0.0]
        seq = _seq(bad)
        with pytest.raises(LLMScoringError):
            scorer.score(1, [0], seq)

    def test_empty_logit_vector_rejected(self) -> None:
        with pytest.raises(LLMScoringError, match="empty logit vector"):
            _log_probability([], 0)

    def test_empty_position_rejected_by_score(self) -> None:
        seq = _seq([])
        with pytest.raises(LLMScoringError):
            scorer.score(1, [0], seq)

    def test_empty_candidate_rejected(self) -> None:
        seq = _seq(_logits(vocab=4, peak=0))
        with pytest.raises(LLMScoringError):
            scorer.score(1, [], seq)


class TestFallbackBehavior:
    def test_selective_disabled_uses_full_read(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
            LlamaCppBatchedCausalLMAdapter,
        )

        from ._gguf_mocks import FakeLlama

        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="unused.gguf", n_ctx=64, _llama=FakeLlama(n_vocab=8)
        )
        assert adapter.selective_logits_supported is False
        result = adapter.forward_batch([[1, 2, 3, 4]], [[1, 1, 1, 1]])
        assert len(result.items[0].logits) == 4
        score = scorer.score(2, [0, 1], result.items[0])
        assert isinstance(score, float)


class TestSharedAndNonSharedPrefix:
    def test_shared_prefix_same_score(self) -> None:
        prefix_logits = [_logits(vocab=4, peak=0, strength=1.0) for _ in range(2)]
        prefix_tail = _logits(vocab=4, peak=1, strength=3.0)
        seq = _seq(*prefix_logits, prefix_tail)
        score = scorer.score(3, [1], seq)
        expected = _ref_log_prob(prefix_tail, 1)
        assert score == pytest.approx(expected, abs=TOLERANCE)

    def test_non_shared_prefix_different_score(self) -> None:
        p0 = _logits(vocab=4, peak=0, strength=3.0)
        p1 = _logits(vocab=4, peak=1, strength=3.0)
        seq_a = _seq(p0)
        seq_b = _seq(p1)
        s_a = scorer.score(1, [0], seq_a)
        s_b = scorer.score(1, [0], seq_b)
        assert s_a != pytest.approx(s_b, abs=TOLERANCE)
