"""Tests for the LogitScorer: causal alignment and mean log-probability."""
from __future__ import annotations

import math

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    SequenceLogits,
)

scorer = LogitScorer()


def _valid_logprobs(p0: float, vocab_size: int = 2) -> list[float]:
    """Return valid log-probs where token 0 has probability ``p0``.

    These sum to 1.0 when exponentiated, so log_softmax returns them unchanged.
    """
    remaining = 1.0 - p0
    if vocab_size > 1:
        p_other = remaining / (vocab_size - 1)
        return [math.log(p0) if i == 0 else math.log(p_other) for i in range(vocab_size)]
    return [math.log(p0)]


def _seq(*positions: list[float]) -> SequenceLogits:
    return SequenceLogits(logits=tuple(tuple(p) for p in positions))


class TestCausalAlignment:
    def test_first_candidate_token_from_last_prefix_position(self) -> None:
        # prefix [p0, p1, p2] (3 tokens), candidate [c0]
        # c0 logprob comes from logits[2] (position = 3-1 = 2)
        seq = _seq(_valid_logprobs(0.5), _valid_logprobs(0.5), _valid_logprobs(0.9))
        score = scorer.score(3, [0], seq)
        assert score == pytest.approx(math.log(0.9))

    def test_second_candidate_token_from_preceding_candidate_position(self) -> None:
        # prefix 3 tokens, candidate [c0, c1]
        # c0 from logits[2], c1 from logits[3]
        seq = _seq(
            _valid_logprobs(0.5), _valid_logprobs(0.5),
            _valid_logprobs(0.9), _valid_logprobs(0.8),
        )
        score = scorer.score(3, [0, 0], seq)
        expected = (math.log(0.9) + math.log(0.8)) / 2
        assert score == pytest.approx(expected)

    def test_not_reading_same_position(self) -> None:
        # With prefix_token_count=3, c0 reads position 2, not position 3
        seq = _seq(
            _valid_logprobs(0.5), _valid_logprobs(0.5),
            _valid_logprobs(0.9), _valid_logprobs(0.1),
        )
        score = scorer.score(3, [0], seq)
        assert score == pytest.approx(math.log(0.9))  # from position 2, not 3


class TestSingleTokenCandidate:
    def test_single_token_score(self) -> None:
        # prefix_token_count=1 → reads position 0
        seq = _seq(_valid_logprobs(0.8))
        score = scorer.score(1, [0], seq)
        assert score == pytest.approx(math.log(0.8))

    def test_single_token_second_vocab_id(self) -> None:
        seq = _seq(_valid_logprobs(0.3))
        score = scorer.score(1, [1], seq)
        assert score == pytest.approx(math.log(0.7))


class TestMultiTokenCandidate:
    def test_mean_not_sum(self) -> None:
        # prefix_token_count=1, candidate [c0, c1]
        # c0 from position 0, c1 from position 1
        seq = _seq(_valid_logprobs(0.9), _valid_logprobs(0.7))
        score = scorer.score(1, [0, 0], seq)
        expected_mean = (math.log(0.9) + math.log(0.7)) / 2
        expected_sum = math.log(0.9) + math.log(0.7)
        assert score == pytest.approx(expected_mean)
        assert score != pytest.approx(expected_sum)


class TestSemanticOrdering:
    def test_higher_score_for_stronger_candidate(self) -> None:
        seq_a = _seq(_valid_logprobs(0.9))
        seq_b = _seq(_valid_logprobs(0.2))
        score_a = scorer.score(1, [0], seq_a)
        score_b = scorer.score(1, [0], seq_b)
        assert score_a > score_b


class TestLengthNormalization:
    def test_longer_candidate_not_systematically_rewarded(self) -> None:
        # 1-token candidate with logprob log(0.9)
        seq_short = _seq(_valid_logprobs(0.9))
        score_short = scorer.score(1, [0], seq_short)

        # 3-token candidate with same per-token logprob log(0.9)
        seq_long = _seq(_valid_logprobs(0.9), _valid_logprobs(0.9), _valid_logprobs(0.9))
        score_long = scorer.score(1, [0, 0, 0], seq_long)

        # Both have the same mean: no length bias
        assert score_short == pytest.approx(score_long)
        assert score_short == pytest.approx(math.log(0.9))


class TestInvalidInputs:
    def test_empty_candidate_tokens_rejected(self) -> None:
        seq = _seq(_valid_logprobs(0.5))
        with pytest.raises(LLMScoringError):
            scorer.score(1, [], seq)

    def test_position_out_of_range_rejected(self) -> None:
        seq = _seq(_valid_logprobs(0.5))
        with pytest.raises(LLMScoringError):
            scorer.score(5, [0], seq)

    def test_token_id_out_of_range_rejected(self) -> None:
        seq = _seq(_valid_logprobs(0.5))
        with pytest.raises(LLMScoringError):
            scorer.score(0, [99], seq)


class TestNoSilentClamping:
    def test_large_negative_score_not_clamped(self) -> None:
        seq = _seq(_valid_logprobs(0.001))
        score = scorer.score(1, [0], seq)
        assert score < 0
        assert score == pytest.approx(math.log(0.001))
