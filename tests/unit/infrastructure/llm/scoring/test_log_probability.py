"""Unit tests for the direct single-token log-probability function."""
from __future__ import annotations

import math
import random

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring import logit_scorer as ls_module
from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import _log_probability
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    SequenceLogits,
)


def _reference_log_softmax(logits: list[float], token_id: int) -> float:
    """Reference implementation: full log-softmax then index."""
    max_val = max(logits)
    shifted = [v - max_val for v in logits]
    log_sum_exp = max_val + math.log(sum(math.exp(v) for v in shifted))
    return logits[token_id] - log_sum_exp


class TestNumericalEquivalence:
    def test_matches_reference_on_random_vectors(self) -> None:
        rng = random.Random(42)
        for _ in range(200):
            n = rng.randint(2, 50)
            logits = [rng.uniform(-10, 10) for _ in range(n)]
            tid = rng.randint(0, n - 1)
            assert _log_probability(logits, tid) == pytest.approx(
                _reference_log_softmax(logits, tid), abs=1e-9
            )

    def test_matches_reference_on_extreme_magnitude(self) -> None:
        cases = [
            ([1000.0, 1001.0, 1002.0], 1),
            ([-1000.0, -1001.0, -999.0], 0),
            ([1e8, 1e8 + 1, 1e8 - 1], 2),
            ([-1e8, -1e8 + 1, -1e8 - 1], 0),
        ]
        for logits, tid in cases:
            assert _log_probability(logits, tid) == pytest.approx(
                _reference_log_softmax(logits, tid), abs=1e-9
            )

    def test_returns_plain_float(self) -> None:
        result = _log_probability([1.0, 2.0, 3.0], 1)
        assert isinstance(result, float)


class TestEmptyVector:
    def test_empty_raises_llm_scoring_error(self) -> None:
        with pytest.raises(LLMScoringError, match="empty logit vector"):
            _log_probability([], 0)


class TestNonFiniteRejection:
    def test_nan_logits_rejected_by_score(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
            LogitScorer,
        )

        bad = [float("nan"), 0.0]
        seq = SequenceLogits(logits=(tuple(bad),))
        with pytest.raises(LLMScoringError):
            LogitScorer().score(1, [0], seq)

    def test_inf_logits_rejected_by_score(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
            LogitScorer,
        )

        bad = [float("inf"), 0.0]
        seq = SequenceLogits(logits=(tuple(bad),))
        with pytest.raises(LLMScoringError):
            LogitScorer().score(1, [0], seq)

    def test_neg_inf_logits_rejected_by_score(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
            LogitScorer,
        )

        bad = [float("-inf"), 0.0]
        seq = SequenceLogits(logits=(tuple(bad),))
        with pytest.raises(LLMScoringError):
            LogitScorer().score(1, [0], seq)


class TestPurePythonFallback:
    def test_fallback_matches_numpy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logits = [random.Random(7).uniform(-5, 5) for _ in range(20)]
        numpy_result = _log_probability(logits, 5)
        monkeypatch.setattr(ls_module, "_np", None)
        fallback_result = _log_probability(logits, 5)
        assert fallback_result == pytest.approx(numpy_result, abs=1e-9)

    def test_fallback_extreme_magnitude(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ls_module, "_np", None)
        logits = [1e8, 1e8 + 1, 1e8 - 1]
        result = _log_probability(logits, 1)
        assert result == pytest.approx(_reference_log_softmax(logits, 1), abs=1e-9)

    def test_fallback_empty_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ls_module, "_np", None)
        with pytest.raises(LLMScoringError, match="empty logit vector"):
            _log_probability([], 0)
