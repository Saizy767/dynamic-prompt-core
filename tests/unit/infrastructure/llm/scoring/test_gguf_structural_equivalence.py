"""Structural-equivalence and causal-alignment tests for the GGUF provider.

Task 2.6: assert ``LogitScorer`` receives identical structural semantics from
the GGUF adapter and the existing contract double, and verify causal logit
alignment with known token IDs.
"""
from __future__ import annotations

import math

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
)
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedLogits,
    SequenceLogits,
)

from ._gguf_mocks import FakeLlama

VOCAB = 8


class _StubContractModel:
    """Mimics the existing Hugging Face adapter contract (plain floats)."""

    def __init__(self, logits_per_position: list[float]) -> None:
        self._lpp = logits_per_position

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        items = []
        for seq in input_ids:
            positions = [tuple(self._lpp) for _ in range(len(seq))]
            items.append(SequenceLogits(logits=tuple(positions)))
        return BatchedLogits(items=tuple(items))


class TestStructuralEquivalence:
    """LogitScorer receives identical structural semantics from both providers."""

    def test_identical_scores_from_gguf_and_contract_double(self) -> None:
        logits = [1.0, 2.0, 0.5, 3.0, 0.0, 1.5, 2.5, 0.25]
        gguf_adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="x.gguf",
            n_ctx=32,
            _llama=FakeLlama(n_vocab=VOCAB, logits_per_position=logits),
        )
        contract_model = _StubContractModel(logits)

        input_ids = [[10, 20, 30, 40]]
        attention_mask = [[1, 1, 1, 1]]

        gguf_result = gguf_adapter.forward_batch(input_ids, attention_mask)
        contract_result = contract_model.forward_batch(input_ids, attention_mask)

        assert len(gguf_result.items) == len(contract_result.items)
        for g_item, c_item in zip(gguf_result.items, contract_result.items, strict=True):
            assert len(g_item.logits) == len(c_item.logits)
            for g_pos, c_pos in zip(g_item.logits, c_item.logits, strict=True):
                assert len(g_pos) == len(c_pos)
                for gv, cv in zip(g_pos, c_pos, strict=True):
                    assert isinstance(gv, float)
                    assert isinstance(cv, float)
                    assert gv == pytest.approx(cv)

        scorer = LogitScorer()
        gguf_scores = scorer.score_batch(
            prefix_token_counts=[2],
            candidate_token_ids=[[1, 2]],
            batched_logits=gguf_result,
        )
        contract_scores = scorer.score_batch(
            prefix_token_counts=[2],
            candidate_token_ids=[[1, 2]],
            batched_logits=contract_result,
        )
        assert gguf_scores[0] == pytest.approx(contract_scores[0])


class TestCausalAlignment:
    """Verify each candidate token is scored from its preceding position."""

    def test_first_candidate_token_from_final_prefix_position(self) -> None:
        c0, c1 = 1, 2
        prefix_count = 3
        peak_c0 = [0.0, 10.0, 0.0, 0.0]
        peak_c1 = [0.0, 0.0, 10.0, 0.0]
        peak_wrong = [10.0, 0.0, 0.0, 0.0]

        logits = SequenceLogits(
            logits=(
                tuple(peak_wrong),
                tuple(peak_wrong),
                tuple(peak_c0),
                tuple(peak_c1),
                tuple(peak_wrong),
            )
        )

        scorer = LogitScorer()
        score = scorer.score(
            prefix_token_count=prefix_count,
            candidate_token_ids=[c0, c1],
            sequence_logits=logits,
        )

        expected = (
            _log_softmax_value(peak_c0, c0) + _log_softmax_value(peak_c1, c1)
        ) / 2
        assert score == pytest.approx(expected, abs=1e-6)
        assert score > -0.01

    def test_wrong_position_would_score_low(self) -> None:
        c0, c1 = 1, 2
        prefix_count = 3
        peak_c0 = [0.0, 10.0, 0.0, 0.0]
        peak_c1 = [0.0, 0.0, 10.0, 0.0]
        peak_wrong = [10.0, 0.0, 0.0, 0.0]

        logits = SequenceLogits(
            logits=(
                tuple(peak_c0),
                tuple(peak_c1),
                tuple(peak_wrong),
                tuple(peak_wrong),
                tuple(peak_c1),
            )
        )

        scorer = LogitScorer()
        score = scorer.score(
            prefix_token_count=prefix_count,
            candidate_token_ids=[c0, c1],
            sequence_logits=logits,
        )
        assert score < -5.0


def _log_softmax_value(logits: list[float], token_id: int) -> float:
    max_val = max(logits)
    shifted = [v - max_val for v in logits]
    log_sum_exp = math.log(sum(math.exp(v) for v in shifted))
    return shifted[token_id] - log_sum_exp
