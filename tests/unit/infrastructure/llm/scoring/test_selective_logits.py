"""Tests for selective-logit capability probe and causal alignment."""
from __future__ import annotations

from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
    _probe_selective_logits,
    scored_positions,
)
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    SequenceLogits,
)

from ._gguf_mocks import FakeLlama

VOCAB = 8


def _adapter(model: FakeLlama | None = None) -> LlamaCppBatchedCausalLMAdapter:
    return LlamaCppBatchedCausalLMAdapter(
        model_path="unused.gguf",
        n_ctx=64,
        _llama=model or FakeLlama(n_vocab=VOCAB),
    )


class TestScoredPositions:
    def test_single_candidate_token(self) -> None:
        positions = scored_positions(prefix_token_count=3, candidate_token_count=1)
        assert list(positions) == [2]

    def test_two_candidate_tokens(self) -> None:
        positions = scored_positions(prefix_token_count=3, candidate_token_count=2)
        assert list(positions) == [2, 3]

    def test_three_candidate_tokens(self) -> None:
        positions = scored_positions(prefix_token_count=5, candidate_token_count=3)
        assert list(positions) == [4, 5, 6]

    def test_zero_candidate_tokens(self) -> None:
        positions = scored_positions(prefix_token_count=3, candidate_token_count=0)
        assert list(positions) == []

    def test_prefix_tail_included(self) -> None:
        positions = scored_positions(prefix_token_count=4, candidate_token_count=2)
        assert min(positions) == 3
        assert max(positions) == 4


class TestCausalAlignment:
    """Verify scored_positions matches LogitScorer's actual position reads."""

    def test_positions_match_logit_scorer_reads(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
            LogitScorer,
        )

        prefix_count = 3
        candidate_ids = [0, 1, 2]
        vocab = 4
        n_positions = prefix_count + len(candidate_ids)
        positions = [
            [0.0] * vocab for _ in range(n_positions)
        ]
        positions[prefix_count - 1][0] = 2.0
        positions[prefix_count][1] = 2.0
        positions[prefix_count + 1][2] = 2.0
        seq = SequenceLogits(logits=tuple(tuple(p) for p in positions))
        scorer = LogitScorer()
        score = scorer.score(prefix_count, candidate_ids, seq)
        assert score == score

        expected = list(scored_positions(prefix_count, len(candidate_ids)))
        actual = [prefix_count + k - 1 for k in range(len(candidate_ids))]
        assert expected == actual

    def test_first_candidate_token_from_prefix_tail(self) -> None:
        positions = scored_positions(prefix_token_count=1, candidate_token_count=1)
        assert list(positions) == [0]


class TestCapabilityProbe:
    def test_probe_returns_false_for_mock(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        assert _probe_selective_logits(model) is False

    def test_adapter_reports_selective_unsupported(self) -> None:
        adapter = _adapter()
        assert adapter.selective_logits_supported is False

    def test_describe_includes_selective_flag(self) -> None:
        adapter = _adapter()
        meta = adapter.describe()
        assert meta["selective_logits_supported"] is False


class TestFallbackBehavior:
    def test_fallback_produces_full_logits(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch([[1, 2, 3, 4]], [[1, 1, 1, 1]])
        assert len(result.items[0].logits) == 4
        for pos in result.items[0].logits:
            assert len(pos) == VOCAB

    def test_fallback_scores_match_baseline(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
            LogitScorer,
        )

        adapter = _adapter()
        result = adapter.forward_batch([[1, 2, 3, 4]], [[1, 1, 1, 1]])
        scorer = LogitScorer()
        score = scorer.score(
            prefix_token_count=2,
            candidate_token_ids=[0, 1],
            sequence_logits=result.items[0],
        )
        assert isinstance(score, float)
