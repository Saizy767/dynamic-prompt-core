"""Tests for batch scoring: tokenizer, model, logit scorer, and scorer."""
from __future__ import annotations

import math
import typing

import pytest

from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer import (
    LLMLogitCandidateScorer,
)
from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedCausalLanguageModel,
    BatchedLogits,
    SequenceLogits,
    SequentialBatchCompatibilityAdapter,
)
from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
    ScoringPromptBuilder,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenization,
    BatchTokenizerAdapter,
)

VOCAB_SIZE = 256


def _uniform_logprobs(vocab_size: int = VOCAB_SIZE) -> list[float]:
    lp = math.log(1.0 / vocab_size)
    return [lp] * vocab_size


class StubBatchTokenizer:
    """Deterministic batch tokenizer: maps each character to its ord value.

    Separate encoding is safe for this tokenizer because each character is
    independently tokenized.
    """

    PAD_TOKEN_ID = 0

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tid) for tid in token_ids)

    def encode_batch(
        self,
        prefixes: list[str],
        candidates: list[str],
    ) -> BatchTokenization:
        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_prefix_counts: list[int] = []
        all_candidate_ids: list[list[int]] = []

        for prefix, candidate in zip(prefixes, candidates, strict=True):
            prefix_ids = [ord(ch) for ch in prefix]
            candidate_ids = [ord(ch) for ch in candidate]
            full_ids = prefix_ids + candidate_ids

            all_input_ids.append(full_ids)
            all_attention_masks.append([1] * len(full_ids))
            all_prefix_counts.append(len(prefix_ids))
            all_candidate_ids.append(candidate_ids)

        max_len = max(len(ids) for ids in all_input_ids) if all_input_ids else 0
        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            all_input_ids[i] = all_input_ids[i] + [self.PAD_TOKEN_ID] * pad_len
            all_attention_masks[i] = all_attention_masks[i] + [0] * pad_len

        return BatchTokenization(
            input_ids=all_input_ids,
            attention_mask=all_attention_masks,
            prefix_token_counts=all_prefix_counts,
            candidate_token_ids=all_candidate_ids,
        )


class StubBatchModel:
    """Deterministic batch model double returning uniform logits per position.

    For a batch of B sequences padded to length S, returns B x S positions of
    VOCAB_SIZE-token logits with a uniform distribution.
    """

    def __init__(self) -> None:
        self._lp = _uniform_logprobs()
        self.forward_batch_calls: list[tuple[list[list[int]], list[list[int]]]] = []

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        self.forward_batch_calls.append((list(input_ids), list(attention_mask)))
        batch_logits: list[SequenceLogits] = []
        for seq in input_ids:
            n = len(seq)
            batch_logits.append(
                SequenceLogits(logits=tuple(tuple(self._lp) for _ in range(n)))
            )
        return BatchedLogits(items=tuple(batch_logits))


class StubBatchModelWithProb:
    """Batch model double that assigns different probs to different tokens."""

    def __init__(self, prob_for_token: dict[int, float]) -> None:
        self._prob_for_token = prob_for_token
        self.forward_batch_calls: list[tuple[list[list[int]], list[list[int]]]] = []

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        self.forward_batch_calls.append((list(input_ids), list(attention_mask)))
        batch_logits: list[SequenceLogits] = []
        for seq in input_ids:
            n = len(seq)
            positions: list[list[float]] = []
            for i in range(n):
                token_id = seq[i] if i < n else -1
                p = self._prob_for_token.get(token_id, 0.5)
                lp0 = math.log(p)
                lp1 = math.log(1.0 - p)
                positions.append([lp0, lp1] + [lp1] * (VOCAB_SIZE - 2))
            batch_logits.append(SequenceLogits(logits=tuple(tuple(p) for p in positions)))
        return BatchedLogits(items=tuple(batch_logits))


class FailingBatchModel:
    """Batch model double that always raises."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        raise self._exc


def _make_batch_scorer(
    tokenizer: StubBatchTokenizer | None = None,
    model: object | None = None,
) -> LLMLogitCandidateScorer:
    return LLMLogitCandidateScorer(
        prompt_builder=ScoringPromptBuilder(),
        tokenizer=tokenizer or StubBatchTokenizer(),
        model=model or StubBatchModel(),  # type: ignore[arg-type]
        logit_scorer=LogitScorer(),
    )


# ---------------------------------------------------------------------------
# Task 1.1: BatchTokenizerAdapter protocol and batch result type
# ---------------------------------------------------------------------------


class TestBatchTokenizerAdapterProtocol:
    def test_is_a_typing_protocol(self) -> None:
        assert typing.is_protocol(BatchTokenizerAdapter)

    def test_is_runtime_checkable(self) -> None:
        assert getattr(BatchTokenizerAdapter, "_is_runtime_protocol", False)

    def test_stub_satisfies_protocol(self) -> None:
        assert isinstance(StubBatchTokenizer(), BatchTokenizerAdapter)


class TestBatchTokenization:
    def test_one_item_per_prompt(self) -> None:
        tokenizer = StubBatchTokenizer()
        result = tokenizer.encode_batch(
            prefixes=["prefix_a", "prefix_b"],
            candidates=["cand_a", "cand_b"],
        )
        assert len(result.input_ids) == 2
        assert len(result.attention_mask) == 2
        assert len(result.prefix_token_counts) == 2
        assert len(result.candidate_token_ids) == 2

    def test_per_item_boundaries_recorded(self) -> None:
        tokenizer = StubBatchTokenizer()
        result = tokenizer.encode_batch(
            prefixes=["ab", "abc"],
            candidates=["xy", "w"],
        )
        assert result.prefix_token_counts[0] == 2
        assert result.prefix_token_counts[1] == 3
        assert result.candidate_token_ids[0] == [ord("x"), ord("y")]
        assert result.candidate_token_ids[1] == [ord("w")]

    def test_right_padding_applied(self) -> None:
        tokenizer = StubBatchTokenizer()
        result = tokenizer.encode_batch(
            prefixes=["ab", "abcde"],
            candidates=["x", "y"],
        )
        len0 = len(result.input_ids[0])
        len1 = len(result.input_ids[1])
        assert len0 == len1  # same padded length

        unpadded_len0 = 3  # "ab" + "x" = 3 tokens
        for i in range(unpadded_len0):
            assert result.input_ids[0][i] != StubBatchTokenizer.PAD_TOKEN_ID
        for i in range(unpadded_len0, len0):
            assert result.input_ids[0][i] == StubBatchTokenizer.PAD_TOKEN_ID

    def test_attention_mask_covers_non_padding(self) -> None:
        tokenizer = StubBatchTokenizer()
        result = tokenizer.encode_batch(
            prefixes=["ab", "abcde"],
            candidates=["x", "y"],
        )
        for i in range(len(result.input_ids)):
            for j in range(len(result.input_ids[i])):
                if result.input_ids[i][j] == StubBatchTokenizer.PAD_TOKEN_ID:
                    assert result.attention_mask[i][j] == 0
                else:
                    assert result.attention_mask[i][j] == 1

    def test_candidate_ids_match_continuation_positions(self) -> None:
        tokenizer = StubBatchTokenizer()
        result = tokenizer.encode_batch(
            prefixes=["hello"],
            candidates=["world"],
        )
        prefix_count = result.prefix_token_counts[0]
        candidate_ids = result.candidate_token_ids[0]
        full_ids = result.input_ids[0]
        unpadded_len = prefix_count + len(candidate_ids)
        assert full_ids[prefix_count:unpadded_len] == candidate_ids


# ---------------------------------------------------------------------------
# Task 1.2: HuggingFaceBatchTokenizerAdapter (protocol satisfaction)
# ---------------------------------------------------------------------------


class TestHuggingFaceBatchTokenizerAdapterProtocol:
    def test_satisfies_batch_protocol(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
            HuggingFaceBatchTokenizerAdapter,
        )

        assert getattr(HuggingFaceBatchTokenizerAdapter, "encode_batch", None) is not None


# ---------------------------------------------------------------------------
# Task 2.1: BatchedCausalLanguageModel protocol and BatchedLogits
# ---------------------------------------------------------------------------


class TestBatchedCausalLanguageModelProtocol:
    def test_is_a_typing_protocol(self) -> None:
        assert typing.is_protocol(BatchedCausalLanguageModel)

    def test_is_runtime_checkable(self) -> None:
        assert getattr(BatchedCausalLanguageModel, "_is_runtime_protocol", False)

    def test_stub_satisfies_protocol(self) -> None:
        assert isinstance(StubBatchModel(), BatchedCausalLanguageModel)


class TestBatchedLogits:
    def test_one_item_per_batch_item(self) -> None:
        model = StubBatchModel()
        result = model.forward_batch(
            input_ids=[[1, 2, 3], [4, 5]],
            attention_mask=[[1, 1, 1], [1, 1, 0]],
        )
        assert len(result.items) == 2

    def test_item_order_preserved(self) -> None:
        model = StubBatchModel()
        result = model.forward_batch(
            input_ids=[[1, 2], [3, 4], [5, 6]],
            attention_mask=[[1, 1], [1, 1], [1, 1]],
        )
        assert len(result.items[0].logits) == 2
        assert len(result.items[1].logits) == 2
        assert len(result.items[2].logits) == 2

    def test_semantic_shape_b_s_v(self) -> None:
        model = StubBatchModel()
        b, s = 3, 5
        input_ids = [[1] * s for _ in range(b)]
        attention_mask = [[1] * s for _ in range(b)]
        result = model.forward_batch(input_ids, attention_mask)
        assert len(result.items) == b  # batch size
        for item in result.items:
            assert len(item.logits) == s  # sequence length
            for pos in item.logits:
                assert len(pos) == VOCAB_SIZE  # vocab size


# ---------------------------------------------------------------------------
# Task 2.2: TorchBatchedCausalLMAdapter (protocol satisfaction)
# ---------------------------------------------------------------------------


class TestTorchBatchedCausalLMAdapterProtocol:
    def test_has_forward_batch(self) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
            TorchBatchedCausalLMAdapter,
        )

        assert getattr(TorchBatchedCausalLMAdapter, "forward_batch", None) is not None


# ---------------------------------------------------------------------------
# Task 3.1 + 3.2: LogitScorer.score_batch
# ---------------------------------------------------------------------------


def _valid_logprobs(p0: float, vocab_size: int = VOCAB_SIZE) -> list[float]:
    remaining = 1.0 - p0
    if vocab_size > 1:
        p_other = remaining / (vocab_size - 1)
        return [math.log(p0) if i == 0 else math.log(p_other) for i in range(vocab_size)]
    return [math.log(p0)]


class TestScoreBatch:
    def test_per_item_alignment_matches_sequential(self) -> None:
        scorer = LogitScorer()
        lp = _valid_logprobs(0.9)
        seq_a = SequenceLogits(logits=(tuple(lp), tuple(lp), tuple(lp)))
        seq_b = SequenceLogits(logits=(tuple(lp), tuple(lp), tuple(lp)))
        batched = BatchedLogits(items=(seq_a, seq_b))

        scores = scorer.score_batch(
            prefix_token_counts=[2, 1],
            candidate_token_ids=[[0], [0, 0]],
            batched_logits=batched,
        )
        assert len(scores) == 2

        single_a = scorer.score(
            prefix_token_count=2,
            candidate_token_ids=[0],
            sequence_logits=seq_a,
        )
        single_b = scorer.score(
            prefix_token_count=1,
            candidate_token_ids=[0, 0],
            sequence_logits=seq_b,
        )
        assert scores[0] == pytest.approx(single_a)
        assert scores[1] == pytest.approx(single_b)

    def test_variable_length_candidates(self) -> None:
        scorer = LogitScorer()
        lp = _valid_logprobs(0.8)
        seq_a = SequenceLogits(logits=(tuple(lp), tuple(lp)))
        seq_b = SequenceLogits(logits=(tuple(lp), tuple(lp), tuple(lp), tuple(lp)))
        batched = BatchedLogits(items=(seq_a, seq_b))

        scores = scorer.score_batch(
            prefix_token_counts=[1, 1],
            candidate_token_ids=[[0], [0, 0, 0]],
            batched_logits=batched,
        )
        assert len(scores) == 2
        assert scores[0] == pytest.approx(math.log(0.8))
        assert scores[1] == pytest.approx(math.log(0.8))

    def test_invalid_score_raises_for_entire_batch(self) -> None:
        scorer = LogitScorer()
        lp = _valid_logprobs(0.5)
        good_seq = SequenceLogits(logits=(tuple(lp), tuple(lp)))
        batched = BatchedLogits(items=(good_seq, good_seq))

        with pytest.raises(LLMScoringError):
            scorer.score_batch(
                prefix_token_counts=[1, 5],
                candidate_token_ids=[[0], [0]],
                batched_logits=batched,
            )

    def test_no_partial_results_on_failure(self) -> None:
        scorer = LogitScorer()
        lp = _valid_logprobs(0.5)
        seq = SequenceLogits(logits=(tuple(lp), tuple(lp)))
        batched = BatchedLogits(items=(seq, seq))

        with pytest.raises(LLMScoringError):
            scorer.score_batch(
                prefix_token_counts=[1, 5],
                candidate_token_ids=[[0], [0]],
                batched_logits=batched,
            )


# ---------------------------------------------------------------------------
# Task 4.1: Batch scorer — one forward_batch call
# ---------------------------------------------------------------------------


class TestBatchScorerOneForwardCall:
    async def test_one_forward_batch_call_for_n_candidates(self) -> None:
        model = StubBatchModel()
        scorer = _make_batch_scorer(model=model)

        candidates = [Candidate("a"), Candidate("b"), Candidate("c")]
        judgments = await scorer.score("text", candidates)

        assert len(judgments) == 3
        assert len(model.forward_batch_calls) == 1

    async def test_judgments_in_input_order(self) -> None:
        model = StubBatchModel()
        scorer = _make_batch_scorer(model=model)

        candidates = [Candidate("c"), Candidate("a"), Candidate("b")]
        judgments = await scorer.score("text", candidates)

        assert [j.candidate.value for j in judgments] == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# Task 4.2: Error translation in batch path
# ---------------------------------------------------------------------------


class TestBatchErrorTranslation:
    async def test_model_error_translated(self) -> None:
        model = FailingBatchModel(RuntimeError("torch error"))
        scorer = _make_batch_scorer(model=model)

        with pytest.raises(CandidateScoringError) as exc_info:
            await scorer.score("text", [Candidate("a")])

        assert "torch error" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    async def test_no_partial_result_on_failure(self) -> None:
        model = FailingBatchModel(RuntimeError("batch fail"))
        scorer = _make_batch_scorer(model=model)

        try:
            await scorer.score("text", [Candidate("a"), Candidate("b"), Candidate("c")])
        except CandidateScoringError:
            pass
        else:
            pytest.fail("should have raised, not returned a partial result")


# ---------------------------------------------------------------------------
# Task 4.3: Sequential compatibility adapter
# ---------------------------------------------------------------------------


class StubSingleModel:
    """Single-sequence model returning uniform logits."""

    def __init__(self) -> None:
        self._lp = _uniform_logprobs()
        self.forward_calls: list[list[int]] = []

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        self.forward_calls.append(list(input_ids))
        n = len(input_ids)
        return SequenceLogits(logits=tuple(tuple(self._lp) for _ in range(n)))


class TestSequentialCompatibilityAdapter:
    def test_satisfies_batched_protocol(self) -> None:
        adapter = SequentialBatchCompatibilityAdapter(StubSingleModel())
        assert isinstance(adapter, BatchedCausalLanguageModel)

    def test_delegates_to_forward_per_item(self) -> None:
        single = StubSingleModel()
        adapter = SequentialBatchCompatibilityAdapter(single)

        result = adapter.forward_batch(
            input_ids=[[1, 2, 3], [4, 5, 0]],
            attention_mask=[[1, 1, 1], [1, 1, 0]],
        )
        assert len(result.items) == 2
        assert len(single.forward_calls) == 2
        assert single.forward_calls[0] == [1, 2, 3]
        assert single.forward_calls[1] == [4, 5]  # padding stripped

    async def test_same_scores_as_native_batch(self) -> None:
        tokenizer = StubBatchTokenizer()
        logit_scorer = LogitScorer()
        builder = ScoringPromptBuilder()

        batch_model = StubBatchModel()
        batch_scorer = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=batch_model,
            logit_scorer=logit_scorer,
        )

        single_model = StubSingleModel()
        compat_adapter = SequentialBatchCompatibilityAdapter(single_model)
        seq_scorer = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=compat_adapter,
            logit_scorer=logit_scorer,
        )

        candidates = [Candidate("a"), Candidate("b"), Candidate("c")]
        batch_judgments = await batch_scorer.score("text", candidates)
        seq_judgments = await seq_scorer.score("text", candidates)

        for bj, sj in zip(batch_judgments, seq_judgments, strict=True):
            assert bj.score == pytest.approx(sj.score)


# ---------------------------------------------------------------------------
# Task 5.1: Batch vs sequential equivalence
# ---------------------------------------------------------------------------


class TestBatchSequentialEquivalence:
    async def test_equivalent_scores_variable_length_candidates(self) -> None:
        tokenizer = StubBatchTokenizer()
        logit_scorer = LogitScorer()
        builder = ScoringPromptBuilder()

        batch_model = StubBatchModel()
        batch_scorer = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=batch_model,
            logit_scorer=logit_scorer,
        )

        single_model = StubSingleModel()
        compat_adapter = SequentialBatchCompatibilityAdapter(single_model)
        seq_scorer = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=compat_adapter,
            logit_scorer=logit_scorer,
        )

        candidates = [
            Candidate("tax"),
            Candidate("financial regulation"),
            Candidate("international financial regulation"),
        ]
        batch_judgments = await batch_scorer.score("text", candidates)
        seq_judgments = await seq_scorer.score("text", candidates)

        batch_map = {j.candidate.value: j.score for j in batch_judgments}
        seq_map = {j.candidate.value: j.score for j in seq_judgments}

        for value in [c.value for c in candidates]:
            assert batch_map[value] == pytest.approx(seq_map[value])


# ---------------------------------------------------------------------------
# Task 5.2: Ordering with strengthened invariant
# ---------------------------------------------------------------------------


class TestBatchOrderingInvariant:
    async def test_judgment_at_index_i_references_candidate_at_index_i(self) -> None:
        model = StubBatchModel()
        scorer = _make_batch_scorer(model=model)

        candidates = [Candidate("alpha"), Candidate("beta"), Candidate("gamma")]
        judgments = await scorer.score("text", candidates)

        assert len(judgments) == len(candidates)
        for i, candidate in enumerate(candidates):
            assert judgments[i].candidate == candidate

    async def test_order_preserved_with_variable_lengths(self) -> None:
        model = StubBatchModel()
        scorer = _make_batch_scorer(model=model)

        candidates = [
            Candidate("x"),
            Candidate("longer candidate"),
            Candidate("medium"),
        ]
        judgments = await scorer.score("text", candidates)

        assert [j.candidate.value for j in judgments] == [
            "x",
            "longer candidate",
            "medium",
        ]
        for i, candidate in enumerate(candidates):
            assert judgments[i].candidate == candidate


# ---------------------------------------------------------------------------
# Task 5.3: Single-candidate batch (batch of size 1)
# ---------------------------------------------------------------------------


class TestSingleCandidateBatch:
    async def test_single_candidate_same_as_sequential(self) -> None:
        tokenizer = StubBatchTokenizer()
        logit_scorer = LogitScorer()
        builder = ScoringPromptBuilder()

        batch_model = StubBatchModel()
        batch_scorer = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=batch_model,
            logit_scorer=logit_scorer,
        )

        single_model = StubSingleModel()
        compat_adapter = SequentialBatchCompatibilityAdapter(single_model)
        seq_scorer = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=compat_adapter,
            logit_scorer=logit_scorer,
        )

        candidate = Candidate("only one")
        batch_judgments = await batch_scorer.score("text", [candidate])
        seq_judgments = await seq_scorer.score("text", [candidate])

        assert len(batch_judgments) == 1
        assert batch_judgments[0].score == pytest.approx(seq_judgments[0].score)


# ---------------------------------------------------------------------------
# Task 5.4: Padding does not change score
# ---------------------------------------------------------------------------


class TestPaddingInvariance:
    async def test_short_candidate_score_unchanged_in_batch(self) -> None:
        tokenizer = StubBatchTokenizer()
        logit_scorer = LogitScorer()
        builder = ScoringPromptBuilder()

        short = Candidate("x")
        long = Candidate("a much longer candidate that causes padding")

        model_alone = StubBatchModel()
        scorer_alone = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=model_alone,
            logit_scorer=logit_scorer,
        )
        judgments_alone = await scorer_alone.score("text", [short])
        score_alone = judgments_alone[0].score

        model_batch = StubBatchModel()
        scorer_batch = LLMLogitCandidateScorer(
            prompt_builder=builder,
            tokenizer=tokenizer,
            model=model_batch,
            logit_scorer=logit_scorer,
        )
        judgments_batch = await scorer_batch.score("text", [short, long])
        score_batch = judgments_batch[0].score

        assert score_alone == pytest.approx(score_batch)
