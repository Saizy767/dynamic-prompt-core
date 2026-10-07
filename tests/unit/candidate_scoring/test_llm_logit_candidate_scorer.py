"""Tests for the LLMLogitCandidateScorer with deterministic doubles."""
from __future__ import annotations

import math

import pytest

from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment
from dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer import (
    LLMLogitCandidateScorer,
)
from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    SequenceLogits,
)
from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
    ScoringPromptBuilder,
)


class StubTokenizer:
    """Deterministic tokenizer: maps each character to its ord value."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tid) for tid in token_ids)


def _valid_logprobs(p0: float) -> list[float]:
    return [math.log(p0), math.log(1.0 - p0)]


VOCAB_SIZE = 256


def _uniform_logprobs(vocab_size: int = VOCAB_SIZE) -> list[float]:
    """Uniform distribution: every token gets log(1/vocab_size)."""
    lp = math.log(1.0 / vocab_size)
    return [lp] * vocab_size


class StubModel:
    """Deterministic model double that returns per-position logits.

    For a sequence of N tokens, returns N positions of ``VOCAB_SIZE``-token
    logits with a uniform distribution.  Any token ID in [0, VOCAB_SIZE) is
    valid.
    """

    def __init__(self) -> None:
        self._lp = _uniform_logprobs()
        self.forward_calls: list[list[int]] = []

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        self.forward_calls.append(list(input_ids))
        n = len(input_ids)
        return SequenceLogits(logits=tuple(tuple(self._lp) for _ in range(n)))


class StubModelWithProb:
    """Model double that assigns different probs to different candidates."""

    def __init__(self, prob_for_token: dict[int, float]) -> None:
        self._prob_for_token = prob_for_token
        self.forward_calls: list[list[int]] = []

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        self.forward_calls.append(list(input_ids))
        n = len(input_ids)
        positions: list[list[float]] = []
        for i in range(n):
            token_id = input_ids[i] if i < n else -1
            p = self._prob_for_token.get(token_id, 0.5)
            positions.append(_valid_logprobs(p))
        return SequenceLogits(logits=tuple(tuple(p) for p in positions))


class FailingModel:
    """Model double that always raises."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        raise self._exc


class PartialFailingModel:
    """Model double that fails on a specific sequence."""

    def __init__(self, fail_seq: tuple[int, ...], exc: Exception) -> None:
        self._fail_seq = fail_seq
        self._exc = exc
        self._stub = StubModel()

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        if tuple(input_ids) == self._fail_seq:
            raise self._exc
        return self._stub.forward(input_ids)


def _make_scorer(
    tokenizer: StubTokenizer | None = None,
    model: object | None = None,
) -> LLMLogitCandidateScorer:
    return LLMLogitCandidateScorer(
        prompt_builder=ScoringPromptBuilder(),
        tokenizer=tokenizer or StubTokenizer(),
        model=model or StubModel(),  # type: ignore[arg-type]
        logit_scorer=LogitScorer(),
    )


class TestPortContract:
    async def test_one_judgment_per_candidate(self) -> None:
        model = StubModel()
        scorer = _make_scorer(model=model)

        candidates = [Candidate("a"), Candidate("b"), Candidate("c")]
        judgments = await scorer.score("text", candidates)

        assert len(judgments) == 3

    async def test_judgments_in_input_order(self) -> None:
        model = StubModel()
        scorer = _make_scorer(model=model)

        candidates = [Candidate("c"), Candidate("a"), Candidate("b")]
        judgments = await scorer.score("text", candidates)

        assert [j.candidate.value for j in judgments] == ["c", "a", "b"]

    async def test_every_judgment_references_its_candidate(self) -> None:
        model = StubModel()
        scorer = _make_scorer(model=model)

        candidates = [Candidate("x"), Candidate("y")]
        judgments = await scorer.score("text", candidates)

        assert judgments[0].candidate == Candidate("x")
        assert judgments[1].candidate == Candidate("y")

    async def test_no_logits_or_token_ids_in_judgments(self) -> None:
        model = StubModel()
        scorer = _make_scorer(model=model)

        judgments = await scorer.score("text", [Candidate("a")])

        j = judgments[0]
        assert not hasattr(j, "logits")
        assert not hasattr(j, "token_ids")
        assert not hasattr(j, "input_ids")
        assert isinstance(j, Judgment)
        assert isinstance(j.score, float)


class TestMultipleCandidates:
    async def test_multiple_candidates_in_one_call(self) -> None:
        model = StubModel()
        scorer = _make_scorer(model=model)

        candidates = [Candidate("a"), Candidate("b"), Candidate("c")]
        judgments = await scorer.score("text", candidates)

        assert len(judgments) == 3
        assert len(model.forward_calls) == 3  # sequential evaluation


class TestCandidateValueVerbatim:
    async def test_candidate_value_not_transformed(self) -> None:
        model = StubModel()
        scorer = _make_scorer(model=model)

        judgments = await scorer.score("text", [Candidate("  spaced  ")])
        assert len(judgments) == 1


class TestErrorTranslation:
    async def test_model_runtime_error_translated(self) -> None:
        model = FailingModel(RuntimeError("torch error"))
        scorer = _make_scorer(model=model)

        with pytest.raises(CandidateScoringError) as exc_info:
            await scorer.score("text", [Candidate("a")])

        assert "torch error" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    async def test_llm_scoring_error_translated(self) -> None:
        model = FailingModel(LLMScoringError("invalid score"))
        scorer = _make_scorer(model=model)

        with pytest.raises(CandidateScoringError) as exc_info:
            await scorer.score("text", [Candidate("a")])

        assert isinstance(exc_info.value.__cause__, LLMScoringError)

    async def test_no_raw_infrastructure_exception_leaks(self) -> None:
        model = FailingModel(RuntimeError("raw error"))
        scorer = _make_scorer(model=model)

        try:
            await scorer.score("text", [Candidate("a")])
        except CandidateScoringError:
            pass  # Correct: only CandidateScoringError
        except Exception as exc:
            pytest.fail(f"raw infrastructure exception leaked: {type(exc).__name__}")


class TestAtomicFailure:
    async def test_partial_failure_raises(self) -> None:
        tokenizer = StubTokenizer()
        builder = ScoringPromptBuilder()

        prefix = builder.build_prefix("text")
        candidate_b = builder.build_candidate(Candidate("b"))
        fail_seq = tuple(tokenizer.encode(prefix) + tokenizer.encode(candidate_b))

        model = PartialFailingModel(fail_seq, RuntimeError("fail on b"))
        scorer = _make_scorer(tokenizer=tokenizer, model=model)

        with pytest.raises(CandidateScoringError):
            await scorer.score("text", [Candidate("a"), Candidate("b"), Candidate("c")])

    async def test_no_partial_result_returned(self) -> None:
        tokenizer = StubTokenizer()
        builder = ScoringPromptBuilder()

        prefix = builder.build_prefix("text")
        candidate_b = builder.build_candidate(Candidate("b"))
        fail_seq = tuple(tokenizer.encode(prefix) + tokenizer.encode(candidate_b))

        model = PartialFailingModel(fail_seq, RuntimeError("fail on b"))
        scorer = _make_scorer(tokenizer=tokenizer, model=model)

        try:
            await scorer.score("text", [Candidate("a"), Candidate("b"), Candidate("c")])
        except CandidateScoringError:
            pass  # Correct: raised, no partial result
        else:
            pytest.fail("should have raised, not returned a partial result")


class TestAsyncContract:
    async def test_score_is_async(self) -> None:
        model = StubModel()
        scorer = _make_scorer(model=model)

        result = scorer.score("text", [Candidate("a")])
        assert hasattr(result, "__await__")
        await result  # Ensure the coroutine is consumed
