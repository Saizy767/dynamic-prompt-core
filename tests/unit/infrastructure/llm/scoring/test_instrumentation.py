"""Unit tests for per-stage scoring instrumentation."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import GGUFProviderError
from dynamic_prompt_core.infrastructure.llm.scoring.instrumentation import (
    COUNTER_FIELDS,
    TIMING_FIELDS,
    ScoringProfile,
    timed_stage,
)
from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
)

from ._gguf_mocks import FakeLlama

VOCAB = 8


def _adapter(model: FakeLlama | None = None) -> LlamaCppBatchedCausalLMAdapter:
    return LlamaCppBatchedCausalLMAdapter(
        model_path="unused.gguf",
        n_ctx=64,
        _llama=model or FakeLlama(n_vocab=VOCAB),
    )


class LazyCountingLogits:
    """Simulates lazy materialization on iteration (``list(eval_logits)``)."""

    def __init__(self, positions: list[list[float]]) -> None:
        self._positions = positions
        self.materialized = False

    def __iter__(self) -> Iterator[list[float]]:
        self.materialized = True
        return iter(self._positions)


class LazyFakeLlama(FakeLlama):
    """``FakeLlama`` whose ``eval_logits`` materializes lazily on iteration."""

    def __init__(
        self,
        n_vocab: int = 8,
        logits_per_position: list[float] | None = None,
    ) -> None:
        super().__init__(n_vocab, logits_per_position)
        self.eval_logits: LazyCountingLogits = LazyCountingLogits([])

    def reset(self) -> None:
        self.reset_count += 1
        self.eval_logits = LazyCountingLogits([])

    def eval(self, tokens: list[int]) -> None:
        self.eval_calls.append(list(tokens))
        n = len(tokens)
        base = self._logits_per_position or [0.0] * self.n_vocab
        self.eval_logits = LazyCountingLogits([list(base) for _ in range(n)])


class TestScoringProfileSchema:
    def test_all_timing_fields_present(self) -> None:
        profile = ScoringProfile()
        for name in TIMING_FIELDS:
            assert hasattr(profile, name)
            assert getattr(profile, name) == 0.0

    def test_all_counter_fields_present(self) -> None:
        profile = ScoringProfile()
        for name in COUNTER_FIELDS:
            assert hasattr(profile, name)
            assert getattr(profile, name) == 0

    def test_as_dict_has_all_fields(self) -> None:
        d = ScoringProfile().as_dict()
        assert set(d.keys()) == set(TIMING_FIELDS) | set(COUNTER_FIELDS)


class TestTimedStage:
    def test_records_non_negative_ms(self) -> None:
        profile = ScoringProfile()
        with timed_stage(profile, "eval_ms"):
            pass
        assert profile.eval_ms >= 0.0

    def test_accumulates_across_uses(self) -> None:
        profile = ScoringProfile()
        with timed_stage(profile, "reset_ms"):
            pass
        first = profile.reset_ms
        with timed_stage(profile, "reset_ms"):
            pass
        assert profile.reset_ms >= first

    def test_records_even_on_exception(self) -> None:
        profile = ScoringProfile()
        with pytest.raises(RuntimeError):
            with timed_stage(profile, "eval_ms"):
                raise RuntimeError("boom")
        assert profile.eval_ms >= 0.0


class TestForwardBatchProfile:
    def test_profile_attached_to_result(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch([[1, 2]], [[1, 1]])
        assert result.profile is not None

    def test_all_timings_non_negative(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch([[1, 2, 3]], [[1, 1, 1]])
        p = result.profile
        assert p is not None
        for name in TIMING_FIELDS:
            assert getattr(p, name) >= 0.0, f"{name} is negative"

    def test_counters_populated(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch([[1, 2], [3, 4]], [[1, 1], [1, 1]])
        p = result.profile
        assert p is not None
        assert p.input_tokens == 4
        assert p.vocab_size == VOCAB
        assert p.candidate_count == 2

    def test_no_user_content_in_profile_dict(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch([[1, 2]], [[1, 1]])
        p = result.profile
        assert p is not None
        d = p.as_dict()
        allowed = set(TIMING_FIELDS) | set(COUNTER_FIELDS)
        assert set(d.keys()) == allowed

    def test_eval_separated_from_logits_read(self) -> None:
        model = LazyFakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model)
        result = adapter.forward_batch([[1, 2, 3]], [[1, 1, 1]])
        p = result.profile
        assert p is not None
        assert p.eval_ms >= 0.0
        assert p.logits_read_ms >= 0.0
        assert model.eval_logits.materialized

    def test_multiple_items_accumulate(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model)
        result = adapter.forward_batch([[1], [2], [3]], [[1], [1], [1]])
        p = result.profile
        assert p is not None
        assert p.candidate_count == 3
        assert model.reset_count == 3


class TestExceptionPreservation:
    def test_completed_timings_available_on_exception(self) -> None:
        bad = [float("nan")] + [0.0] * (VOCAB - 1)
        model = FakeLlama(n_vocab=VOCAB, logits_per_position=bad)
        adapter = _adapter(model)
        with pytest.raises(GGUFProviderError, match="NaN"):
            adapter.forward_batch([[1, 2]], [[1, 1]])
        p = adapter.last_profile
        assert p is not None
        assert p.eval_ms >= 0.0
        assert p.logits_read_ms >= 0.0
        assert p.logits_convert_ms >= 0.0

    def test_instrumentation_does_not_change_error_type(self) -> None:
        model = FakeLlama(n_vocab=VOCAB, logits_per_position=[0.0] * (VOCAB + 1))
        adapter = _adapter(model)
        with pytest.raises(GGUFProviderError, match="vocabulary dimension"):
            adapter.forward_batch([[1]], [[1]])

    def test_inf_logits_still_rejected(self) -> None:
        bad = [float("inf")] + [0.0] * (VOCAB - 1)
        model = FakeLlama(n_vocab=VOCAB, logits_per_position=bad)
        adapter = _adapter(model)
        with pytest.raises(GGUFProviderError, match="NaN"):
            adapter.forward_batch([[1]], [[1]])
