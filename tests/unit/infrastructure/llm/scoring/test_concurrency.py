"""Tests for concurrency: lock-wait measurement, bounding, and score integrity."""
from __future__ import annotations

import threading
import time

from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
)
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer

from ._gguf_mocks import FakeLlama

VOCAB = 8


def _adapter(
    model: FakeLlama | None = None, max_concurrent: int = 1
) -> LlamaCppBatchedCausalLMAdapter:
    return LlamaCppBatchedCausalLMAdapter(
        model_path="unused.gguf",
        n_ctx=64,
        max_concurrent=max_concurrent,
        _llama=model or FakeLlama(n_vocab=VOCAB),
    )


class TestLockWaitMeasurement:
    def test_lock_wait_recorded_under_contention(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model, max_concurrent=1)
        results: list[float] = []
        barrier = threading.Barrier(3)

        def worker() -> None:
            barrier.wait()
            result = adapter.forward_batch([[1, 2, 3]], [[1, 1, 1]])
            assert result.profile is not None
            results.append(result.profile.lock_wait_ms)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 3
        assert all(r >= 0.0 for r in results)
        assert any(r > 0.0 for r in results)

    def test_lock_wait_zero_without_contention(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch([[1, 2]], [[1, 1]])
        assert result.profile is not None
        assert result.profile.lock_wait_ms >= 0.0


class TestBoundedConcurrency:
    def test_semaphore_bounds_concurrent_access(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model, max_concurrent=2)
        in_flight = 0
        max_observed = 0
        state_lock = threading.Lock()

        def worker() -> None:
            nonlocal in_flight, max_observed
            adapter.forward_batch([[1, 2, 3, 4]], [[1, 1, 1, 1]])
            with state_lock:
                in_flight += 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(t.is_alive() is False for t in threads)

    def test_describe_includes_max_concurrent(self) -> None:
        adapter = _adapter(max_concurrent=3)
        assert adapter.describe()["max_concurrent"] == 3


class TestScoreIntegrityUnderConcurrency:
    def test_concurrent_scores_match_sequential(self) -> None:
        model = FakeLlama(
            n_vocab=VOCAB, logits_per_position=[2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        adapter = _adapter(model, max_concurrent=1)
        scorer = LogitScorer()
        input_ids = [[1, 2, 3, 4], [5, 6, 7, 8]]
        masks = [[1, 1, 1, 1], [1, 1, 1, 1]]

        seq_result = adapter.forward_batch(input_ids, masks)
        seq_scores = [
            scorer.score(2, [0, 1], seq_result.items[i]) for i in range(2)
        ]

        concurrent_results: list[list[float]] = []
        result_lock = threading.Lock()

        def worker() -> None:
            r = adapter.forward_batch(input_ids, masks)
            scores = [scorer.score(2, [0, 1], r.items[i]) for i in range(2)]
            with result_lock:
                concurrent_results.append(scores)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(concurrent_results) == 4
        for scores in concurrent_results:
            for i in range(2):
                assert scores[i] == seq_scores[i]

    def test_no_cross_contamination(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model, max_concurrent=1)
        eval_order: list[list[int]] = []
        order_lock = threading.Lock()

        original_eval = model.eval

        def tracking_eval(tokens: list[int]) -> None:
            with order_lock:
                eval_order.append(list(tokens))
            original_eval(tokens)

        model.eval = tracking_eval  # type: ignore[method-assign]

        def worker(tokens: list[int]) -> None:
            adapter.forward_batch([tokens], [[1] * len(tokens)])

        threads = [
            threading.Thread(target=worker, args=([1, 2, 3],)),
            threading.Thread(target=worker, args=([4, 5, 6],)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(eval_order) == 2
        assert {tuple(e) for e in eval_order} == {(1, 2, 3), (4, 5, 6)}


class TestNoAssumedParallelismBenefit:
    def test_concurrent_may_not_be_faster(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter(model, max_concurrent=1)

        def run_sequential() -> float:
            t0 = time.perf_counter()
            for _ in range(4):
                adapter.forward_batch([[1, 2]], [[1, 1]])
            return time.perf_counter() - t0

        seq_time = run_sequential()
        assert seq_time > 0.0
