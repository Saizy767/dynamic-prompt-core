"""Opt-in real-model regression and benchmark tests.

Gated behind the ``gguf`` marker and ``DPC_GGUF_MODEL`` env var.  Skipped
by default — the default ``pytest`` run MUST NOT require a large model,
network access, or a compatible GPU.  Reports a clear skip reason when
prerequisites are missing.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gguf


@pytest.fixture()
def gguf_model_path() -> str:
    path = os.environ.get("DPC_GGUF_MODEL")
    if not path:
        pytest.skip("set DPC_GGUF_MODEL to a GGUF path to run real-model tests")
    if not os.path.isfile(path):
        pytest.skip(f"DPC_GGUF_MODEL={path} is not a readable file")
    return path


class TestRealModelRegression:
    def test_real_model_scores_are_finite(self, gguf_model_path: str) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
            LlamaCppBatchedCausalLMAdapter,
        )
        from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
            LogitScorer,
        )

        adapter = LlamaCppBatchedCausalLMAdapter(model_path=gguf_model_path, n_ctx=512)
        result = adapter.forward_batch([[1, 2, 3, 4]], [[1, 1, 1, 1]])
        scorer = LogitScorer()
        score = scorer.score(2, [0, 1], result.items[0])
        assert isinstance(score, float)


class TestRealModelBenchmark:
    def test_real_model_benchmark_runs(self, gguf_model_path: str) -> None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "tools"))
        import scoring_benchmark as bench

        from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
            LlamaCppBatchedCausalLMAdapter,
        )

        adapter = LlamaCppBatchedCausalLMAdapter(model_path=gguf_model_path, n_ctx=512)
        report = bench.run_benchmark(
            adapter, {"mode": "real", "model_path": gguf_model_path}
        )
        assert len(report["scenarios"]) == 6
