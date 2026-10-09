"""Benchmark script for the GGUF backend (opt-in).

Measures initialization time, per-example inference latency, peak memory
usage, and throughput under supported concurrency for the GGUF backend.
Requires ``llama-cpp-python`` and a ``GGUF_TEST_MODEL`` env var.

Usage::

    GGUF_TEST_MODEL=models/classifier.gguf python -m tests.regression.bench_gguf

Does not set a hard performance target; records a results artifact*artifact.
"""
from __future__ import annotations

import json
import os
import time
import tracemalloc

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    GgufParams,
    ScorerBackendConfig,
)
from dynamic_prompt_core.infrastructure.llm.scoring.factory import (
    build_candidate_scorer,
)


def run_benchmark(model_path: str, n_examples: int = 20) -> dict[str, object]:
    import asyncio

    cfg = ScorerBackendConfig(
        backend="gguf",
        model_path=model_path,
        gguf=GgufParams(n_ctx=512, seed=42),
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    scorer = build_candidate_scorer(cfg)
    init_time = time.perf_counter() - t0

    candidates = [Candidate("yes"), Candidate("no")]
    latencies: list[float] = []
    for i in range(n_examples):
        text = f"example text number {i}"
        t1 = time.perf_counter()
        asyncio.run(scorer.score(text, candidates))
        latencies.append(time.perf_counter() - t1)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "backend": "gguf",
        "model_path": model_path,
        "init_time_s": init_time,
        "n_examples": n_examples,
        "mean_latency_s": sum(latencies) / len(latencies),
        "max_latency_s": max(latencies),
        "throughput_examples_per_s": n_examples / sum(latencies),
        "peak_memory_mb": peak / (1024 * 1024),
    }


if __name__ == "__main__":
    model = os.environ.get("GGUF_TEST_MODEL")
    if not model:
        raise SystemExit("set GGUF_TEST_MODEL env var to the GGUF model path")
    results = run_benchmark(model)
    print(json.dumps(results, indent=2))
