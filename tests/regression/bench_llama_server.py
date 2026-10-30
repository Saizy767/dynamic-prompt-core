"""Benchmark script for the llama_server scoring backend (opt-in).

Measures initialization time, per-example classification latency, and
throughput against a running llama-server. Requires the
``LLAMA_SERVER_ENDPOINT`` env var (default ``http://127.0.0.1:8080``).

Usage::

    LLAMA_SERVER_ENDPOINT=http://127.0.0.1:8080 \
        python -m tests.regression.bench_llama_server
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification_rules import ClassificationRules
from dynamic_prompt_core.infrastructure.llm.classifier_factory import build_classifier
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    LlamaServerParams,
    ScorerBackendConfig,
)

_DEFAULT_RULES = ClassificationRules(rules=("rule a", "rule b", "rule c"))


async def run_benchmark(endpoint: str, n_examples: int = 20) -> dict[str, object]:
    cfg = ScorerBackendConfig(
        backend="llama_server",
        llama_server=LlamaServerParams(endpoint=endpoint),
    )

    t0 = time.perf_counter()
    classifier = build_classifier(cfg, [Candidate("0"), Candidate("1")])
    init_time = time.perf_counter() - t0

    latencies: list[float] = []
    for i in range(n_examples):
        text = f"example text number {i}"
        t1 = time.perf_counter()
        await classifier.classify(text, _DEFAULT_RULES)
        latencies.append(time.perf_counter() - t1)

    await classifier.aclose()

    return {
        "backend": "llama_server",
        "endpoint": endpoint,
        "init_time_s": init_time,
        "n_examples": n_examples,
        "mean_latency_s": sum(latencies) / len(latencies),
        "max_latency_s": max(latencies),
        "throughput_examples_per_s": n_examples / sum(latencies),
    }


if __name__ == "__main__":
    endpoint = os.environ.get("LLAMA_SERVER_ENDPOINT", "http://127.0.0.1:8080")
    results = asyncio.run(run_benchmark(endpoint))
    print(json.dumps(results, indent=2))
