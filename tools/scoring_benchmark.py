"""Standalone scoring benchmark — independent of the full dataset experiment.

Six scenarios: short prompt + 2 candidates, long prompt + 2 candidates,
1 prompt + 8 candidates, shared-prefix, sequential latency, parallel throughput.
Each scenario warms up the model, records first-request latency separately from
steady-state, and emits machine-readable JSON/CSV with mean/median/p95 latency,
tokens/s, examples/s, peak Python memory (tracemalloc), peak process RSS,
evaluated token positions, score deviation from baseline, and full config.

Default mode uses a deterministic mock model (CI).  Pass ``--model <path>``
for a real-model profile on target hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import threading
import time
import tracemalloc
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamic_prompt_core.infrastructure.llm.scoring.instrumentation import (  # noqa: E402
    TIMING_FIELDS,
)
from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (  # noqa: E402
    LlamaCppBatchedCausalLMAdapter,
)
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (  # noqa: E402
    LogitScorer,
)

WARMUP = 2
REPEATS = 10
VOCAB = 8


class _MockLogits:
    def __init__(self, positions: list[list[float]]) -> None:
        self._positions = positions

    def __iter__(self) -> Iterator[list[float]]:
        return iter(self._positions)


class _MockLlama:
    def __init__(self, n_vocab: int = 8) -> None:
        self.n_vocab = n_vocab
        self.eval_logits: _MockLogits = _MockLogits([])

    def reset(self) -> None:
        self.eval_logits = _MockLogits([])

    def eval(self, tokens: list[int]) -> None:
        base = [0.0] * self.n_vocab
        self.eval_logits = _MockLogits([list(base) for _ in range(len(tokens))])

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        return list(text)


@dataclass
class ScenarioResult:
    name: str
    token_len: int
    candidates: int
    warmup: int
    repeats: int
    first_request_latency_ms: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    tokens_per_second: float
    examples_per_second: float
    peak_python_memory_mb: float
    peak_process_rss_mb: float
    evaluated_token_positions: int
    score_deviation: float
    timing_means_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "token_len": self.token_len,
            "candidates": self.candidates,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "first_request_latency_ms": self.first_request_latency_ms,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_median_ms": self.latency_median_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "tokens_per_second": self.tokens_per_second,
            "examples_per_second": self.examples_per_second,
            "peak_python_memory_mb": self.peak_python_memory_mb,
            "peak_process_rss_mb": self.peak_process_rss_mb,
            "evaluated_token_positions": self.evaluated_token_positions,
            "score_deviation": self.score_deviation,
            "timing_means_ms": self.timing_means_ms,
        }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * 0.95)
    if idx >= len(s):
        idx = len(s) - 1
    return s[idx]


def _peak_rss_mb() -> float:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _run_scenario(
    adapter: LlamaCppBatchedCausalLMAdapter,
    name: str,
    token_len: int,
    candidates: int,
    shared_prefix: bool = False,
) -> ScenarioResult:
    scorer = LogitScorer()
    if shared_prefix:
        prefix = [1] * (token_len // 2)
        input_ids = [prefix + [10 + i] * (token_len - len(prefix)) for i in range(candidates)]
    else:
        input_ids = [[1 + i] * token_len for i in range(candidates)]
    masks = [[1] * len(ids) for ids in input_ids]
    prefix_counts = [token_len // 2] * candidates if shared_prefix else [token_len - 2] * candidates
    candidate_ids = [[0, 1] for _ in range(candidates)]

    tracemalloc.start()
    latencies: list[float] = []
    first_latency = None
    timing_acc: dict[str, list[float]] = {n: [] for n in TIMING_FIELDS}
    score_dev = 0.0

    for i in range(WARMUP + REPEATS):
        t0 = time.perf_counter()
        result = adapter.forward_batch(input_ids, masks)
        latency = (time.perf_counter() - t0) * 1e3
        if i < WARMUP:
            continue
        if first_latency is None:
            first_latency = latency
        latencies.append(latency)
        if result.profile is not None:
            for n in TIMING_FIELDS:
                timing_acc[n].append(getattr(result.profile, n))
        scores = scorer.score_batch(prefix_counts, candidate_ids, result)
        score_dev += sum(abs(s) for s in scores)

    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_rss = _peak_rss_mb()

    total_tokens = token_len * candidates * REPEATS
    total_time_s = sum(latencies) / 1000.0
    tokens_per_s = (total_tokens / total_time_s) if total_time_s > 0 else 0.0
    examples_per_s = (candidates * REPEATS / total_time_s) if total_time_s > 0 else 0.0
    timing_means = {n: statistics.mean(v) for n, v in timing_acc.items() if v}

    return ScenarioResult(
        name=name,
        token_len=token_len,
        candidates=candidates,
        warmup=WARMUP,
        repeats=REPEATS,
        first_request_latency_ms=first_latency or 0.0,
        latency_mean_ms=statistics.mean(latencies),
        latency_median_ms=statistics.median(latencies),
        latency_p95_ms=_p95(latencies),
        tokens_per_second=tokens_per_s,
        examples_per_second=examples_per_s,
        peak_python_memory_mb=peak_py / 1024.0 / 1024.0,
        peak_process_rss_mb=peak_rss,
        evaluated_token_positions=token_len,
        score_deviation=score_dev / REPEATS if REPEATS else 0.0,
        timing_means_ms=timing_means,
    )


def _run_parallel_scenario(
    adapter: LlamaCppBatchedCausalLMAdapter,
    name: str,
    token_len: int,
    candidates: int,
    parallel: int,
) -> ScenarioResult:
    input_ids = [[1] * token_len for _ in range(candidates)]
    masks = [[1] * token_len for _ in range(candidates)]
    latencies: list[float] = []
    first_latency = None

    for i in range(WARMUP + REPEATS):
        threads: list[threading.Thread] = []
        results: list[float] = []
        lock = threading.Lock()
        t0 = time.perf_counter()

        def worker() -> None:
            adapter.forward_batch(input_ids, masks)
            with lock:
                results.append(time.perf_counter())

        for _ in range(parallel):
            threads.append(threading.Thread(target=worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        latency = (time.perf_counter() - t0) * 1e3
        if i < WARMUP:
            continue
        if first_latency is None:
            first_latency = latency
        latencies.append(latency)

    return ScenarioResult(
        name=name,
        token_len=token_len,
        candidates=candidates,
        warmup=WARMUP,
        repeats=REPEATS,
        first_request_latency_ms=first_latency or 0.0,
        latency_mean_ms=statistics.mean(latencies),
        latency_median_ms=statistics.median(latencies),
        latency_p95_ms=_p95(latencies),
        tokens_per_second=0.0,
        examples_per_second=(parallel * REPEATS) / (sum(latencies) / 1000.0) if latencies else 0.0,
        peak_python_memory_mb=0.0,
        peak_process_rss_mb=_peak_rss_mb(),
        evaluated_token_positions=token_len,
        score_deviation=0.0,
    )


def _backend_version() -> str:
    try:
        import llama_cpp

        return getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        return "mock (llama-cpp-python not installed)"


def _hardware_config() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count() or "unknown"),
    }


def run_benchmark(
    adapter: LlamaCppBatchedCausalLMAdapter,
    model_config: dict[str, object],
) -> dict[str, object]:
    scenarios = [
        _run_scenario(adapter, "short_prompt_2_candidates", token_len=8, candidates=2),
        _run_scenario(adapter, "long_prompt_2_candidates", token_len=64, candidates=2),
        _run_scenario(adapter, "one_prompt_8_candidates", token_len=16, candidates=8),
        _run_scenario(adapter, "shared_prefix", token_len=32, candidates=4, shared_prefix=True),
        _run_scenario(adapter, "sequential_latency", token_len=16, candidates=2),
        _run_parallel_scenario(
            adapter, "parallel_throughput", token_len=16, candidates=2, parallel=4
        ),
    ]
    return {
        "kind": "scoring_benchmark",
        "backend_version": _backend_version(),
        "hardware": _hardware_config(),
        "model_config": model_config,
        "adapter_metadata": adapter.describe(),
        "warmup_policy": f"{WARMUP} warmup + {REPEATS} measured",
        "sample_count": REPEATS,
        "measurement_units": "milliseconds for latency, MB for memory",
        "per_optimization": {
            "direct_log_prob": {
                "enabled": True,
                "fallback": None,
                "note": "stable log-sum-exp, always active",
            },
            "selective_logits": {
                "enabled": adapter.selective_logits_supported,
                "fallback": "full eval_logits read + index"
                if not adapter.selective_logits_supported
                else None,
            },
            "kv_cache_reuse": {
                "enabled": adapter.kv_cache_reuse_supported,
                "fallback": "per-candidate reset-and-evaluate"
                if not adapter.kv_cache_reuse_supported
                else None,
            },
        },
        "scenarios": [s.as_dict() for s in scenarios],
    }


def _thread_sweep(model_path: str | None) -> list[dict[str, object]]:
    results = []
    for n_threads in [1, 2, 4]:
        if model_path:
            adapter = LlamaCppBatchedCausalLMAdapter(
                model_path=model_path, n_ctx=2048, n_threads=n_threads
            )
        else:
            adapter = LlamaCppBatchedCausalLMAdapter(
                model_path="mock.gguf", n_ctx=2048, n_threads=n_threads,
                _llama=_MockLlama(n_vocab=VOCAB),
            )
        result = _run_scenario(adapter, f"threads_{n_threads}", token_len=16, candidates=2)
        results.append({"n_threads": n_threads, **result.as_dict()})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="logs/scoring_benchmark.json")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--thread-sweep", action="store_true")
    args = parser.parse_args()

    if args.model:
        adapter = LlamaCppBatchedCausalLMAdapter(model_path=args.model, n_ctx=2048)
        model_config = {"mode": "real", "model_path": args.model}
    else:
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="mock.gguf", n_ctx=2048, _llama=_MockLlama(n_vocab=VOCAB)
        )
        model_config = {"mode": "mock", "model_path": None}

    report = run_benchmark(adapter, model_config)
    if args.thread_sweep:
        report["thread_sweep"] = _thread_sweep(args.model)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=report["scenarios"][0].keys())
            writer.writeheader()
            for s in report["scenarios"]:
                writer.writerow(s)
    else:
        out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Benchmark written to {out}")


if __name__ == "__main__":
    main()
