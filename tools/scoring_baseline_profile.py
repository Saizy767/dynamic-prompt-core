"""Produce a baseline scoring performance profile (machine-readable JSON).

Runs short-prompt and long-prompt scenarios (2 candidates each) through the
GGUF adapter and records per-stage timings, stage time shares, token counts,
candidate counts, model/backend/hardware configuration, cold-start latency,
and steady-state statistics.  The profile distinguishes ``eval`` from lazy
``eval_logits`` materialization and Python post-processing.

Default mode uses a deterministic mock model (no ``llama-cpp-python`` needed).
Pass ``--model <path>`` for a real-model profile on target hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamic_prompt_core.infrastructure.llm.scoring.instrumentation import (  # noqa: E402
    TIMING_FIELDS,
)
from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (  # noqa: E402
    LlamaCppBatchedCausalLMAdapter,
)

WARMUP = 2
REPEATS = 10
VOCAB = 151936


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
        n = len(tokens)
        base = [0.0] * self.n_vocab
        self.eval_logits = _MockLogits([list(base) for _ in range(n)])

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        return list(text)


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


def _run_scenario(
    adapter: LlamaCppBatchedCausalLMAdapter,
    token_len: int,
    candidates: int,
) -> dict[str, object]:
    input_ids = [[1] * token_len for _ in range(candidates)]
    attention_mask = [[1] * token_len for _ in range(candidates)]
    profiles = []
    first_latency = None
    for i in range(WARMUP + REPEATS):
        t0 = time.perf_counter()
        result = adapter.forward_batch(input_ids, attention_mask)
        latency = (time.perf_counter() - t0) * 1e3
        if i < WARMUP:
            continue
        if first_latency is None:
            first_latency = latency
        assert result.profile is not None
        profiles.append(result.profile)
    timing_means = {}
    timing_shares = {}
    for name in TIMING_FIELDS:
        values = [getattr(p, name) for p in profiles]
        mean = statistics.mean(values)
        timing_means[name] = mean
    total = sum(timing_means.values())
    for name in TIMING_FIELDS:
        timing_shares[name] = (timing_means[name] / total) if total > 0 else 0.0
    latencies = []
    for p in profiles:
        latencies.append(sum(getattr(p, n) for n in TIMING_FIELDS))
    return {
        "token_len": token_len,
        "candidates": candidates,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "first_request_latency_ms": first_latency,
        "steady_state_latency_mean_ms": statistics.mean(latencies),
        "steady_state_latency_median_ms": statistics.median(latencies),
        "timing_means_ms": timing_means,
        "timing_shares": timing_shares,
        "input_tokens": profiles[-1].input_tokens,
        "vocab_size": profiles[-1].vocab_size,
        "candidate_count": profiles[-1].candidate_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="GGUF model path (real-model mode)")
    parser.add_argument(
        "--output",
        default="logs/baseline_profile.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    if args.model:
        adapter = LlamaCppBatchedCausalLMAdapter(model_path=args.model, n_ctx=2048)
        model_config = {"mode": "real", "model_path": args.model}
    else:
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="mock.gguf", n_ctx=2048, _llama=_MockLlama(n_vocab=VOCAB)
        )
        model_config = {"mode": "mock", "model_path": None}

    scenarios = {
        "short_prompt_2_candidates": _run_scenario(adapter, token_len=8, candidates=2),
        "long_prompt_2_candidates": _run_scenario(adapter, token_len=64, candidates=2),
    }

    report = {
        "kind": "baseline_profile",
        "backend_version": _backend_version(),
        "hardware": _hardware_config(),
        "model_config": model_config,
        "adapter_metadata": adapter.describe(),
        "scenarios": scenarios,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Baseline profile written to {out}")


if __name__ == "__main__":
    main()
