"""Tests for the standalone scoring benchmark module."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "tools"))

import scoring_benchmark as bench  # noqa: E402

from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (  # noqa: E402
    LlamaCppBatchedCausalLMAdapter,
)

REQUIRED_FIELDS = {
    "name",
    "token_len",
    "candidates",
    "warmup",
    "repeats",
    "first_request_latency_ms",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "tokens_per_second",
    "examples_per_second",
    "peak_python_memory_mb",
    "peak_process_rss_mb",
    "evaluated_token_positions",
    "score_deviation",
    "timing_means_ms",
}

EXPECTED_SCENARIOS = {
    "short_prompt_2_candidates",
    "long_prompt_2_candidates",
    "one_prompt_8_candidates",
    "shared_prefix",
    "sequential_latency",
    "parallel_throughput",
}


def _mock_adapter() -> LlamaCppBatchedCausalLMAdapter:
    return LlamaCppBatchedCausalLMAdapter(
        model_path="mock.gguf", n_ctx=64, _llama=bench._MockLlama(n_vocab=8)
    )


class TestBenchmarkOutput:
    def test_all_six_scenarios_present(self) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        names = {s["name"] for s in report["scenarios"]}
        assert names == EXPECTED_SCENARIOS

    def test_all_required_fields_present(self) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        for scenario in report["scenarios"]:
            assert REQUIRED_FIELDS <= set(scenario.keys()), (
                f"missing fields in {scenario['name']}"
            )

    def test_fields_are_correctly_typed(self) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        for s in report["scenarios"]:
            assert isinstance(s["name"], str)
            assert isinstance(s["latency_mean_ms"], float)
            assert isinstance(s["latency_median_ms"], float)
            assert isinstance(s["latency_p95_ms"], float)
            assert isinstance(s["tokens_per_second"], float)
            assert isinstance(s["peak_python_memory_mb"], float)
            assert isinstance(s["evaluated_token_positions"], int)
            assert isinstance(s["candidates"], int)

    def test_first_request_separated_from_steady_state(self) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        for s in report["scenarios"]:
            assert "first_request_latency_ms" in s
            assert "latency_mean_ms" in s

    def test_json_export_parseable(self, tmp_path: Path) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        out = tmp_path / "bench.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        loaded = json.loads(out.read_text())
        assert len(loaded["scenarios"]) == 6


class TestPerOptimizationReporting:
    def test_per_optimization_section_present(self) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        assert "per_optimization" in report
        po = report["per_optimization"]
        assert "direct_log_prob" in po
        assert "selective_logits" in po
        assert "kv_cache_reuse" in po

    def test_disabled_optimizations_have_fallback(self) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        po = report["per_optimization"]
        assert po["selective_logits"]["enabled"] is False
        assert po["selective_logits"]["fallback"] is not None
        assert po["kv_cache_reuse"]["enabled"] is False
        assert po["kv_cache_reuse"]["fallback"] is not None


class TestWarmupPolicy:
    def test_warmup_recorded(self) -> None:
        report = bench.run_benchmark(_mock_adapter(), {"mode": "mock"})
        for s in report["scenarios"]:
            assert s["warmup"] > 0
            assert s["repeats"] > 0
