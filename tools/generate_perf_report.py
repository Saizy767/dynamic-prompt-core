"""Generate a performance report from benchmark results.

Reads the baseline profile and benchmark JSON, produces a machine-readable
``performance_report.json`` and a human-readable ``performance_report.md``
under ``docs/`` or ``logs/``.  Documents backend limitations, capability-probe
results, hardware/software config, and measured speedup (or its absence).
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (  # noqa: E402
    LlamaCppBatchedCausalLMAdapter,
)


def _backend_version() -> str:
    try:
        import llama_cpp

        return getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        return "mock (llama-cpp-python not installed)"


def main() -> None:
    logs = Path(__file__).resolve().parents[1] / "logs"
    benchmark_path = logs / "scoring_benchmark.json"
    baseline_path = logs / "baseline_profile.json"

    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {}
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    _mock = type(
        "M", (),
        {"n_vocab": 8, "reset": lambda s: None, "eval": lambda s, t: None, "eval_logits": []},
    )
    adapter = LlamaCppBatchedCausalLMAdapter(
        model_path="mock.gguf", n_ctx=64, _llama=_mock(),
    )

    report = {
        "kind": "performance_report",
        "backend_version": _backend_version(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python_version": platform.python_version(),
        },
        "baseline_profile": baseline.get("scenarios", {}),
        "benchmark_scenarios": benchmark.get("scenarios", []),
        "per_optimization": benchmark.get("per_optimization", {}),
        "backend_limitations": {
            "selective_logits": {
                "supported": adapter.selective_logits_supported,
                "status": "disabled — no verified per-position API in llama-cpp-python 0.3.x",
                "fallback": "full eval_logits read + index",
            },
            "kv_cache_reuse": {
                "supported": adapter.kv_cache_reuse_supported,
                "status": "disabled — no verified KV-cache save/restore in llama-cpp-python 0.3.x",
                "fallback": "per-candidate reset-and-evaluate",
            },
            "thread_safety": {
                "proven": False,
                "status": "serialization lock retained; not removed without proven thread-safety",
            },
        },
        "dominant_bottleneck": (
            "eval + logits_read (forward pass) — see baseline_profile timing_shares"
        ),
        "measured_speedup": (
            "not claimed — optimizations are gated; direct log-prob is active, "
            "selective extraction and KV-cache reuse fall back to baseline"
        ),
        "note": "No speedup is claimed that is not supported by measurements. "
            "Selective extraction and KV-cache reuse are disabled until their "
            "capabilities are verified on the installed backend.",
    }

    out_json = logs / "performance_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Performance Report: GGUF Scoring Optimization",
        "",
        f"**Backend:** {_backend_version()}",
        f"**Platform:** {platform.platform()}",
        "",
        "## Optimizations",
        "",
        "| Optimization | Enabled | Fallback |",
        "|---|---|---|",
        "| Direct log-prob (stable log-sum-exp) | yes | n/a |",
        f"| Selective logit extraction | {adapter.selective_logits_supported} |"
        " full eval_logits read |",
        f"| KV-cache prefix reuse | {adapter.kv_cache_reuse_supported} |"
        " per-candidate reset+eval |",
        "",
        "## Backend Limitations",
        "",
        "- **Selective logits:** no verified per-position API in"
        " llama-cpp-python 0.3.x; falls back to full read.",
        "- **KV-cache reuse:** no verified save/restore mechanism;"
        " falls back to per-candidate reset+eval.",
        "- **Thread safety:** serialization lock retained;"
        " not removed without proven thread-safety.",
        "",
        "## Bottleneck",
        "",
        "The baseline profile separates `eval_ms` from `logits_read_ms` (including lazy",
        "materialization) and Python post-processing. See `logs/baseline_profile.json`",
        "for per-stage timing shares.",
        "",
        "## Measured Speedup",
        "",
        "No speedup is claimed that is not supported by measurements. The direct",
        "log-probability optimization is active. Selective extraction and KV-cache",
        "reuse are disabled until their capabilities are verified on the installed",
        "backend; their fallbacks preserve baseline-equivalent scores.",
        "",
    ]
    out_md = logs / "performance_report.md"
    out_md.write_text("\n".join(lines))

    print(f"Report written to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
