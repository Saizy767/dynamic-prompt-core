"""Metric invariance to judgment scores (candidate-scoring-validation spec).

Varying ``Judgment`` scores without changing the selected candidate must not
change the optimization metrics: metrics consume only ``predicted_decision``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dynamic_prompt_core.application.services.metrics.metrics import (
    MetricsConfig,
    compute_metrics,
    load_results,
)


def _make_row(
    rid: int,
    true_label: int,
    predicted_decision: int,
    judgment_scores: dict[str, float],
) -> dict[str, Any]:
    return {
        "id": rid,
        "text": f"sample text {rid}",
        "true_label": true_label,
        "predicted_decision": predicted_decision,
        "selected_candidate": str(predicted_decision),
        "judgment_scores": judgment_scores,
        "theses_raw": ["a thesis"],
        "theses_norm": ["a thesis"],
        "extract_status": "ok",
        "classify_status": "ok",
        "extract_latency_ms": 1.0,
        "classify_latency_ms": 1.0,
        "raw_extract": None,
    }


def _write_artifact(tmp_path: Path, rows: list[dict[str, Any]]) -> str:
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "results_run1_classify-v0_dev_20260101T000000Z.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _compute(tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = _write_artifact(tmp_path, rows)
    return compute_metrics(load_results(path), MetricsConfig(), path)


def test_score_change_without_selection_change_is_metric_invariant(
    tmp_path: Path,
) -> None:
    """Same predicted_decisions + different scores => identical metrics."""
    baseline_rows = [
        _make_row(0, 1, 1, {"0": -3.2, "1": -1.1}),
        _make_row(1, 0, 0, {"0": -0.5, "1": -4.7}),
        _make_row(2, 1, 1, {"0": -2.9, "1": -1.4}),
        _make_row(3, 0, 1, {"0": -1.9, "1": -1.8}),
    ]
    # Wildly different scores, identical selections (same ranking order).
    rescored_rows = [
        _make_row(0, 1, 1, {"0": 0.71, "1": 0.29}),
        _make_row(1, 0, 0, {"0": 0.99, "1": 0.01}),
        _make_row(2, 1, 1, {"0": 0.60, "1": 0.40}),
        _make_row(3, 0, 1, {"0": 0.505, "1": 0.495}),
    ]

    baseline = _compute(tmp_path / "a", baseline_rows)
    rescored = _compute(tmp_path / "b", rescored_rows)

    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
        assert baseline[key] == rescored[key], f"metric {key} changed with scores"
