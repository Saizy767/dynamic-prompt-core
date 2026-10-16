"""Unit tests for stage1-metrics validation hardening."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from dynamic_prompt_core.application.services.metrics.metrics import (
    MetricsConfig,
    MetricsError,
    _print_summary,
    compare_versions,
    compute_metrics,
    load_results,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _make_row(
    rid: int,
    true_label: int = 0,
    predicted_decision: int | None = 0,
    classify_status: str = "ok",
    text: str = "sample text here",
    theses_norm: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": rid,
        "text": text,
        "true_label": true_label,
        "predicted_decision": predicted_decision,
        "classify_status": classify_status,
        "theses_norm": theses_norm or [],
    }


def _write_artifact(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    name: str = "results_testrun_v0_dev_20260101T000000Z.jsonl",
) -> str:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def _compute(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    name: str = "results_testrun_v0_dev_20260101T000000Z.jsonl",
) -> dict[str, Any]:
    path = _write_artifact(tmp_path, rows, name)
    loaded = load_results(path)
    config = MetricsConfig()
    return compute_metrics(loaded, config, path)


# --------------------------------------------------------------------------- #
#  1.2  Characterization tests – happy path
# --------------------------------------------------------------------------- #
class TestHappyPath:
    """Characterization tests that pass against the current implementation."""

    def test_happy_path_metrics_are_floats(self, tmp_path: Path) -> None:
        rows = [_make_row(i, true_label=i % 2, predicted_decision=i % 2) for i in range(10)]
        m = _compute(tmp_path, rows)
        assert isinstance(m["accuracy"], float)
        assert isinstance(m["precision"]["0"], float)
        assert isinstance(m["precision"]["1"], float)
        assert isinstance(m["recall"]["0"], float)
        assert isinstance(m["recall"]["1"], float)
        assert isinstance(m["f1"]["macro"], float)
        assert m["counts"]["ok"] == 10

    def test_happy_path_perfect_accuracy(self, tmp_path: Path) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=0),
            _make_row(1, true_label=1, predicted_decision=1),
            _make_row(2, true_label=0, predicted_decision=0),
            _make_row(3, true_label=1, predicted_decision=1),
        ]
        m = _compute(tmp_path, rows)
        assert m["accuracy"] == 1.0


# --------------------------------------------------------------------------- #
#  1.3  Failing tests for desired new behavior
# --------------------------------------------------------------------------- #
class TestNewBehaviorDefects:
    """Tests that expose defects in the current implementation."""

    def test_invalid_true_label_raises(self, tmp_path: Path) -> None:
        rows = [_make_row(0, true_label=2, predicted_decision=0)]
        with pytest.raises(MetricsError, match="true_label"):
            _compute(tmp_path, rows)

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        rows = [
            _make_row(42, true_label=0, predicted_decision=0),
            _make_row(42, true_label=1, predicted_decision=1),
        ]
        with pytest.raises(MetricsError, match="duplicate"):
            _compute(tmp_path, rows)

    def test_missing_prediction_raises(self, tmp_path: Path) -> None:
        rows = [_make_row(0, true_label=0, predicted_decision=None, classify_status="ok")]
        with pytest.raises(MetricsError, match="predicted_decision"):
            _compute(tmp_path, rows)

    def test_all_failed_returns_none_metrics(self, tmp_path: Path) -> None:
        rows = [
            _make_row(i, true_label=0, predicted_decision=None, classify_status="failed")
            for i in range(5)
        ]
        m = _compute(tmp_path, rows)
        assert m["metrics_available"] is False
        assert m["accuracy"] is None

    def test_accuracy_over_total_present(self, tmp_path: Path) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=0),
            _make_row(1, true_label=1, predicted_decision=1),
        ]
        m = _compute(tmp_path, rows)
        assert "accuracy_over_total" in m
        assert isinstance(m["accuracy_over_total"], float)


# --------------------------------------------------------------------------- #
#  2.  Label value validation
# --------------------------------------------------------------------------- #
class TestLabelValidation:
    def test_invalid_true_label_rejected(self, tmp_path: Path) -> None:
        rows = [_make_row(0, true_label=2, predicted_decision=0)]
        with pytest.raises(MetricsError, match="true_label"):
            _compute(tmp_path, rows)

    def test_invalid_true_label_string_rejected(self, tmp_path: Path) -> None:
        rows = [_make_row(0, true_label="yes", predicted_decision=0)]
        with pytest.raises(MetricsError, match="true_label"):
            _compute(tmp_path, rows)

    def test_invalid_prediction_rejected(self, tmp_path: Path) -> None:
        rows = [_make_row(0, true_label=0, predicted_decision="yes")]
        with pytest.raises(MetricsError, match="predicted_decision"):
            _compute(tmp_path, rows)

    def test_invalid_prediction_out_of_range(self, tmp_path: Path) -> None:
        rows = [_make_row(0, true_label=0, predicted_decision=5)]
        with pytest.raises(MetricsError, match="predicted_decision"):
            _compute(tmp_path, rows)

    def test_failed_null_prediction_accepted(self, tmp_path: Path) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=None, classify_status="failed"),
            _make_row(1, true_label=1, predicted_decision=1, classify_status="ok"),
        ]
        m = _compute(tmp_path, rows)
        assert m["counts"]["failed"] == 1
        assert m["counts"]["ok"] == 1


# --------------------------------------------------------------------------- #
#  3.  Duplicate ID detection
# --------------------------------------------------------------------------- #
class TestDuplicateId:
    def test_duplicate_id_rejected(self, tmp_path: Path) -> None:
        rows = [
            _make_row(42, true_label=0, predicted_decision=0),
            _make_row(42, true_label=1, predicted_decision=1),
        ]
        with pytest.raises(MetricsError, match="duplicate"):
            _compute(tmp_path, rows)

    def test_unique_ids_accepted(self, tmp_path: Path) -> None:
        rows = [_make_row(i, true_label=i % 2, predicted_decision=i % 2) for i in range(10)]
        m = _compute(tmp_path, rows)
        assert m["counts"]["ok"] == 10


# --------------------------------------------------------------------------- #
#  4.  Prediction presence for ok records
# --------------------------------------------------------------------------- #
class TestPredictionPresence:
    def test_missing_prediction_rejected(self, tmp_path: Path) -> None:
        rows = [_make_row(0, true_label=0, predicted_decision=None, classify_status="ok")]
        with pytest.raises(MetricsError, match="predicted_decision"):
            _compute(tmp_path, rows)

    def test_failed_missing_prediction_accepted(self, tmp_path: Path) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=None, classify_status="failed"),
        ]
        path = _write_artifact(tmp_path, rows)
        loaded = load_results(path)
        assert len(loaded) == 1


# --------------------------------------------------------------------------- #
#  5.  Empty and all-failed set handling
# --------------------------------------------------------------------------- #
class TestEmptySet:
    _NONE_FIELDS = (
        "accuracy",
        "accuracy_over_total",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "prediction_distribution",
        "true_distribution",
        "bias",
        "groups",
    )

    def test_all_failed(self, tmp_path: Path) -> None:
        rows = [
            _make_row(i, true_label=0, predicted_decision=None, classify_status="failed")
            for i in range(5)
        ]
        m = _compute(tmp_path, rows)
        assert m["metrics_available"] is False
        for field in self._NONE_FIELDS:
            assert m[field] is None, f"{field} should be None"
        assert m["counts"]["total"] == 5
        assert m["counts"]["failed"] == 5

    def test_empty_artifact(self, tmp_path: Path) -> None:
        m = _compute(tmp_path, [])
        assert m["metrics_available"] is False
        for field in self._NONE_FIELDS:
            assert m[field] is None, f"{field} should be None"
        assert m["counts"]["total"] == 0

    def test_mixed_ok_failed(self, tmp_path: Path) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=0, classify_status="ok"),
            _make_row(1, true_label=1, predicted_decision=None, classify_status="failed"),
        ]
        m = _compute(tmp_path, rows)
        assert m["metrics_available"] is True
        assert isinstance(m["accuracy"], float)
        assert isinstance(m["precision"], dict)
        assert isinstance(m["f1"], dict)
        assert isinstance(m["confusion_matrix"], dict)


# --------------------------------------------------------------------------- #
#  6.  Explicit metric denominator
# --------------------------------------------------------------------------- #
class TestDenominator:
    def test_accuracy_over_total(self, tmp_path: Path) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=0),
            _make_row(1, true_label=1, predicted_decision=1),
            _make_row(2, true_label=0, predicted_decision=1),
        ]
        m = _compute(tmp_path, rows)
        assert "accuracy_over_total" in m
        assert m["accuracy_over_total"] == pytest.approx(2 / 3)

    def test_denominator_fields(self, tmp_path: Path) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=0, classify_status="ok"),
            _make_row(1, true_label=1, predicted_decision=None, classify_status="failed"),
        ]
        m = _compute(tmp_path, rows)
        assert m["denominator"] == {"ok_count": 1, "total_count": 2}
        assert m["denominator_excludes_failures"] is True

    def test_denominator_with_failures(self, tmp_path: Path) -> None:
        rows = []
        rid = 0
        for _ in range(150):
            rows.append(_make_row(rid, true_label=0, predicted_decision=0))
            rid += 1
        for _ in range(30):
            rows.append(_make_row(rid, true_label=1, predicted_decision=0))
            rid += 1
        for _ in range(20):
            rows.append(
                _make_row(rid, true_label=0, predicted_decision=None, classify_status="failed")
            )
            rid += 1
        m = _compute(tmp_path, rows)
        assert m["accuracy"] == pytest.approx(150 / 180)
        assert m["accuracy_over_total"] == pytest.approx(150 / 200)
        assert m["denominator"] == {"ok_count": 180, "total_count": 200}
        assert m["denominator_excludes_failures"] is True

    def test_denominator_no_failures(self, tmp_path: Path) -> None:
        rows = []
        rid = 0
        for _ in range(150):
            rows.append(_make_row(rid, true_label=0, predicted_decision=0))
            rid += 1
        for _ in range(50):
            rows.append(_make_row(rid, true_label=1, predicted_decision=0))
            rid += 1
        m = _compute(tmp_path, rows)
        assert m["accuracy"] == pytest.approx(150 / 200)
        assert m["accuracy_over_total"] == pytest.approx(150 / 200)
        assert m["denominator_excludes_failures"] is False


# --------------------------------------------------------------------------- #
#  7.  Precision / recall / F1 cross-checks
# --------------------------------------------------------------------------- #
class TestPrfCrossCheck:
    def test_balanced(self, tmp_path: Path) -> None:
        rows = []
        rid = 0
        for _ in range(50):
            rows.append(_make_row(rid, true_label=1, predicted_decision=1))
            rid += 1
        for _ in range(50):
            rows.append(_make_row(rid, true_label=0, predicted_decision=0))
            rid += 1
        m = _compute(tmp_path, rows)
        assert m["precision"]["0"] == pytest.approx(1.0)
        assert m["precision"]["1"] == pytest.approx(1.0)
        assert m["recall"]["0"] == pytest.approx(1.0)
        assert m["recall"]["1"] == pytest.approx(1.0)
        assert m["f1"]["0"] == pytest.approx(1.0)
        assert m["f1"]["1"] == pytest.approx(1.0)
        assert m["f1"]["macro"] == pytest.approx(1.0)
        assert m["f1"]["weighted"] == pytest.approx(1.0)

    def test_single_class(self, tmp_path: Path) -> None:
        rows = []
        rid = 0
        for _ in range(50):
            rows.append(_make_row(rid, true_label=1, predicted_decision=1))
            rid += 1
        for _ in range(50):
            rows.append(_make_row(rid, true_label=0, predicted_decision=1))
            rid += 1
        m = _compute(tmp_path, rows)
        assert m["precision"]["0"] == pytest.approx(0.0)
        assert m["recall"]["0"] == pytest.approx(0.0)
        assert m["f1"]["0"] == pytest.approx(0.0)
        assert m["precision"]["1"] == pytest.approx(0.5)
        assert m["recall"]["1"] == pytest.approx(1.0)
        assert m["f1"]["1"] == pytest.approx(2 / 3)

        from sklearn.metrics import precision_recall_fscore_support

        y_true = [r["true_label"] for r in rows]
        y_pred = [r["predicted_decision"] for r in rows]
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        assert m["precision"]["0"] == pytest.approx(float(p[0]))
        assert m["precision"]["1"] == pytest.approx(float(p[1]))
        assert m["recall"]["0"] == pytest.approx(float(r[0]))
        assert m["recall"]["1"] == pytest.approx(float(r[1]))
        assert m["f1"]["0"] == pytest.approx(float(f[0]))
        assert m["f1"]["1"] == pytest.approx(float(f[1]))

    def test_imbalanced(self, tmp_path: Path) -> None:
        rows = []
        rid = 0
        for _ in range(170):
            rows.append(_make_row(rid, true_label=0, predicted_decision=0))
            rid += 1
        for _ in range(10):
            rows.append(_make_row(rid, true_label=0, predicted_decision=1))
            rid += 1
        for _ in range(15):
            rows.append(_make_row(rid, true_label=1, predicted_decision=1))
            rid += 1
        for _ in range(5):
            rows.append(_make_row(rid, true_label=1, predicted_decision=0))
            rid += 1
        m = _compute(tmp_path, rows)
        macro_f1 = (m["f1"]["0"] + m["f1"]["1"]) / 2
        assert m["f1"]["macro"] == pytest.approx(macro_f1)
        total_support = 180 + 20
        weighted_f1 = (m["f1"]["0"] * 180 + m["f1"]["1"] * 20) / total_support
        assert m["f1"]["weighted"] == pytest.approx(weighted_f1)

    def test_zero_division(self, tmp_path: Path) -> None:
        rows = []
        rid = 0
        for _ in range(10):
            rows.append(_make_row(rid, true_label=1, predicted_decision=1))
            rid += 1
        m = _compute(tmp_path, rows)
        assert m["precision"]["0"] == 0.0
        assert m["recall"]["0"] == 0.0
        assert m["f1"]["0"] == 0.0
        assert not math.isnan(m["precision"]["0"])
        assert not math.isnan(m["f1"]["0"])


# --------------------------------------------------------------------------- #
#  8.  compare_versions and CLI integration
# --------------------------------------------------------------------------- #
class TestCompareVersions:
    def test_compare_none_metrics(self, tmp_path: Path) -> None:
        failed_rows = [
            _make_row(i, true_label=0, predicted_decision=None, classify_status="failed")
            for i in range(5)
        ]
        ok_rows = [
            _make_row(0, true_label=0, predicted_decision=0),
            _make_row(1, true_label=1, predicted_decision=1),
        ]
        path0 = _write_artifact(tmp_path, failed_rows, "results_run_v0_dev_20260101T000000Z.jsonl")
        path1 = _write_artifact(tmp_path, ok_rows, "results_run_v1_dev_20260101T000000Z.jsonl")
        config = MetricsConfig()
        result = compare_versions(path0, path1, config)
        assert result["deltas"]["accuracy"] is None
        assert result["deltas"]["macro_f1"] is None
        assert result["deltas"]["minority_f1"] is None
        assert result["v0"]["metrics_available"] is False
        assert result["v1"]["metrics_available"] is True

    def test_print_summary_metrics_available(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=0),
            _make_row(1, true_label=1, predicted_decision=1),
        ]
        m = _compute(tmp_path, rows)
        _print_summary(m)
        out = capsys.readouterr().out
        assert "accuracy" in out
        assert "accuracy/total" in out

    def test_print_summary_metrics_unavailable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rows = [
            _make_row(0, true_label=0, predicted_decision=None, classify_status="failed"),
        ]
        m = _compute(tmp_path, rows)
        _print_summary(m)
        out = capsys.readouterr().out
        assert "unavailable" in out
