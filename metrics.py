"""
Stage 1 metrics.

Loads a stage1-baseline-runner results artifact, computes classification
quality metrics (accuracy, per-class / macro / weighted F1, confusion matrix,
minority-class F1, prediction distribution and bias, group-by-length breakdown,
parse-failure counts), persists a json metrics artifact tied to the prompt
version and run, and compares two prompt-version metrics on the same split.

Usage:
    python metrics.py compute --results data/results/results_*.jsonl
    python metrics.py compare --v0 data/results/results_*_v0_*.jsonl \
                              --v1 data/results/results_*_v1_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_BIAS_THRESHOLD = 0.15
DEFAULT_GROUP_SHORT_MAX = 10
DEFAULT_GROUP_LONG_MIN = 30
DEFAULT_OUTPUT_DIR = "data/results"

REQUIRED_FIELDS = (
    "id",
    "text",
    "true_label",
    "predicted_decision",
    "confidence",
    "classify_status",
    "theses_norm",
)

_ARTIFACT_NAME_RE = re.compile(
    r"^results_(?P<run_id>.+)_(?P<prompt_version>.+)_(?P<split>dev|holdout)_(?P<timestamp>[^.]+)\.jsonl$"
)


class MetricsError(ValueError):
    """Raised when a results artifact is invalid or cannot be loaded."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class MetricsConfig:
    bias_threshold: float = DEFAULT_BIAS_THRESHOLD
    group_short_max: int = DEFAULT_GROUP_SHORT_MAX
    group_long_min: int = DEFAULT_GROUP_LONG_MIN
    output_dir: str = DEFAULT_OUTPUT_DIR

    @classmethod
    def from_config(
        cls, config_path: str = DEFAULT_CONFIG_PATH
    ) -> "MetricsConfig":
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        m = config.get("metrics", {})
        return cls(
            bias_threshold=float(m.get("bias_threshold", DEFAULT_BIAS_THRESHOLD)),
            group_short_max=int(m.get("group_short_max", DEFAULT_GROUP_SHORT_MAX)),
            group_long_min=int(m.get("group_long_min", DEFAULT_GROUP_LONG_MIN)),
            output_dir=m.get("output_dir", DEFAULT_OUTPUT_DIR),
        )


# --------------------------------------------------------------------------- #
#  Artifact loading and validation
# --------------------------------------------------------------------------- #
def load_results(path: str) -> List[Dict[str, Any]]:
    """Load and schema-validate a results_*.jsonl artifact.

    Raises MetricsError naming the first offending record's id and the missing
    field when a record does not conform.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _validate_row(row)
            rows.append(row)
    return rows


def _validate_row(row: Dict[str, Any]) -> None:
    rid = row.get("id")
    for fname in REQUIRED_FIELDS:
        if fname not in row or row[fname] is None:
            raise MetricsError(
                f"record id={rid!r} is missing required field '{fname}'"
            )


def parse_artifact_name(path: str) -> Dict[str, str]:
    """Extract run_id, prompt_version, split, timestamp from a results filename."""
    basename = os.path.basename(path)
    m = _ARTIFACT_NAME_RE.match(basename)
    if not m:
        raise MetricsError(
            f"artifact filename '{basename}' does not match the pattern "
            f"results_{{run_id}}_{{prompt_version}}_{{split}}_{{timestamp}}.jsonl"
        )
    return {
        "run_id": m.group("run_id"),
        "prompt_version": m.group("prompt_version"),
        "split": m.group("split"),
        "timestamp": m.group("timestamp"),
    }


# --------------------------------------------------------------------------- #
#  Metric computation helpers
# --------------------------------------------------------------------------- #
def _filter_ok(
    rows: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Split rows into ok_rows (classify_status=ok) and parse_counts."""
    ok_rows = [r for r in rows if r["classify_status"] == "ok"]
    parse_counts = dict(Counter(r["classify_status"] for r in rows))
    return ok_rows, parse_counts


def _compute_accuracy(ok_rows: List[Dict[str, Any]]) -> float:
    if not ok_rows:
        return 0.0
    correct = sum(
        1 for r in ok_rows if r["predicted_decision"] == r["true_label"]
    )
    return correct / len(ok_rows)


def _compute_prf(
    ok_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float, float]:
    """Per-class precision/recall/F1 plus macro-F1 and weighted-F1.

    Returns (precision, recall, f1, macro_f1, weighted_f1) where the per-class
    dicts are keyed by string label "0" and "1".
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for metric computation. "
            "Install with: pip install scikit-learn numpy"
        ) from exc

    if not ok_rows:
        zeros = {"0": 0.0, "1": 0.0}
        return zeros, zeros, zeros, 0.0, 0.0

    y_true = [r["true_label"] for r in ok_rows]
    y_pred = [r["predicted_decision"] for r in ok_rows]

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    precision = {"0": float(p[0]), "1": float(p[1])}
    recall = {"0": float(r[0]), "1": float(r[1])}
    f1 = {"0": float(f[0]), "1": float(f[1])}

    macro_f1 = float((f[0] + f[1]) / 2)
    total_support = int(s[0] + s[1])
    weighted_f1 = (
        float((f[0] * s[0] + f[1] * s[1]) / total_support)
        if total_support > 0
        else 0.0
    )

    return precision, recall, f1, macro_f1, weighted_f1


def _compute_confusion_matrix(
    ok_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build a labeled 2x2 confusion matrix {tp, tn, fp, fn, matrix}."""
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for metric computation. "
            "Install with: pip install scikit-learn numpy"
        ) from exc

    if not ok_rows:
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "matrix": [[0, 0], [0, 0]]}

    y_true = [r["true_label"] for r in ok_rows]
    y_pred = [r["predicted_decision"] for r in ok_rows]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    tp = int(cm[1][1])

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "matrix": [[tn, fp], [fn, tp]],
    }


def _minority_class(ok_rows: List[Dict[str, Any]]) -> int:
    """Determine the minority class from true_label distribution."""
    label_counts = Counter(r["true_label"] for r in ok_rows)
    if not label_counts:
        return 0
    return min(label_counts, key=label_counts.get)


def _prediction_distribution(
    ok_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Counts and shares of predicted_decision 0 and 1."""
    counts = Counter(r["predicted_decision"] for r in ok_rows)
    n = len(ok_rows)
    return {
        "0": float(counts.get(0, 0) / n) if n > 0 else 0.0,
        "1": float(counts.get(1, 0) / n) if n > 0 else 0.0,
        "counts": {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))},
    }


def _true_distribution(
    ok_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Counts and shares of true_label 0 and 1."""
    counts = Counter(r["true_label"] for r in ok_rows)
    n = len(ok_rows)
    return {
        "0": float(counts.get(0, 0) / n) if n > 0 else 0.0,
        "1": float(counts.get(1, 0) / n) if n > 0 else 0.0,
        "counts": {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))},
    }


def _bias_detection(
    pred_dist: Dict[str, Any],
    true_dist: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Compare predicted and true distributions; flag bias above threshold."""
    gap = max(
        abs(pred_dist["0"] - true_dist["0"]),
        abs(pred_dist["1"] - true_dist["1"]),
    )
    return {
        "gap": float(gap),
        "biased": bool(gap > threshold),
        "toward": 1 if pred_dist["1"] > true_dist["1"] else 0,
    }


# --------------------------------------------------------------------------- #
#  Group breakdown
# --------------------------------------------------------------------------- #
def length_group(word_count: int, short_max: int, long_min: int) -> str:
    """Classify a word count into short / medium / long."""
    if word_count < short_max:
        return "short"
    if word_count <= long_min:
        return "medium"
    return "long"


def _group_breakdown(
    ok_rows: List[Dict[str, Any]], config: MetricsConfig
) -> Dict[str, Any]:
    """Per-group accuracy and F1 by text length."""
    groups: Dict[str, List[Dict[str, Any]]] = {
        "short": [],
        "medium": [],
        "long": [],
    }
    for r in ok_rows:
        wc = len(r["text"].split())
        g = length_group(wc, config.group_short_max, config.group_long_min)
        groups[g].append(r)

    result: Dict[str, Any] = {}
    for name in ("short", "medium", "long"):
        rows_g = groups[name]
        count = len(rows_g)
        if count == 0:
            result[name] = {"count": 0, "accuracy": None, "f1": None}
        else:
            acc = _compute_accuracy(rows_g)
            _, _, f1_dict, _, _ = _compute_prf(rows_g)
            macro_f1 = (f1_dict["0"] + f1_dict["1"]) / 2
            result[name] = {
                "count": count,
                "accuracy": float(acc),
                "f1": float(macro_f1),
            }
    return result


# --------------------------------------------------------------------------- #
#  Full metrics computation
# --------------------------------------------------------------------------- #
def compute_metrics(
    rows: List[Dict[str, Any]],
    config: MetricsConfig,
    source_artifact: str,
) -> Dict[str, Any]:
    """Compute the full metrics dict from validated result rows."""
    name_parts = parse_artifact_name(source_artifact)
    ok_rows, parse_counts = _filter_ok(rows)

    accuracy = _compute_accuracy(ok_rows)
    precision, recall, f1, macro_f1, weighted_f1 = _compute_prf(ok_rows)
    confusion = _compute_confusion_matrix(ok_rows)
    minority = _minority_class(ok_rows)
    minority_f1 = f1[str(minority)]
    pred_dist = _prediction_distribution(ok_rows)
    true_dist = _true_distribution(ok_rows)
    bias = _bias_detection(pred_dist, true_dist, config.bias_threshold)
    groups = _group_breakdown(ok_rows, config)

    counts = {
        "total": len(rows),
        "ok": parse_counts.get("ok", 0),
        "failed": parse_counts.get("failed", 0),
        "repaired": parse_counts.get("repaired", 0),
        "not_attempted": parse_counts.get("not_attempted", 0),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return {
        "run_id": name_parts["run_id"],
        "prompt_version": name_parts["prompt_version"],
        "split": name_parts["split"],
        "timestamp": timestamp,
        "source_artifact": os.path.basename(source_artifact),
        "counts": counts,
        "accuracy": float(accuracy),
        "precision": precision,
        "recall": recall,
        "f1": {
            "0": f1["0"],
            "1": f1["1"],
            "macro": macro_f1,
            "weighted": weighted_f1,
            "minority": minority_f1,
        },
        "minority_class": minority,
        "confusion_matrix": confusion,
        "prediction_distribution": pred_dist,
        "true_distribution": true_dist,
        "bias": bias,
        "groups": groups,
        "comparison": None,
    }


# --------------------------------------------------------------------------- #
#  Metrics artifact persistence
# --------------------------------------------------------------------------- #
def write_metrics(metrics_dict: Dict[str, Any], output_dir: str) -> str:
    """Write metrics as metrics_{run_id}_{prompt_version}_{split}_{timestamp}.json."""
    os.makedirs(output_dir, exist_ok=True)
    name = (
        f"metrics_{metrics_dict['run_id']}_{metrics_dict['prompt_version']}_"
        f"{metrics_dict['split']}_{metrics_dict['timestamp']}.json"
    )
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    return path


def load_metrics(path: str) -> Dict[str, Any]:
    """Load a metrics json artifact."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Version comparison
# --------------------------------------------------------------------------- #
def compare_versions(
    results_path_v0: str,
    results_path_v1: str,
    config: MetricsConfig,
) -> Dict[str, Any]:
    """Compare two results artifacts on the same split.

    Returns per-metric deltas (v1 - v0) and the list of ids whose
    predicted_decision changed, with direction (fixed / broke / flip).
    """
    parts0 = parse_artifact_name(results_path_v0)
    parts1 = parse_artifact_name(results_path_v1)

    if parts0["split"] != parts1["split"]:
        raise MetricsError(
            f"cannot compare artifacts with different splits: "
            f"v0 split={parts0['split']!r}, v1 split={parts1['split']!r}"
        )

    rows0 = load_results(results_path_v0)
    rows1 = load_results(results_path_v1)

    metrics0 = compute_metrics(rows0, config, results_path_v0)
    metrics1 = compute_metrics(rows1, config, results_path_v1)

    deltas = {
        "accuracy": metrics1["accuracy"] - metrics0["accuracy"],
        "macro_f1": metrics1["f1"]["macro"] - metrics0["f1"]["macro"],
        "minority_f1": (
            metrics1["f1"]["minority"] - metrics0["f1"]["minority"]
        ),
    }

    pred_by_id_v0 = {
        r["id"]: r["predicted_decision"]
        for r in rows0
        if r["classify_status"] == "ok"
    }
    pred_by_id_v1 = {
        r["id"]: r["predicted_decision"]
        for r in rows1
        if r["classify_status"] == "ok"
    }
    true_by_id = {}
    for r in rows0:
        if r["classify_status"] == "ok":
            true_by_id[r["id"]] = r["true_label"]
    for r in rows1:
        if r["classify_status"] == "ok":
            true_by_id.setdefault(r["id"], r["true_label"])

    common_ids = set(pred_by_id_v0.keys()) & set(pred_by_id_v1.keys())
    only_v0 = set(pred_by_id_v0.keys()) - set(pred_by_id_v1.keys())
    only_v1 = set(pred_by_id_v1.keys()) - set(pred_by_id_v0.keys())

    changed: List[Dict[str, Any]] = []
    for rid in sorted(common_ids, key=lambda x: str(x)):
        p0 = pred_by_id_v0[rid]
        p1 = pred_by_id_v1[rid]
        if p0 == p1:
            continue
        true_label = true_by_id.get(rid)
        correct0 = (p0 == true_label) if true_label is not None else False
        correct1 = (p1 == true_label) if true_label is not None else False
        if not correct0 and correct1:
            direction = "fixed"
        elif correct0 and not correct1:
            direction = "broke"
        else:
            direction = "flip"
        changed.append(
            {"id": rid, "v0": p0, "v1": p1, "direction": direction}
        )

    return {
        "v0": {
            "run_id": parts0["run_id"],
            "prompt_version": parts0["prompt_version"],
            "split": parts0["split"],
        },
        "v1": {
            "run_id": parts1["run_id"],
            "prompt_version": parts1["prompt_version"],
            "split": parts1["split"],
        },
        "deltas": deltas,
        "changed": changed,
        "overlap": {
            "common": len(common_ids),
            "only_v0": len(only_v0),
            "only_v1": len(only_v1),
        },
    }


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _print_summary(metrics: Dict[str, Any]) -> None:
    c = metrics["counts"]
    print("\n" + "=" * 60)
    print(
        f"Metrics  (run_id={metrics['run_id']}, "
        f"split={metrics['split']}, prompt_version={metrics['prompt_version']})"
    )
    print("=" * 60)
    print(f"  Total          : {c['total']}")
    print(f"  ok             : {c['ok']}")
    print(f"  failed         : {c['failed']}")
    print(f"  repaired       : {c['repaired']}")
    print(f"  not_attempted  : {c['not_attempted']}")
    print(f"  accuracy       : {metrics['accuracy']:.4f}")
    print(f"  macro-F1       : {metrics['f1']['macro']:.4f}")
    print(f"  minority-F1    : {metrics['f1']['minority']:.4f}  (class {metrics['minority_class']})")
    print(f"  biased         : {metrics['bias']['biased']}  (gap={metrics['bias']['gap']:.4f})")
    print("=" * 60)


def _cmd_compute(args: argparse.Namespace) -> None:
    config = MetricsConfig.from_config(args.config)
    rows = load_results(args.results)
    metrics = compute_metrics(rows, config, args.results)
    path = write_metrics(metrics, config.output_dir)
    _print_summary(metrics)
    print(f"Artifact: {path}")


def _cmd_compare(args: argparse.Namespace) -> None:
    config = MetricsConfig.from_config(args.config)
    result = compare_versions(args.v0, args.v1, config)
    d = result["deltas"]
    print("\n" + "=" * 60)
    print(
        f"Comparison  (v0={result['v0']['prompt_version']}, "
        f"v1={result['v1']['prompt_version']}, split={result['v0']['split']})"
    )
    print("=" * 60)
    print(f"  Δ accuracy     : {d['accuracy']:+.4f}")
    print(f"  Δ macro-F1     : {d['macro_f1']:+.4f}")
    print(f"  Δ minority-F1  : {d['minority_f1']:+.4f}")
    print(f"  changed        : {len(result['changed'])} records")
    directions = Counter(c["direction"] for c in result["changed"])
    print(f"    fixed={directions.get('fixed', 0)}, "
          f"broke={directions.get('broke', 0)}, "
          f"flip={directions.get('flip', 0)}")
    ov = result["overlap"]
    print(f"  overlap        : common={ov['common']}, "
          f"only_v0={ov['only_v0']}, only_v1={ov['only_v1']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 metrics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compute = sub.add_parser("compute", help="Compute metrics from a results artifact")
    p_compute.add_argument("--results", required=True, help="Path to results_*.jsonl")
    p_compute.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p_compute.set_defaults(func=_cmd_compute)

    p_compare = sub.add_parser("compare", help="Compare two results artifacts")
    p_compare.add_argument("--v0", required=True, help="Path to v0 results_*.jsonl")
    p_compare.add_argument("--v1", required=True, help="Path to v1 results_*.jsonl")
    p_compare.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p_compare.set_defaults(func=_cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
