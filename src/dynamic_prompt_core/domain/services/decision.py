"""Domain service: accept/rollback decision rule for version comparison."""
from __future__ import annotations

from typing import Any


class DecisionError(ValueError):
    """Raised when a metric cannot be extracted or is unknown."""


def extract_metric(metrics: dict[str, Any], metric_name: str) -> float:
    """Extract a metric value from a metrics dict by name.

    Config names (macro_f1, minority_f1, weighted_f1) map to the f1 sub-dict
    keys used by stage1-metrics (macro, minority, weighted).
    """
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    f1_key_map = {
        "macro_f1": "macro",
        "minority_f1": "minority",
        "weighted_f1": "weighted",
    }
    if metric_name in f1_key_map:
        return float(metrics["f1"][f1_key_map[metric_name]])
    raise DecisionError(
        f"unknown metric: {metric_name!r} "
        f"(expected accuracy, macro_f1, minority_f1, or weighted_f1)"
    )


def decide(
    metrics_new: dict[str, Any],
    metrics_active: dict[str, Any],
    decision_metric: str,
    tie_breaker_metric: str,
) -> tuple[str, str]:
    """Decide whether to accept or roll back the new version.

    Accept when the new version's decision_metric is >= the active's.
    When equal, accept only if the tie_breaker_metric is strictly higher.
    Otherwise roll back.

    Returns (decision, reason) where decision is "accept" or "rollback".
    """
    primary = decision_metric
    tie = tie_breaker_metric

    new_primary = extract_metric(metrics_new, primary)
    active_primary = extract_metric(metrics_active, primary)

    if new_primary > active_primary:
        return "accept", f"{primary} improved ({active_primary:.4f} -> {new_primary:.4f})"

    if new_primary == active_primary:
        new_tie = extract_metric(metrics_new, tie)
        active_tie = extract_metric(metrics_active, tie)
        if new_tie > active_tie:
            return (
                "accept",
                f"{primary} tied ({new_primary:.4f}), {tie} improved "
                f"({active_tie:.4f} -> {new_tie:.4f})",
            )
        return (
            "accept",
            f"{primary} tied ({new_primary:.4f}), {tie} did not improve "
            f"({active_tie:.4f} -> {new_tie:.4f})",
        )

    return (
        "rollback",
        f"{primary} degraded ({active_primary:.4f} -> {new_primary:.4f})",
    )
