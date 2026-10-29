"""
Version-comparison decision logic for the optimization cycle.

``decide`` compares the new prompt version's metrics against the active
version's — the cycle evaluates both on the holdout split — and returns
"accept" or "rollback" with a human-readable reason: accept when the
decision metric strictly improves, or when tied and the tie-breaker metric
strictly improves.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import Any

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_DECISION_METRIC = "macro_f1"
DEFAULT_TIE_BREAKER_METRIC = "minority_f1"


class VersionComparatorError(ValueError):
    """Raised when an artifact is invalid or comparison fails."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class VersionComparatorConfig:
    decision_metric: str = DEFAULT_DECISION_METRIC
    tie_breaker_metric: str = DEFAULT_TIE_BREAKER_METRIC

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_CONFIG_PATH) -> VersionComparatorConfig:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("version_comparator", {})
        return cls(
            decision_metric=s.get("decision_metric", DEFAULT_DECISION_METRIC),
            tie_breaker_metric=s.get("tie_breaker_metric", DEFAULT_TIE_BREAKER_METRIC),
        )


# --------------------------------------------------------------------------- #
#  Decision logic
# --------------------------------------------------------------------------- #
def _extract_metric(metrics: dict[str, Any], metric_name: str) -> float:
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
    raise VersionComparatorError(
        f"unknown metric: {metric_name!r} "
        f"(expected accuracy, macro_f1, minority_f1, or weighted_f1)"
    )


def decide(
    metrics_new: dict[str, Any],
    metrics_active: dict[str, Any],
    config: VersionComparatorConfig,
) -> tuple[str, str]:
    """Decide whether to accept or roll back the new version.

    Accept when the new version's decision_metric is strictly greater than
    the active's. When equal, accept only if the tie_breaker_metric is
    strictly higher. Otherwise roll back.

    Returns (decision, reason) where decision is "accept" or "rollback".
    """
    if metrics_new.get("f1") is None or metrics_active.get("f1") is None:
        return "rollback", "holdout metrics unavailable (f1 missing or None)"

    primary = config.decision_metric
    tie = config.tie_breaker_metric

    new_primary = _extract_metric(metrics_new, primary)
    active_primary = _extract_metric(metrics_active, primary)

    if new_primary > active_primary:
        return "accept", f"{primary} improved ({active_primary:.4f} -> {new_primary:.4f})"

    new_tie = _extract_metric(metrics_new, tie)
    active_tie = _extract_metric(metrics_active, tie)

    if new_primary == active_primary:
        if new_tie > active_tie:
            return (
                "accept",
                f"{primary} tied ({new_primary:.4f}), {tie} improved "
                f"({active_tie:.4f} -> {new_tie:.4f})",
            )
        return (
            "rollback",
            f"{primary} tied ({new_primary:.4f}), {tie} did not improve "
            f"({active_tie:.4f} -> {new_tie:.4f})",
        )

    return (
        "rollback",
        f"{primary} degraded ({active_primary:.4f} -> {new_primary:.4f})",
    )
