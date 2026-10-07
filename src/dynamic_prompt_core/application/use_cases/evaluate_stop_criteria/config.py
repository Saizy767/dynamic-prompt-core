"""Typed stop-criteria configuration for the evaluate_stop_criteria use case.

The composition root builds ``StopCriteriaConfig`` from the ``[stop_criteria]``
TOML section and passes it through ``StopEvaluationContext``.  The use case
itself never opens ``config.toml``.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass

DEFAULT_PLATEAU_WINDOW = 3
DEFAULT_MIN_IMPROVEMENT = 0.01
DEFAULT_DEGRADATION_WINDOW = 3
DEFAULT_STAGNATION_WINDOW = 3
DEFAULT_MAX_TOTAL_ROUNDS = 10
DEFAULT_DECISION_METRIC = "macro_f1"

ALL_CRITERIA: tuple[str, ...] = (
    "budget_exhausted",
    "no_candidates_available",
    "plateau_detected",
    "metric_degradation",
    "rule_stagnation",
    "rollback_streak",
)


@dataclass(frozen=True)
class StopCriteriaConfig:
    """Configuration for content-based stop criteria.

    ``enabled_criteria`` of ``None`` means all criteria are enabled.  A list
    restricts evaluation to the named criteria.
    """

    plateau_window: int = DEFAULT_PLATEAU_WINDOW
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT
    degradation_window: int = DEFAULT_DEGRADATION_WINDOW
    stagnation_window: int = DEFAULT_STAGNATION_WINDOW
    max_total_rounds: int = DEFAULT_MAX_TOTAL_ROUNDS
    max_total_tokens: int | None = None
    enabled_criteria: list[str] | None = None
    decision_metric: str = DEFAULT_DECISION_METRIC

    @classmethod
    def from_toml(cls, config_path: str) -> StopCriteriaConfig:
        """Build a ``StopCriteriaConfig`` from the ``[stop_criteria]`` section."""
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("stop_criteria", {})

        max_tokens_raw = s.get("max_total_tokens")
        max_total_tokens: int | None = (
            int(max_tokens_raw) if max_tokens_raw is not None else None
        )

        enabled_raw = s.get("enabled_criteria")
        enabled_criteria: list[str] | None = (
            list(enabled_raw) if enabled_raw is not None else None
        )

        return cls(
            plateau_window=int(s.get("plateau_window", DEFAULT_PLATEAU_WINDOW)),
            min_improvement=float(s.get("min_improvement", DEFAULT_MIN_IMPROVEMENT)),
            degradation_window=int(
                s.get("degradation_window", DEFAULT_DEGRADATION_WINDOW)
            ),
            stagnation_window=int(
                s.get("stagnation_window", DEFAULT_STAGNATION_WINDOW)
            ),
            max_total_rounds=int(
                s.get("max_total_rounds", DEFAULT_MAX_TOTAL_ROUNDS)
            ),
            max_total_tokens=max_total_tokens,
            enabled_criteria=enabled_criteria,
            decision_metric=s.get("decision_metric", DEFAULT_DECISION_METRIC),
        )
