"""Typed cycle configuration for the run_cycle use case.

The composition root builds ``CycleConfig`` from the ``[cycle_orchestrator]``
TOML section and passes it through ``RunCycleInput``.  The use case itself
never opens ``config.toml``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
        StopCriteriaConfig,
    )

DEFAULT_MAX_ROUNDS = 5
DEFAULT_MAX_CONSECUTIVE_ROLLBACKS = 2
DEFAULT_DECISION_METRIC = "macro_f1"
DEFAULT_TIE_BREAKER_METRIC = "minority_f1"
DEFAULT_DEV_SPLIT = "dev"
DEFAULT_HOLDOUT_SPLIT = "holdout"
DEFAULT_OUTPUT_DIR = "data/results"


@dataclass(frozen=True)
class CycleConfig:
    """Configuration for a single optimization cycle run.

    ``duration`` controls how many dataset rows each round processes:
    - ``0`` (default): process all rows in the split.
    - positive ``int``: process that many rows in every round.
    - ``list[int]``: run ``len(list)`` rounds, round *i* processes
      ``list[i]`` rows (overrides ``max_rounds``).
    """

    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_consecutive_rollbacks: int = DEFAULT_MAX_CONSECUTIVE_ROLLBACKS
    decision_metric: str = DEFAULT_DECISION_METRIC
    tie_breaker_metric: str = DEFAULT_TIE_BREAKER_METRIC
    dev_split: str = DEFAULT_DEV_SPLIT
    holdout_split: str = DEFAULT_HOLDOUT_SPLIT
    run_id: str = ""
    output_dir: str = DEFAULT_OUTPUT_DIR
    prompt_store_path: str = ""
    thesis_bank_path: str = ""
    candidate_queue_path: str = ""
    stop_on_first_error: bool = False
    dump_state_after_each_round: bool = True
    use_teacher_refinement: bool = False
    duration: int | list[int] = 0
    stop_criteria_config: StopCriteriaConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.duration, list) and self.duration:
            object.__setattr__(self, "max_rounds", len(self.duration))

    def duration_for_round(self, round_number: int) -> int | None:
        """Return the row limit for a given 1-indexed round number.

        Returns ``None`` when all rows should be processed.
        """
        d = self.duration
        if isinstance(d, list):
            idx = round_number - 1
            if 0 <= idx < len(d):
                val = d[idx]
                return val if val > 0 else None
            return None
        if d > 0:
            return d
        return None

    @classmethod
    def from_toml(cls, config_path: str) -> CycleConfig:
        """Build a ``CycleConfig`` from the ``[cycle_orchestrator]`` section."""
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("cycle_orchestrator", {})

        raw_duration: Any = s.get("duration", 0)
        if isinstance(raw_duration, list):
            duration: int | list[int] = [int(v) for v in raw_duration]
        else:
            duration = int(raw_duration)

        max_rounds = int(s.get("max_rounds", DEFAULT_MAX_ROUNDS))
        if isinstance(duration, list) and duration:
            max_rounds = len(duration)

        return cls(
            max_rounds=max_rounds,
            max_consecutive_rollbacks=int(
                s.get("max_consecutive_rollbacks", DEFAULT_MAX_CONSECUTIVE_ROLLBACKS)
            ),
            decision_metric=s.get("decision_metric", DEFAULT_DECISION_METRIC),
            tie_breaker_metric=s.get("tie_breaker_metric", DEFAULT_TIE_BREAKER_METRIC),
            dev_split=s.get("dev_split", DEFAULT_DEV_SPLIT),
            holdout_split=s.get("holdout_split", DEFAULT_HOLDOUT_SPLIT),
            run_id=s.get("run_id", ""),
            output_dir=s.get("output_dir", DEFAULT_OUTPUT_DIR),
            prompt_store_path=s.get("prompt_store_path", ""),
            thesis_bank_path=s.get("thesis_bank_path", ""),
            candidate_queue_path=s.get("candidate_queue_path", ""),
            stop_on_first_error=bool(s.get("stop_on_first_error", False)),
            dump_state_after_each_round=bool(s.get("dump_state_after_each_round", True)),
            use_teacher_refinement=bool(s.get("use_teacher_refinement", False)),
            duration=duration,
        )
