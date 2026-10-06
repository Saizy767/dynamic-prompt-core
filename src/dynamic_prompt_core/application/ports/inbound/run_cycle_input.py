from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunCycleInput:
    """Inbound port: input DTO for the run_cycle use case."""

    config_path: str
    max_rounds: int = 5
    dev_split: str = "dev"
    holdout_split: str = "holdout"
    resume_from: str | None = None
