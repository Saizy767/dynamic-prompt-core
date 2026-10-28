"""Inbound port: input DTO for the run_cycle use case."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunCycleInput:
    """Input for the run_cycle use case.

    ``config`` is the resolved ``CycleConfig``; the composition root applies
    any CLI overrides to it before calling the use case.  When ``config`` is
    ``None`` the use case loads it from ``config_path``.
    """

    config_path: str
    resume_from: str | None = None
    config: Any = None
