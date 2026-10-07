"""Typed dependency object for the evaluate_stop_criteria use case."""

from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.application.ports.outbound.run_repository import (
    RunRepository,
)

__all__ = ["EvaluateStopCriteriaDeps"]


@dataclass(frozen=True)
class EvaluateStopCriteriaDeps:
    """All outbound ports the evaluate_stop_criteria use case needs.

    ``run_repository`` is used to persist the ``stop_decision`` artifact when
    the decision is ``stop``.
    """

    run_repository: RunRepository
