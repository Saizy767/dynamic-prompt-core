from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.application.ports.inbound.run_cycle_input import RunCycleInput
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import RunCycleDeps


@dataclass(frozen=True)
class RunCycleResult:
    """Result of a run_cycle execution."""

    total_rounds: int
    stop_reason: str
    final_active_version: str
    accepted_history: list[str]
    rollback_history: list[dict[str, str | int]]


async def run_cycle(deps: RunCycleDeps, run_input: RunCycleInput) -> RunCycleResult:
    """Execute the optimization loop for a configured number of rounds.

    This use case receives all dependencies via the typed ``deps`` object.
    It does not instantiate infrastructure classes directly.
    """
    raise NotImplementedError
