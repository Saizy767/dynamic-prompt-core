from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.application.ports.outbound.llm_client import LLMClient
from dynamic_prompt_core.application.ports.outbound.task_store import TaskStore


@dataclass(frozen=True)
class RunCycleDeps:
    """Typed dependency object for the run_cycle use case.

    All outbound ports the use case needs are declared here. Adding a field
    is a breaking change to the use case's interface.
    """

    llm_client: LLMClient
    task_store: TaskStore
