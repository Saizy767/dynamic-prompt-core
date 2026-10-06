"""Typed dependency object for the refine_theses use case."""
from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.application.ports.outbound.run_repository import (
    RunRepository,
)
from dynamic_prompt_core.application.ports.outbound.teacher_llm_client import (
    TeacherLLMClient,
)


@dataclass(frozen=True)
class RefineThesesDeps:
    """All outbound ports the refine_theses use case needs.

    Adding a field is a breaking change to the use case's interface.
    """

    teacher_llm_client: TeacherLLMClient
    run_repository: RunRepository
