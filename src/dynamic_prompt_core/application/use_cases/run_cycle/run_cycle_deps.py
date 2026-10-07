"""Typed dependency object for the run_cycle use case."""

from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.application.ports.outbound.dataset_repository import (
    DatasetRepository,
)
from dynamic_prompt_core.application.ports.outbound.embedding_client import (
    EmbeddingClient,
)
from dynamic_prompt_core.application.ports.outbound.llm_client import LLMClient
from dynamic_prompt_core.application.ports.outbound.normalizer import Normalizer
from dynamic_prompt_core.application.ports.outbound.prompt_repository import (
    PromptRepository,
)
from dynamic_prompt_core.application.ports.outbound.run_repository import (
    RunRepository,
)
from dynamic_prompt_core.application.ports.outbound.teacher_llm_client import (
    TeacherLLMClient,
)


@dataclass(frozen=True)
class RunCycleDeps:
    """All outbound ports the run_cycle use case needs.

    Adding a field is a breaking change to the use case's interface.
    ``teacher_llm_client`` is optional (``None`` when refinement is disabled).
    """

    llm_client: LLMClient
    prompt_repository: PromptRepository
    run_repository: RunRepository
    dataset_repository: DatasetRepository
    embedding_client: EmbeddingClient
    normalizer: Normalizer
    teacher_llm_client: TeacherLLMClient | None = None
