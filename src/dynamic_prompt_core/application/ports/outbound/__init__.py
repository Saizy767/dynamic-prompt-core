"""Outbound ports: contracts for external dependencies."""
from __future__ import annotations

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

__all__ = [
    "DatasetRepository",
    "EmbeddingClient",
    "LLMClient",
    "Normalizer",
    "PromptRepository",
    "RunRepository",
    "TeacherLLMClient",
]
