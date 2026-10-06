from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class TeacherLLMClient(Protocol):
    """Outbound port: contract for teacher-model LLM calls.

    Distinct from LLMClient to allow independent configuration and swapping
    of the teacher model used in Stage 4 refinement.
    """

    async def classify(
        self,
        text: str,
        model: type[T],
        *,
        system_prompt: str | None = None,
        max_tokens: int = 128,
        truncate_tokens: int = 300,
    ) -> T | None:
        """Classify a single text using the teacher model."""
        ...

    async def extract_theses(
        self,
        text: str,
        model: type[T],
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
    ) -> T | None:
        """Extract theses from a single text using the teacher model."""
        ...
