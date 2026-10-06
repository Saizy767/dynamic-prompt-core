"""Outbound port: contract for teacher-model refinement calls."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dynamic_prompt_core.domain.services.thesis_refinement import RefinementReview

__all__ = ["RefinementReview", "TeacherLLMClient"]


@runtime_checkable
class TeacherLLMClient(Protocol):
    """Outbound port: contract for teacher-model LLM calls.

    Distinct from LLMClient to allow independent configuration and swapping
    of the teacher model used in Stage 4 refinement. The implementation is
    responsible for assembling the system prompt from its own configuration.
    """

    async def review_theses(
        self,
        text: str,
        theses: list[str],
    ) -> RefinementReview:
        """Ask the teacher model to review candidate theses for a text.

        Returns a ``RefinementReview`` carrying the structured plan
        (keep / reformulate / drop / add) plus latency and usage.
        """
        ...
