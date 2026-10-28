from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class LLMClient(Protocol):
    """Outbound port: contract for LLM inference calls."""

    async def extract_theses(
        self,
        text: str,
        model: type[T],
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        truncate_tokens: int = 2000,
    ) -> T | None:
        """Extract theses from a single text."""
        ...
