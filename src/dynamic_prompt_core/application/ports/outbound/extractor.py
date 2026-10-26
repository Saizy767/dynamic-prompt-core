"""Outbound port: contract for key-phrase extraction."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import aiohttp

    from dynamic_prompt_core.infrastructure.llm.extractor import ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    """Outbound port: extract key phrases from input text.

    The extractor performs an independent model call with its own system
    prompt.  It does not receive classification results.
    """

    async def extract(
        self,
        session: aiohttp.ClientSession,
        text: str,
        true_val: int | None = None,
    ) -> ExtractionResult:
        """Extract key phrases from ``text``."""
        ...

    @staticmethod
    def skipped() -> ExtractionResult:
        """Return a skipped extraction result."""
        ...
