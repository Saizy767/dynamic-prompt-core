from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """Outbound port: contract for LLM inference calls.

    Implementations own their HTTP transport: no session or connection
    object crosses this boundary. Callers close the transport with
    ``aclose`` when done.
    """

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

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text* with the client's tokenizer."""
        ...

    async def aclose(self) -> None:
        """Release the underlying transport resources."""
        ...
