from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Outbound port: contract for computing text embeddings."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Return a single embedding vector."""
        ...
