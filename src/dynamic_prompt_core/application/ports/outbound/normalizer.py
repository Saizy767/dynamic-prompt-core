from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Normalizer(Protocol):
    """Outbound port: contract for thesis normalization."""

    def normalize_thesis(self, text: str, lang: str = "ru") -> str:
        """Normalize a single thesis."""
        ...

    def normalize_theses(self, theses: list[str], lang: str = "ru") -> list[str]:
        """Normalize a list of theses."""
        ...
