"""Public API for the NLP infrastructure package."""
from __future__ import annotations

from dynamic_prompt_core.infrastructure.nlp.normalizer import (
    normalize_theses,
    normalize_thesis,
)

__all__ = [
    "normalize_theses",
    "normalize_thesis",
]
