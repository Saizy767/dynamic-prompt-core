"""Tokenizer seam for candidate scoring.

The ``TokenizerAdapter`` protocol decouples the scoring algorithm from a
concrete tokenizer library.  ``HuggingFaceTokenizerAdapter`` wraps
``transformers.AutoTokenizer`` for production use.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from transformers import AutoTokenizer


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Infrastructure seam: convert text into model tokens and back."""

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token IDs."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode ``token_ids`` back into text."""
        ...


class HuggingFaceTokenizerAdapter:
    """Concrete adapter wrapping ``transformers.AutoTokenizer``."""

    def __init__(self, model_name_or_path: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True))
