"""Pydantic schema for thesis extraction structured output."""
from __future__ import annotations

from pydantic import BaseModel, field_validator


class ThesisExtraction(BaseModel):
    """Structured output for thesis extraction: 3-5 theses, each 2-6 words."""
    theses: list[str]

    @field_validator("theses")
    @classmethod
    def _thesis_count(cls, v: list[str]) -> list[str]:
        if not 3 <= len(v) <= 5:
            raise ValueError(f"expected 3-5 theses, got {len(v)}")
        for t in v:
            word_count = len(t.split())
            if not 2 <= word_count <= 6:
                raise ValueError(
                    f"each thesis must be 2-6 words; "
                    f"'{t}' has {word_count}"
                )
        return v
