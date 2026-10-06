"""Pydantic schema for binary classification structured output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class ClassificationResult(BaseModel):
    """Structured output for binary classification."""
    decision: Literal[0, 1]
    confidence: int

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError(f"confidence must be 0-100, got {v}")
        return v
