"""
Pydantic v2 schemas for structured output via AsyncTask.classify /
AsyncTask.extract_theses.

ClassificationResult  -> AsyncTask.classify(model=ClassificationResult)
ThesisExtraction      -> AsyncTask.extract_theses(model=ThesisExtraction)
"""
from __future__ import annotations

from typing import List, Literal

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


class ThesisExtraction(BaseModel):
    """Structured output for thesis extraction: 3-5 theses, each 2-6 words."""
    theses: List[str]

    @field_validator("theses")
    @classmethod
    def _thesis_count(cls, v: List[str]) -> List[str]:
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


class RuleFormulation(BaseModel):
    """Structured output for rule formulation: a single-sentence rule."""
    rule: str
