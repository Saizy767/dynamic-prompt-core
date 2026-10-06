"""Pydantic schema for rule formulation structured output."""
from __future__ import annotations

from pydantic import BaseModel


class RuleFormulation(BaseModel):
    """Structured output for rule formulation: a single-sentence rule."""
    rule: str
