"""Pydantic schemas for refinement and rule formulation structured output."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RuleFormulation(BaseModel):
    """Structured output for rule formulation: a single-sentence rule."""
    rule: str


class ReformulateItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: str = Field(alias="from")
    to: str


class DropItem(BaseModel):
    thesis: str
    reason: str


class RefinementPlan(BaseModel):
    keep: list[str] = []
    reformulate: list[ReformulateItem] = []
    drop: list[DropItem] = []
    add: list[str] = []
