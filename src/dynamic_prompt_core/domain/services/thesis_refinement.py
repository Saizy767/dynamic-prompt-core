"""Pure domain logic for applying a teacher refinement plan to candidate theses.

This module is framework-free: it uses only the Python standard library.
The ``RefinementPlanData`` dataclass mirrors the Pydantic ``RefinementPlan``
from ``application.schemas.refinement`` but without any framework dependency.
``RefinementReview`` is the typed result of a teacher review call, kept in
domain so both the port and the infrastructure adapter can import it without
creating a layer cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RefinementReview:
    """Typed result of a teacher-model review call."""

    keep: list[str] = field(default_factory=list)
    reformulate: list[dict[str, str]] = field(default_factory=list)
    drop: list[dict[str, str]] = field(default_factory=list)
    add: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReformulatePair:
    from_: str
    to: str


@dataclass(frozen=True)
class DropEntry:
    thesis: str
    reason: str


@dataclass(frozen=True)
class RefinementPlanData:
    keep: list[str] = field(default_factory=list)
    reformulate: list[ReformulatePair] = field(default_factory=list)
    drop: list[DropEntry] = field(default_factory=list)
    add: list[str] = field(default_factory=list)


def apply_refinement(
    theses_raw: list[str],
    plan: RefinementPlanData,
    *,
    filter_noisy: bool,
    filter_interpretive: bool,
    allow_additions: bool,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    """Build ``(theses_refined, filtered_out, added)`` from a refinement plan.

    ``theses_refined`` order: keep, then reformulated, then added.
    Reformulations longer than the original are discarded in favour of the
    original (length enforcement). Drops are recorded in ``filtered_out``
    with their reason. Added theses are tagged ``source="teacher"``.
    """
    theses_refined: list[str] = []
    filtered_out: list[dict[str, str]] = []
    added: list[dict[str, str]] = []

    for t in plan.keep:
        theses_refined.append(t)

    for item in plan.reformulate:
        if len(item.to) > len(item.from_):
            theses_refined.append(item.from_)
        else:
            theses_refined.append(item.to)

    for drop_item in plan.drop:
        if filter_noisy or filter_interpretive:
            filtered_out.append(
                {"thesis": drop_item.thesis, "reason": drop_item.reason}
            )
        else:
            theses_refined.append(drop_item.thesis)

    if allow_additions:
        for t in plan.add:
            theses_refined.append(t)
            added.append({"text": t, "source": "teacher"})

    return theses_refined, filtered_out, added
