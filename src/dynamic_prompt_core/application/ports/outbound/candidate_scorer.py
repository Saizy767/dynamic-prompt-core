"""Outbound port: contract for candidate scoring."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment


@runtime_checkable
class CandidateScorer(Protocol):
    """Outbound port: evaluate candidates against a classification input.

    Contract (all-or-nothing):
        A successful call returns exactly one Judgment per supplied candidate.
        If any candidate cannot be evaluated, the call raises an exception.
        Partial results (fewer judgments than candidates) are not supported.

    Scoring semantics:
        The scorer evaluates candidates independently and produces one judgment
        per candidate. It does not decide the final classification, select a
        winning candidate, rank candidates for selection, or aggregate
        judgments. Score calibration, candidate ranking, candidate selection,
        and judgment aggregation are all outside this port.

    The port exposes no logits, token IDs, tokenizer APIs, model objects, or
    provider-specific types.
    """

    async def score(
        self,
        text: str,
        candidates: list[Candidate],
    ) -> list[Judgment]:
        """Evaluate candidates against the classification input text.

        Returns exactly one Judgment per candidate. Raises an exception if any
        candidate cannot be evaluated.
        """
        ...


def validate_unique_candidates(candidates: list[Candidate]) -> None:
    """Reject a candidate list containing duplicates.

    Called at the application boundary before the scorer is invoked. Duplicate
    candidates constitute an invalid request, not a scorer-specific concern.
    """
    seen: set[Candidate] = set()
    for candidate in candidates:
        if candidate in seen:
            raise ValueError(f"duplicate candidate: {candidate.value!r}")
        seen.add(candidate)
