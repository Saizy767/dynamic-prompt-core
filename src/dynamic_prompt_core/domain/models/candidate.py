"""Domain model for a classification candidate."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One possible classification hypothesis that can be evaluated.

    The value is stored verbatim. No normalization (strip, lower, casefold)
    is applied; the caller or a future candidate-generation layer owns
    normalization.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("candidate value must not be empty or blank")
