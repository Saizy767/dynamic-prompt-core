"""Domain model for the result of evaluating a candidate."""
from __future__ import annotations

import math
from dataclasses import dataclass

from dynamic_prompt_core.domain.models.candidate import Candidate


@dataclass(frozen=True)
class Judgment:
    """The result of evaluating one Candidate against a classification input.

    The score is an opaque numeric ranking signal where higher means stronger
    model-backed support for the candidate. It is not a probability, confidence
    value, or calibrated likelihood. No [0, 1] constraint is imposed.
    """

    candidate: Candidate
    score: float

    def __post_init__(self) -> None:
        if math.isnan(self.score) or math.isinf(self.score):
            raise ValueError(
                f"score must be a finite number, got {self.score!r}"
            )
