"""Domain model for the result of evaluating a candidate."""
from __future__ import annotations

import math
from dataclasses import dataclass

from dynamic_prompt_core.domain.models.candidate import Candidate


@dataclass(frozen=True)
class Judgment:
    """The result of evaluating one Candidate against a classification input.

    The score is a probability in [0, 1] derived from softmax over all
    candidates' mean log-probability scores.  Higher means stronger
    model-backed support for the candidate.  Scores across all candidates
    for one input sum to 1.0.
    """

    candidate: Candidate
    score: float

    def __post_init__(self) -> None:
        if math.isnan(self.score) or math.isinf(self.score):
            raise ValueError(
                f"score must be a finite number, got {self.score!r}"
            )
