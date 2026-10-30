"""Domain model for the result of evaluating a candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dynamic_prompt_core.domain.models.candidate import Candidate


@dataclass(frozen=True)
class Judgment:
    """The result of evaluating one Candidate against a classification input.

    The score is an opaque numeric ranking signal: higher means stronger
    model-backed support for the candidate.  Only the ORDER of scores across
    candidates carries meaning — the infrastructure currently stores a
    softmax normalization of the mean log-probabilities (so scores happen to
    lie in [0, 1] and sum to 1.0), but no domain or application logic SHALL
    interpret score magnitudes as probabilities or confidence.
    """

    candidate: Candidate
    score: float

    def __post_init__(self) -> None:
        if math.isnan(self.score) or math.isinf(self.score):
            raise ValueError(f"score must be a finite number, got {self.score!r}")
