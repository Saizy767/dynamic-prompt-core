"""Application-internal classification policy: turns judgments into a decision.

The ``ClassificationPolicy`` protocol is an application-internal strategy
abstraction, NOT an outbound port.  It is a pure decision rule with no
infrastructure coupling.  The concrete ``ArgmaxClassificationPolicy`` selects
the highest-scoring candidate; ties are broken uniformly at random.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from dynamic_prompt_core.domain.models.classification import Classification
from dynamic_prompt_core.domain.models.judgment import Judgment


@runtime_checkable
class ClassificationPolicy(Protocol):
    """Application strategy: convert judgments into a classification decision.

    The policy consumes ``Judgment`` objects and produces a ``Classification``.
    It performs no model inference, tokenization, logits extraction, prompt
    construction, or provider API calls.
    """

    def classify(self, judgments: list[Judgment]) -> Classification:
        """Return the classification decision derived from ``judgments``."""
        ...


class ArgmaxClassificationPolicy:
    """Select the candidate with the maximum score.

    Ties are broken uniformly at random among all candidates sharing the
    maximum score.  Empty judgments raise ``ValueError``.  The input
    judgment order is preserved in ``Classification.judgments``.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def classify(self, judgments: list[Judgment]) -> Classification:
        if not judgments:
            raise ValueError("judgments must not be empty")

        best_score = max(j.score for j in judgments)
        tied = [j for j in judgments if j.score == best_score]
        best = self._rng.choice(tied) if len(tied) > 1 else tied[0]

        return Classification(selected=best.candidate, judgments=tuple(judgments))
