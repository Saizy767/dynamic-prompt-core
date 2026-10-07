"""Application-internal classification policy: turns judgments into a decision.

The ``ClassificationPolicy`` protocol is an application-internal strategy
abstraction, NOT an outbound port.  It is a pure decision rule with no
infrastructure coupling.  The concrete ``ArgmaxClassificationPolicy`` selects
the highest-scoring candidate with deterministic first-in-order tie-breaking.
"""
from __future__ import annotations

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

    Ties are broken deterministically: the judgment appearing first in the
    input order wins.  Empty judgments raise ``ValueError``.  The input
    judgment order is preserved in ``Classification.judgments``.
    """

    def classify(self, judgments: list[Judgment]) -> Classification:
        if not judgments:
            raise ValueError("judgments must not be empty")

        best = judgments[0]
        for judgment in judgments[1:]:
            if judgment.score > best.score:
                best = judgment

        return Classification(selected=best.candidate, judgments=tuple(judgments))
