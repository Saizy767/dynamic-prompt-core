"""Typed dependency object for the ClassifyInput use case."""
from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.application.ports.outbound.candidate_scorer import (
    CandidateScorer,
)
from dynamic_prompt_core.application.services.classification_policy import (
    ClassificationPolicy,
)


@dataclass(frozen=True)
class ClassifyInputDeps:
    """Dependencies for the classify_input use case.

    ``scorer`` is the outbound port for candidate evaluation.
    ``policy`` is the application-internal classification strategy.
    """

    scorer: CandidateScorer
    policy: ClassificationPolicy
