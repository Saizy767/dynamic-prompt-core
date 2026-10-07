"""ClassifyInput use case: validate → score → classify."""
from __future__ import annotations

from dynamic_prompt_core.application.ports.outbound.candidate_scorer import (
    validate_unique_candidates,
)
from dynamic_prompt_core.application.use_cases.classify_input.deps import (
    ClassifyInputDeps,
)
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification import Classification


async def classify_input(
    deps: ClassifyInputDeps,
    text: str,
    candidates: list[Candidate],
) -> Classification:
    """Score candidates against ``text`` and classify the result.

    Validates candidate uniqueness at the application boundary, invokes the
    scorer to produce judgments, then invokes the policy to produce a
    classification decision.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    validate_unique_candidates(candidates)

    judgments = await deps.scorer.score(text, candidates)

    return deps.policy.classify(judgments)
