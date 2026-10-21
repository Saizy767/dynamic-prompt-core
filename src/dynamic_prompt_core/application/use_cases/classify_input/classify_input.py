"""ClassifyInput use case: validate → score → classify."""
from __future__ import annotations

from collections.abc import Sequence

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
    rules: Sequence[str],
) -> Classification:
    """Score candidates against ``text`` and classify the result.

    ``rules`` are the active classification rules from the selected prompt
    version's ``PromptArtifact.layers.rules``.

    Validates candidate uniqueness at the application boundary, invokes the
    scorer to produce judgments, then invokes the policy to produce a
    classification decision.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    validate_unique_candidates(candidates)

    judgments = await deps.scorer.score(text, candidates, rules)

    return deps.policy.classify(judgments)
