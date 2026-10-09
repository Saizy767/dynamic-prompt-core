"""Concrete CandidateScorer backed by LLM logits.

Translates application-level scoring semantics into model mechanics: prompt
construction, tokenization, model inference, logit extraction, and score
calculation.  All model internals remain behind the ``CandidateScorer`` port
boundary.

Batch scoring evaluates all candidates from one ``score()`` invocation through
a single batched model forward pass.  The application sees only
``await scorer.score(text, candidates)`` and cannot tell whether batching
occurred.
"""
from __future__ import annotations

import asyncio
import logging

from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment
from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedCausalLanguageModel,
)
from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
    ScoringPromptBuilder,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenizerAdapter,
)

log = logging.getLogger(__name__)


class LLMLogitCandidateScorer:
    """CandidateScorer implementation that derives scores from LLM logits.

    Constructor dependencies are infrastructure-specific collaborators that do
    not appear in the ``CandidateScorer`` port.  The application sees only
    ``await scorer.score(text, candidates)``.
    """

    def __init__(
        self,
        prompt_builder: ScoringPromptBuilder,
        tokenizer: BatchTokenizerAdapter,
        model: BatchedCausalLanguageModel,
        logit_scorer: LogitScorer,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._tokenizer = tokenizer
        self._model = model
        self._logit_scorer = logit_scorer

    def describe(self) -> dict[str, object]:
        """Return backend metadata from the model adapter (infrastructure-internal).

        Delegates to the model adapter's ``describe()`` if available.  This is
        NOT part of the ``CandidateScorer`` port; it is an infrastructure-internal
        hook used by the composition root to build run metadata.
        """
        describe_fn = getattr(self._model, "describe", None)
        if describe_fn is not None:
            return dict(describe_fn())
        return {}

    async def score(
        self,
        text: str,
        candidates: list[Candidate],
    ) -> list[Judgment]:
        """Evaluate candidates against ``text`` using model logits.

        Returns exactly one ``Judgment`` per candidate in input order.
        Raises ``CandidateScoringError`` if any candidate cannot be evaluated.
        """
        try:
            prefix = self._prompt_builder.build_prefix(text)
            candidate_texts = [
                self._prompt_builder.build_candidate(c) for c in candidates
            ]
            prefixes = [prefix] * len(candidates)

            log.info(
                "CandidateScorer.score: text=%d chars, %d candidates",
                len(text), len(candidates),
            )
            batch = self._tokenizer.encode_batch(prefixes, candidate_texts)

            batched_logits = await asyncio.to_thread(
                self._model.forward_batch,
                batch.input_ids,
                batch.attention_mask,
            )

            scores = self._logit_scorer.score_batch(
                prefix_token_counts=batch.prefix_token_counts,
                candidate_token_ids=batch.candidate_token_ids,
                batched_logits=batched_logits,
            )

            for candidate, score in zip(candidates, scores, strict=True):
                log.debug(
                    "CandidateScorer: candidate=%r score=%.6f",
                    candidate.value, score,
                )

            return [
                Judgment(candidate=candidate, score=score)
                for candidate, score in zip(candidates, scores, strict=True)
            ]

        except LLMScoringError as exc:
            raise CandidateScoringError(f"batch scoring failed: {exc}") from exc
        except CandidateScoringError:
            raise
        except Exception as exc:
            raise CandidateScoringError(f"batch scoring failed: {exc}") from exc
