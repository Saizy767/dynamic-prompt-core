"""Concrete CandidateScorer backed by LLM logits.

Translates application-level scoring semantics into model mechanics: prompt
construction, tokenization, model inference, logit extraction, and score
calculation.  All model internals remain behind the ``CandidateScorer`` port
boundary.
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
    CausalLanguageModel,
)
from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
    ScoringPromptBuilder,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    TokenizerAdapter,
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
        tokenizer: TokenizerAdapter,
        model: CausalLanguageModel,
        logit_scorer: LogitScorer,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._tokenizer = tokenizer
        self._model = model
        self._logit_scorer = logit_scorer

    async def score(
        self,
        text: str,
        candidates: list[Candidate],
    ) -> list[Judgment]:
        """Evaluate candidates against ``text`` using model logits.

        Returns exactly one ``Judgment`` per candidate in input order.
        Raises ``CandidateScoringError`` if any candidate cannot be evaluated.
        """
        judgments: list[Judgment] = []
        for candidate in candidates:
            judgment = await self._score_one(text, candidate)
            judgments.append(judgment)
        return judgments

    async def _score_one(self, text: str, candidate: Candidate) -> Judgment:
        try:
            prefix = self._prompt_builder.build_prefix(text)
            candidate_text = self._prompt_builder.build_candidate(candidate)

            prefix_token_ids = self._tokenizer.encode(prefix)
            candidate_token_ids = self._tokenizer.encode(candidate_text)
            prefix_token_count = len(prefix_token_ids)

            full_sequence = prefix_token_ids + candidate_token_ids

            sequence_logits = await asyncio.to_thread(
                self._model.forward, full_sequence
            )

            score = self._logit_scorer.score(
                prefix_token_count=prefix_token_count,
                candidate_token_ids=candidate_token_ids,
                sequence_logits=sequence_logits,
            )

            return Judgment(candidate=candidate, score=score)

        except LLMScoringError as exc:
            raise CandidateScoringError(
                f"scoring failed for candidate {candidate.value!r}: {exc}"
            ) from exc
        except CandidateScoringError:
            raise
        except Exception as exc:
            raise CandidateScoringError(
                f"scoring failed for candidate {candidate.value!r}: {exc}"
            ) from exc
