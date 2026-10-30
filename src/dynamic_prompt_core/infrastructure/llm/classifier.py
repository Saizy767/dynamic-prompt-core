"""Concrete Classifier backed by LLM logits.

Encapsulates the full classification process: rule validation, candidate
handling, logit-based scoring, and final decision.  Returns a
``Classification`` directly.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from typing import Any

from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification import Classification
from dynamic_prompt_core.domain.models.classification_rules import ClassificationRules
from dynamic_prompt_core.domain.models.judgment import Judgment
from dynamic_prompt_core.infrastructure.llm.scoring.classifier_prompt_builder import (
    ClassifierPromptBuilder,
)
from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedCausalLanguageModel,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenizerAdapter,
)

log = logging.getLogger(__name__)


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_z = max(scores)
    exps = [math.exp(z - max_z) for z in scores]
    total = sum(exps)
    return [e / total for e in exps]


class LLMLogitClassifier:
    """Classifier implementation that derives scores from LLM logits.

    Absorbs the responsibilities previously held by ``CandidateScorer`` and
    ``ClassificationPolicy``: prompt construction, tokenization, model
    inference, logit extraction, score calculation, softmax, and argmax
    decision with input-order tie-breaking.
    """

    def __init__(
        self,
        prompt_builder: ClassifierPromptBuilder,
        tokenizer: BatchTokenizerAdapter,
        model: BatchedCausalLanguageModel,
        logit_scorer: LogitScorer,
        candidates: list[Candidate],
    ) -> None:
        self._prompt_builder = prompt_builder
        self._tokenizer = tokenizer
        self._model = model
        self._logit_scorer = logit_scorer
        self._candidates = candidates

    def describe(self) -> dict[str, object]:
        describe_fn = getattr(self._model, "describe", None)
        if describe_fn is not None:
            return dict(describe_fn())
        return {}

    async def aclose(self) -> None:
        for adapter in (self._model, self._tokenizer):
            close_fn = getattr(adapter, "aclose", None)
            if close_fn is not None:
                await close_fn()

    async def classify(
        self,
        text: str,
        rules: ClassificationRules,
    ) -> Classification:
        """Classify ``text`` using logit-based scoring.

        Returns a ``Classification`` with the selected candidate and full
        judgment set.  Raises ``CandidateScoringError`` on any failure.
        """
        try:
            log.info(
                "Classifier.classify: START text=%d chars, %d candidates, %d rules",
                len(text),
                len(self._candidates),
                len(rules.rules),
            )

            prefix = self._prompt_builder.build_prefix(text, rules.rules)
            candidate_texts = [self._prompt_builder.build_candidate(c) for c in self._candidates]
            prefixes = [prefix] * len(self._candidates)

            encode_fn = self._tokenizer.encode_batch
            if inspect.iscoroutinefunction(encode_fn):
                batch = await encode_fn(prefixes, candidate_texts)
            else:
                batch = encode_fn(prefixes, candidate_texts)

            forward_fn = self._model.forward_batch
            forward_kwargs: dict[str, Any] = {
                "prefix_token_counts": batch.prefix_token_counts,
                "candidate_token_ids": batch.candidate_token_ids,
            }
            if inspect.iscoroutinefunction(forward_fn):
                batched_logits = await forward_fn(
                    batch.input_ids,
                    batch.attention_mask,
                    **forward_kwargs,
                )
            else:
                batched_logits = await asyncio.to_thread(
                    forward_fn,
                    batch.input_ids,
                    batch.attention_mask,
                    **forward_kwargs,
                )

            scores = self._logit_scorer.score_batch(
                prefix_token_counts=batch.prefix_token_counts,
                candidate_token_ids=batch.candidate_token_ids,
                batched_logits=batched_logits,
            )

            probabilities = _softmax(scores)

            judgments = [
                Judgment(candidate=c, score=prob)
                for c, prob in zip(self._candidates, probabilities, strict=True)
            ]

            best = max(judgments, key=lambda j: j.score)
            classification = Classification(selected=best.candidate, judgments=tuple(judgments))

            log.info(
                "Classifier.classify: END selected=%s scores=%s",
                best.candidate.value,
                {c.value: f"{p:.6f}" for c, p in zip(self._candidates, probabilities, strict=True)},
            )

            return classification

        except LLMScoringError as exc:
            raise CandidateScoringError(f"classification failed: {exc}") from exc
        except CandidateScoringError:
            raise
        except Exception as exc:
            raise CandidateScoringError(f"classification failed: {exc}") from exc
