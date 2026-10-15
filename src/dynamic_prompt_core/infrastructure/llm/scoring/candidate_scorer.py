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
import inspect
import logging
import math
import time

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


def _softmax(scores: list[float]) -> list[float]:
    """Convert raw log-probability scores to probabilities via softmax.

    ``P(i) = e^{z_i} / sum(e^{z_j})``

    Returns values in [0, 1] that sum to 1.0.  Numerically stable:
    subtracts ``max(z)`` before exponentiation.
    """
    if not scores:
        return []
    max_z = max(scores)
    exps = [math.exp(z - max_z) for z in scores]
    total = sum(exps)
    return [e / total for e in exps]


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

    async def aclose(self) -> None:
        """Close HTTP sessions on adapters that own them.

        Calls ``aclose()`` on the model and tokenizer adapters if they
        expose it.  Safe to call on adapters without ``aclose()`` (Torch,
        HuggingFace, LlamaCpp) — ``getattr`` returns ``None``.
        """
        for adapter in (self._model, self._tokenizer):
            close_fn = getattr(adapter, "aclose", None)
            if close_fn is not None:
                await close_fn()

    async def score(
        self,
        text: str,
        candidates: list[Candidate],
    ) -> list[Judgment]:
        """Evaluate candidates against ``text`` using model logits.

        Returns exactly one ``Judgment`` per candidate in input order.
        Raises ``CandidateScoringError`` if any candidate cannot be evaluated.

        Timing relationship: ``forward`` is the wall-clock of the
        ``asyncio.to_thread`` call (the aggregate, including thread-offload
        overhead).  The adapter sub-stages (lock_wait + reset + eval +
        logits_read + logits_convert + logits_validate) decompose the
        in-thread work and are reported separately — they are NOT added to
        ``forward``, so no interval is double-counted.  ``scoring_ms`` is the
        logit-to-score post-processing measured independently of the forward
        call.
        """
        try:
            t_total = time.perf_counter()
            log.info(
                "CandidateScorer.score: START text=%d chars, %d candidates",
                len(text),
                len(candidates),
            )

            t0 = time.perf_counter()
            prefix = self._prompt_builder.build_prefix(text)
            candidate_texts = [self._prompt_builder.build_candidate(c) for c in candidates]
            prefixes = [prefix] * len(candidates)
            t_prompt = time.perf_counter() - t0

            t0 = time.perf_counter()
            encode_fn = self._tokenizer.encode_batch
            if inspect.iscoroutinefunction(encode_fn):
                batch = await encode_fn(prefixes, candidate_texts)
            else:
                batch = encode_fn(prefixes, candidate_texts)
            t_tokenize = time.perf_counter() - t0

            t0 = time.perf_counter()
            forward_fn = self._model.forward_batch
            if inspect.iscoroutinefunction(forward_fn):
                batched_logits = await forward_fn(batch.input_ids, batch.attention_mask)
            else:
                batched_logits = await asyncio.to_thread(
                    forward_fn,
                    batch.input_ids,
                    batch.attention_mask,
                )
            t_forward = time.perf_counter() - t0

            t0 = time.perf_counter()
            scores = self._logit_scorer.score_batch(
                prefix_token_counts=batch.prefix_token_counts,
                candidate_token_ids=batch.candidate_token_ids,
                batched_logits=batched_logits,
            )
            t_logit = time.perf_counter() - t0

            adapter_profile = batched_logits.profile
            if adapter_profile is not None:
                adapter_profile.scoring_ms = t_logit * 1e3
                adapter_profile.scored_tokens = sum(len(c) for c in batch.candidate_token_ids)

            for candidate, score in zip(candidates, scores, strict=True):
                log.debug(
                    "CandidateScorer: candidate=%r raw_score=%.6f",
                    candidate.value,
                    score,
                )

            probabilities = _softmax(scores)

            t_total = time.perf_counter() - t_total
            if adapter_profile is not None:
                p = adapter_profile
                log.info(
                    "CandidateScorer.score: END text=%d chars, %d candidates "
                    "total=%.2fms | prompt=%.2fms tokenize=%.2fms "
                    "forward=%.2fms logit=%.2fms "
                    "| adapter: lock=%.2fms reset=%.2fms eval=%.2fms "
                    "logits_read=%.2fms convert=%.2fms validate=%.2fms "
                    "scoring=%.2fms | tokens: input=%d scored=%d "
                    "vocab=%d candidates=%d",
                    len(text),
                    len(candidates),
                    t_total * 1e3,
                    t_prompt * 1e3,
                    t_tokenize * 1e3,
                    t_forward * 1e3,
                    t_logit * 1e3,
                    p.lock_wait_ms,
                    p.reset_ms,
                    p.eval_ms,
                    p.logits_read_ms,
                    p.logits_convert_ms,
                    p.logits_validate_ms,
                    p.scoring_ms,
                    p.input_tokens,
                    p.scored_tokens,
                    p.vocab_size,
                    p.candidate_count,
                )
            else:
                log.info(
                    "CandidateScorer.score: END text=%d chars, %d candidates "
                    "total=%.2fms | prompt=%.2fms tokenize=%.2fms "
                    "forward=%.2fms logit=%.2fms",
                    len(text),
                    len(candidates),
                    t_total * 1e3,
                    t_prompt * 1e3,
                    t_tokenize * 1e3,
                    t_forward * 1e3,
                    t_logit * 1e3,
                )

            return [
                Judgment(candidate=candidate, score=prob)
                for candidate, prob in zip(candidates, probabilities, strict=True)
            ]

        except LLMScoringError as exc:
            raise CandidateScoringError(f"batch scoring failed: {exc}") from exc
        except CandidateScoringError:
            raise
        except Exception as exc:
            raise CandidateScoringError(f"batch scoring failed: {exc}") from exc
