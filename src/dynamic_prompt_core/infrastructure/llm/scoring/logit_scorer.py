"""Logit-to-score algorithm: mean candidate-token log-probability with causal alignment.

For a causal language model, logits at position ``i`` predict the token at
position ``i + 1``.  Given prefix tokens ``p_0 ... p_m`` and candidate tokens
``c_0 ... c_{n-1}``, the candidate-token log-probabilities are::

    candidate_logprob_0 = log_softmax(logits[m])[c_0_id]
    candidate_logprob_1 = log_softmax(logits[m+1])[c_1_id]
    ...
    candidate_logprob_{n-1} = log_softmax(logits[m+n-2])[c_{n-1}_id]

The score is the arithmetic mean of these log-probabilities.  Mean (not sum)
normalizes by candidate token count, avoiding systematic length bias.

The score is a ranking/evidence signal, not a probability.  No
softmax-to-probability claim is made.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedLogits,
    SequenceLogits,
)


def _log_softmax(logits: Sequence[float]) -> list[float]:
    """Compute log_softmax of a logit vector numerically stably."""
    values = list(logits)
    if not values:
        return []
    max_val = max(values)
    shifted = [v - max_val for v in values]
    log_sum_exp = math.log(sum(math.exp(v) for v in shifted))
    return [v - log_sum_exp for v in shifted]


class LogitScorer:
    """Compute candidate scores from model logits using causal alignment.

    Given ``prefix_token_count`` (the number of tokens before the candidate),
    ``candidate_token_ids`` (the candidate's token IDs), and the model's
    ``SequenceLogits``, the scorer extracts each candidate token's
    log-probability from the preceding position's logit distribution and
    returns their arithmetic mean.
    """

    def score(
        self,
        prefix_token_count: int,
        candidate_token_ids: list[int],
        sequence_logits: SequenceLogits,
    ) -> float:
        if not candidate_token_ids:
            raise LLMScoringError("candidate token IDs must not be empty")

        log_probs: list[float] = []
        for k, token_id in enumerate(candidate_token_ids):
            position = prefix_token_count + k - 1
            if position < 0 or position >= len(sequence_logits.logits):
                raise LLMScoringError(
                    f"logit position {position} out of range for "
                    f"sequence length {len(sequence_logits.logits)}"
                )
            position_logits = sequence_logits.logits[position]
            if token_id < 0 or token_id >= len(position_logits):
                raise LLMScoringError(
                    f"token id {token_id} out of range for vocab size "
                    f"{len(position_logits)}"
                )
            ls = _log_softmax(position_logits)
            log_probs.append(ls[token_id])

        result = sum(log_probs) / len(log_probs)

        if math.isnan(result) or math.isinf(result):
            raise LLMScoringError(
                f"computed score is not a finite number: {result!r}"
            )

        return result

    def score_batch(
        self,
        prefix_token_counts: list[int],
        candidate_token_ids: list[list[int]],
        batched_logits: BatchedLogits,
    ) -> list[float]:
        """Score all batch items using causal alignment and mean log-probability.

        Returns one score per batch item.  Padding tokens never contribute
        because each item's ``prefix_token_count`` and ``candidate_token_ids``
        are tracked independently from the unpadded sequence.  If any item
        produces an invalid score (NaN, infinity), raises ``LLMScoringError``
        for the entire batch — no partial results.
        """
        if len(prefix_token_counts) != len(candidate_token_ids):
            raise LLMScoringError(
                f"prefix_token_counts ({len(prefix_token_counts)}) and "
                f"candidate_token_ids ({len(candidate_token_ids)}) must have "
                "equal length"
            )
        if len(prefix_token_counts) != len(batched_logits.items):
            raise LLMScoringError(
                f"batch items ({len(prefix_token_counts)}) do not match "
                f"logits items ({len(batched_logits.items)})"
            )

        scores: list[float] = []
        for i, (prefix_count, cand_ids) in enumerate(
            zip(prefix_token_counts, candidate_token_ids, strict=True)
        ):
            item_logits = batched_logits.items[i]
            score = self.score(
                prefix_token_count=prefix_count,
                candidate_token_ids=cand_ids,
                sequence_logits=item_logits,
            )
            scores.append(score)

        return scores
