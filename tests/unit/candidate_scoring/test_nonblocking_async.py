"""Test that the real LLMLogitCandidateScorer does not block the event loop.

Uses a deliberately blocking fake model adapter to prove that the real
infrastructure boundary offloads blocking work via asyncio.to_thread.
Does not assert the specific executor mechanism.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer import (
    LLMLogitCandidateScorer,
)
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedLogits,
    SequenceLogits,
)
from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
    ScoringPromptBuilder,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenization,
)

VOCAB_SIZE = 4


class _BlockingModel:
    """Fake batched model that blocks on a threading.Event in forward_batch.

    This simulates a blocking model inference call. If the scorer runs this
    directly on the event loop, no other asyncio task can make progress.
    If the scorer offloads it (e.g., via asyncio.to_thread), the event loop
    remains free.
    """

    def __init__(self, block_event: threading.Event) -> None:
        self._block_event = block_event

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        self._block_event.wait()
        uniform = [1.0 / VOCAB_SIZE] * VOCAB_SIZE
        items = tuple(
            SequenceLogits(logits=tuple(tuple(uniform) for _ in seq))
            for seq in input_ids
        )
        return BatchedLogits(items=items)


class _StubTokenizer:
    """Minimal fake batch tokenizer returning deterministic tokenization."""

    def encode_batch(
        self,
        prefixes: list[str],
        candidates: list[str],
    ) -> BatchTokenization:
        batch_size = len(prefixes)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        prefix_token_counts: list[int] = []
        candidate_token_ids: list[list[int]] = []

        for i in range(batch_size):
            prefix_ids = [1, 2, 3]
            candidate_ids = [ord(c) % VOCAB_SIZE for c in candidates[i]]
            seq = prefix_ids + candidate_ids
            input_ids.append(seq)
            attention_mask.append([1] * len(seq))
            prefix_token_counts.append(len(prefix_ids))
            candidate_token_ids.append(candidate_ids)

        return BatchTokenization(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_token_counts=prefix_token_counts,
            candidate_token_ids=candidate_token_ids,
        )


@pytest.mark.asyncio
async def test_scorer_does_not_block_event_loop() -> None:
    """The real LLMLogitCandidateScorer must not synchronously block the
    event loop on model inference.

    A fake model adapter deliberately blocks on a threading.Event. While
    scoring is awaited, a concurrent asyncio task must make progress,
    proving blocking inference is offloaded off the event loop.
    """
    block_event = threading.Event()
    model = _BlockingModel(block_event)
    tokenizer = _StubTokenizer()
    scorer = LLMLogitCandidateScorer(
        prompt_builder=ScoringPromptBuilder(),
        tokenizer=tokenizer,
        model=model,
        logit_scorer=LogitScorer(),
    )

    candidates = [Candidate("0"), Candidate("1")]
    concurrent_progress = False

    async def _concurrent_task() -> None:
        nonlocal concurrent_progress
        await asyncio.sleep(0.01)
        concurrent_progress = True

    scoring_task = asyncio.create_task(scorer.score("text", candidates))
    progress_task = asyncio.create_task(_concurrent_task())

    await asyncio.sleep(0.05)

    assert concurrent_progress, (
        "Event loop was blocked during scoring — the concurrent task did not "
        "make progress while the model inference was blocking"
    )

    block_event.set()
    judgments = await scoring_task
    await progress_task

    assert len(judgments) == 2
    assert judgments[0].candidate == Candidate("0")
    assert judgments[1].candidate == Candidate("1")
