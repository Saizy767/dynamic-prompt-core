"""Opt-in integration test for real-model batch scoring.

Gated by the ``REAL_MODEL_TEST`` environment variable. Skipped by default.
Run with: ``REAL_MODEL_TEST=1 pytest tests/unit/candidate_scoring/test_batch_integration.py``
"""
from __future__ import annotations

import math
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("REAL_MODEL_TEST"),
    reason="Set REAL_MODEL_TEST=1 to run real-model batch scoring integration test",
)


@pytest.fixture
def real_model_name() -> str:
    return os.environ.get("REAL_MODEL_NAME", "gpt2")


class TestRealModelBatchScoring:
    async def test_batch_scoring_with_real_model(self, real_model_name: str) -> None:
        from transformers import AutoModelForCausalLM

        from dynamic_prompt_core.domain.models.candidate import Candidate
        from dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer import (
            LLMLogitCandidateScorer,
        )
        from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
        from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
            TorchBatchedCausalLMAdapter,
        )
        from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
            ScoringPromptBuilder,
        )
        from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
            HuggingFaceBatchTokenizerAdapter,
        )

        tokenizer = HuggingFaceBatchTokenizerAdapter(real_model_name)
        model = AutoModelForCausalLM.from_pretrained(real_model_name)
        batch_model = TorchBatchedCausalLMAdapter(model)

        scorer = LLMLogitCandidateScorer(
            prompt_builder=ScoringPromptBuilder(),
            tokenizer=tokenizer,
            model=batch_model,
            logit_scorer=LogitScorer(),
        )

        text = "The company introduced new reporting requirements."
        candidates = [
            Candidate("financial regulation"),
            Candidate("sports"),
            Candidate("weather"),
        ]

        judgments = await scorer.score(text, candidates)

        assert len(judgments) == 3
        for j in judgments:
            assert math.isfinite(j.score)

        assert [j.candidate.value for j in judgments] == [
            "financial regulation",
            "sports",
            "weather",
        ]
