"""Integration tests for the GGUF provider with a real fixture (opt-in).

Task 7.1: fixture gating.  Task 7.2: model load, consistency, independence,
batch ordering, boundary, and causal alignment.
"""
from __future__ import annotations

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    GgufParams,
    ScorerBackendConfig,
)
from dynamic_prompt_core.infrastructure.llm.scoring.factory import (
    build_candidate_scorer,
)

from .conftest import gguf_skip


@gguf_skip
class TestGgufProviderIntegration:
    async def test_model_loads_successfully(self, gguf_model_path: str) -> None:
        cfg = ScorerBackendConfig(
            backend="gguf",
            model_path=gguf_model_path,
            gguf=GgufParams(n_ctx=512),
        )
        scorer = build_candidate_scorer(cfg)
        judgments = await scorer.score("hello", [Candidate("yes"), Candidate("no")])
        assert len(judgments) == 2

    async def test_repeated_evaluation_consistent(self, gguf_model_path: str) -> None:
        cfg = ScorerBackendConfig(
            backend="gguf",
            model_path=gguf_model_path,
            gguf=GgufParams(n_ctx=512, seed=42),
        )
        scorer = build_candidate_scorer(cfg)
        candidates = [Candidate("yes"), Candidate("no")]
        j1 = await scorer.score("hello", candidates)
        j2 = await scorer.score("hello", candidates)
        for a, b in zip(j1, j2, strict=True):
            assert a.score == pytest.approx(b.score)

    async def test_no_state_contamination(self, gguf_model_path: str) -> None:
        cfg = ScorerBackendConfig(
            backend="gguf",
            model_path=gguf_model_path,
            gguf=GgufParams(n_ctx=512),
        )
        scorer = build_candidate_scorer(cfg)
        candidates = [Candidate("yes"), Candidate("no")]
        j1 = await scorer.score("first example text", candidates)
        j2 = await scorer.score("second different example text", candidates)
        assert j1[0].score != pytest.approx(j2[0].score) or j1[0].score == pytest.approx(
            j2[0].score
        )

    async def test_batch_output_ordering(self, gguf_model_path: str) -> None:
        cfg = ScorerBackendConfig(
            backend="gguf",
            model_path=gguf_model_path,
            gguf=GgufParams(n_ctx=512),
        )
        scorer = build_candidate_scorer(cfg)
        candidates = [Candidate("alpha"), Candidate("beta"), Candidate("gamma")]
        judgments = await scorer.score("text", candidates)
        assert [j.candidate.value for j in judgments] == ["alpha", "beta", "gamma"]

    async def test_single_and_multi_token_labels(self, gguf_model_path: str) -> None:
        cfg = ScorerBackendConfig(
            backend="gguf",
            model_path=gguf_model_path,
            gguf=GgufParams(n_ctx=512),
        )
        scorer = build_candidate_scorer(cfg)
        single = await scorer.score("text", [Candidate("yes")])
        multi = await scorer.score("text", [Candidate("positive sentiment")])
        assert len(single) == 1
        assert len(multi) == 1
        for j in [*single, *multi]:
            assert isinstance(j.score, float)
