"""Unit tests for BaselineRunner classification via candidate scoring."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamic_prompt_core.application.services.classification_policy import (
    ArgmaxClassificationPolicy,
)
from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
    BaselineRunner,
    ResultRow,
    RunnerConfig,
)
from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment
from dynamic_prompt_core.infrastructure.llm import CallResult, ResponseStatus


@dataclass
class _FakeRecord:
    id: int
    text: str
    label: int


def _make_candidates() -> list[Candidate]:
    return [Candidate("0"), Candidate("1")]


def _make_extract_cr() -> CallResult:
    return CallResult(
        parsed=None,
        raw_content=None,
        latency_ms=1.0,
        parse_status=None,
        status=ResponseStatus.UNEXPECTED_SHAPE,
        error="mock",
    )


def _make_runner(
    scorer: Any = None,
    policy: Any = None,
    candidates: list[Candidate] | None = None,
) -> BaselineRunner:
    task = MagicMock()
    task.extract_theses_detailed = AsyncMock(return_value=_make_extract_cr())
    config = RunnerConfig()
    runner = BaselineRunner(
        task=task,
        config=config,
        split="dev",
        scorer=scorer or AsyncMock(),
        policy=policy or ArgmaxClassificationPolicy(),
        candidates=candidates or _make_candidates(),
    )
    runner._checkpoint_write = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_process_example_classifies_via_candidate_scoring() -> None:
    """The runner uses CandidateScorer.score -> ClassificationPolicy.classify."""
    candidates = _make_candidates()
    scorer = AsyncMock()
    scorer.score.return_value = [
        Judgment(candidate=candidates[0], score=-1.5),
        Judgment(candidate=candidates[1], score=-0.3),
    ]
    policy = ArgmaxClassificationPolicy()
    runner = _make_runner(scorer=scorer, policy=policy, candidates=candidates)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=1, text="some text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    scorer.score.assert_called_once()
    row = results[0]
    assert row is not None
    assert row.classify_status == "ok"
    assert row.predicted_decision == 1
    assert row.selected_candidate == "1"
    assert "0" in row.judgment_scores
    assert "1" in row.judgment_scores


@pytest.mark.asyncio
async def test_process_example_scoring_failure_maps_to_failed() -> None:
    """CandidateScoringError maps to classify_status=failed."""
    scorer = AsyncMock()
    scorer.score.side_effect = CandidateScoringError("scoring failed")
    runner = _make_runner(scorer=scorer)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=2, text="text", label=0)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    row = results[0]
    assert row is not None
    assert row.classify_status == "failed"
    assert row.predicted_decision is None
    assert row.selected_candidate is None
    assert row.judgment_scores == {}


@pytest.mark.asyncio
async def test_process_example_candidate_ordering_preserved() -> None:
    """Judgment scores preserve candidate ordering from the scorer."""
    candidates = [Candidate("alpha"), Candidate("beta"), Candidate("gamma")]
    scorer = AsyncMock()
    scorer.score.return_value = [
        Judgment(candidate=candidates[0], score=-0.5),
        Judgment(candidate=candidates[1], score=-0.1),
        Judgment(candidate=candidates[2], score=-0.8),
    ]
    runner = _make_runner(scorer=scorer, candidates=candidates)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=3, text="text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    row = results[0]
    assert row is not None
    assert row.selected_candidate == "beta"
    assert row.predicted_decision is None
    assert list(row.judgment_scores.keys()) == ["alpha", "beta", "gamma"]


def test_result_row_schema_has_no_confidence_or_raw_classify() -> None:
    """ResultRow must not have confidence or raw_classify fields."""
    row = ResultRow(id=1, text="text", true_label=0)
    d = row.to_dict()
    assert "confidence" not in d
    assert "raw_classify" not in d
    assert "selected_candidate" in d
    assert "judgment_scores" in d
    assert "predicted_decision" in d


@pytest.mark.asyncio
async def test_predicted_decision_integer_conversion() -> None:
    """Candidate("1") -> predicted_decision == 1 (int conversion succeeds)."""
    candidates = [Candidate("0"), Candidate("1")]
    scorer = AsyncMock()
    scorer.score.return_value = [
        Judgment(candidate=candidates[0], score=0.2),
        Judgment(candidate=candidates[1], score=0.8),
    ]
    runner = _make_runner(scorer=scorer, candidates=candidates)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=1, text="text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    row = results[0]
    assert row is not None
    assert row.predicted_decision == 1
    assert row.selected_candidate == "1"


@pytest.mark.asyncio
async def test_predicted_decision_non_integer_falls_back_to_none() -> None:
    """A non-integer selected value -> predicted_decision == None."""
    candidates = [Candidate("yes"), Candidate("no")]
    scorer = AsyncMock()
    scorer.score.return_value = [
        Judgment(candidate=candidates[0], score=0.8),
        Judgment(candidate=candidates[1], score=0.2),
    ]
    runner = _make_runner(scorer=scorer, candidates=candidates)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=2, text="text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    row = results[0]
    assert row is not None
    assert row.predicted_decision is None
    assert row.selected_candidate == "yes"


@pytest.mark.asyncio
async def test_judgment_scores_populated_from_classification_judgments() -> None:
    """judgment_scores is populated from Classification.judgments."""
    candidates = [Candidate("0"), Candidate("1")]
    scorer = AsyncMock()
    scorer.score.return_value = [
        Judgment(candidate=candidates[0], score=-1.5),
        Judgment(candidate=candidates[1], score=-0.3),
    ]
    runner = _make_runner(scorer=scorer, candidates=candidates)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=3, text="text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    row = results[0]
    assert row is not None
    assert row.judgment_scores == {"0": -1.5, "1": -0.3}
