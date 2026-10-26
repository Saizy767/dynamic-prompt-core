"""Unit tests for BaselineRunner classification via the Classifier port."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamic_prompt_core.application.ports.outbound.classifier import Classifier
from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
    BaselineRunner,
    ResultRow,
    RunnerConfig,
)
from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification import Classification
from dynamic_prompt_core.domain.models.classification_rules import ClassificationRules
from dynamic_prompt_core.domain.models.judgment import Judgment
from dynamic_prompt_core.domain.prompts import CLASSIFICATION_PROMPT_V0, PromptArtifact
from dynamic_prompt_core.domain.prompts.base import PromptLayer
from dynamic_prompt_core.infrastructure.llm import CallResult, ResponseStatus
from dynamic_prompt_core.infrastructure.llm.extractor import ExtractionResult, Extractor


@dataclass
class _FakeRecord:
    id: int
    text: str
    label: int


def _make_candidates() -> list[Candidate]:
    return [Candidate("0"), Candidate("1")]


def _prompt_with_rules(rules: list[str] | None = None) -> PromptArtifact:
    layers = PromptLayer(
        role="role", task="task",
        rules=rules or ["rule a", "rule b", "rule c"],
        output_contract="contract", fallback="fallback",
    )
    return PromptArtifact(version="test", layers=layers, text="test")


def _make_extract_cr() -> CallResult:
    return CallResult(
        parsed=None,
        raw_content=None,
        latency_ms=1.0,
        parse_status=None,
        status=ResponseStatus.UNEXPECTED_SHAPE,
        error="mock",
    )


def _make_extract_result(status: str = "ok") -> ExtractionResult:
    return ExtractionResult(
        theses_raw=["a b"],
        theses_norm=["a b"],
        status=status,
        latency_ms=1.0,
        raw_extract=None,
        call_result=_make_extract_cr(),
    )


def _default_classification(candidates: list[Candidate]) -> Classification:
    return Classification(
        selected=candidates[1],
        judgments=(
            Judgment(candidate=candidates[0], score=-1.5),
            Judgment(candidate=candidates[1], score=-0.3),
        ),
    )


class FakeClassifier:
    """Stub Classifier returning a predetermined Classification or raising."""

    def __init__(
        self,
        classification: Classification | None = None,
        error: Exception | None = None,
    ) -> None:
        self.classify = AsyncMock()
        if error is not None:
            self.classify.side_effect = error
        else:
            self.classify.return_value = classification


class FakeExtractor:
    """Stub Extractor returning a predetermined ExtractionResult."""

    def __init__(self, result: ExtractionResult | None = None) -> None:
        self.extract = AsyncMock(return_value=result)


def _make_runner(
    classifier: Any = None,
    extractor: Any = None,
    candidates: list[Candidate] | None = None,
    **kwargs: Any,
) -> BaselineRunner:
    cands = candidates or _make_candidates()
    config = RunnerConfig()
    runner = BaselineRunner(
        classifier=classifier or FakeClassifier(_default_classification(cands)),
        extractor=extractor or FakeExtractor(_make_extract_result()),
        config=config,
        split="dev",
        candidates=cands,
        classify_prompt=_prompt_with_rules(),
        **kwargs,
    )
    runner._checkpoint_write = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_process_example_classifies_via_candidate_scoring() -> None:
    """The runner uses Classifier.classify -> Classification."""
    candidates = _make_candidates()
    classification = Classification(
        selected=candidates[1],
        judgments=(
            Judgment(candidate=candidates[0], score=-1.5),
            Judgment(candidate=candidates[1], score=-0.3),
        ),
    )
    classifier = FakeClassifier(classification)
    runner = _make_runner(classifier=classifier, candidates=candidates)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=1, text="some text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    classifier.classify.assert_called_once()
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
    classifier = FakeClassifier(error=CandidateScoringError("scoring failed"))
    runner = _make_runner(classifier=classifier)

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
    """Judgment scores preserve candidate ordering from the classifier."""
    candidates = [Candidate("alpha"), Candidate("beta"), Candidate("gamma")]
    classification = Classification(
        selected=candidates[1],
        judgments=(
            Judgment(candidate=candidates[0], score=-0.5),
            Judgment(candidate=candidates[1], score=-0.1),
            Judgment(candidate=candidates[2], score=-0.8),
        ),
    )
    classifier = FakeClassifier(classification)
    runner = _make_runner(classifier=classifier, candidates=candidates)

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
    classification = Classification(
        selected=candidates[1],
        judgments=(
            Judgment(candidate=candidates[0], score=0.2),
            Judgment(candidate=candidates[1], score=0.8),
        ),
    )
    classifier = FakeClassifier(classification)
    runner = _make_runner(classifier=classifier, candidates=candidates)

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
    classification = Classification(
        selected=candidates[0],
        judgments=(
            Judgment(candidate=candidates[0], score=0.8),
            Judgment(candidate=candidates[1], score=0.2),
        ),
    )
    classifier = FakeClassifier(classification)
    runner = _make_runner(classifier=classifier, candidates=candidates)

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
    classification = Classification(
        selected=candidates[1],
        judgments=(
            Judgment(candidate=candidates[0], score=-1.5),
            Judgment(candidate=candidates[1], score=-0.3),
        ),
    )
    classifier = FakeClassifier(classification)
    runner = _make_runner(classifier=classifier, candidates=candidates)

    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=3, text="text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    row = results[0]
    assert row is not None
    assert row.judgment_scores == {"0": -1.5, "1": -0.3}


@pytest.mark.asyncio
async def test_runner_skip_extraction() -> None:
    """When skip_extraction=True, extraction is skipped and only classification runs."""
    candidates = _make_candidates()
    classification = _default_classification(candidates)
    classifier = FakeClassifier(classification)
    extractor = FakeExtractor(_make_extract_result())

    runner = BaselineRunner(
        classifier=classifier,
        extractor=extractor,
        config=RunnerConfig(),
        split="holdout",
        candidates=candidates,
        classify_prompt=_prompt_with_rules(),
        skip_extraction=True,
    )
    runner._checkpoint_write = AsyncMock()
    runner._semaphore = asyncio.Semaphore(1)

    example = _FakeRecord(id=1, text="some text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    extractor.extract.assert_not_called()
    classifier.classify.assert_called_once()

    row = results[0]
    assert row is not None
    assert row.extract_status == "skipped"
    assert row.theses_raw == []
    assert row.theses_norm == []
    assert row.extract_latency_ms == 0.0
    assert row.raw_extract is None
    assert row.classify_status == "ok"


@pytest.mark.asyncio
async def test_stats_no_double_count() -> None:
    """ok and classify_ok are separate; a fully-successful example counts each once."""
    candidates = _make_candidates()
    classification = _default_classification(candidates)
    classifier = FakeClassifier(classification)
    extractor = FakeExtractor(_make_extract_result(status="ok"))

    runner = BaselineRunner(
        classifier=classifier,
        extractor=extractor,
        config=RunnerConfig(),
        split="dev",
        candidates=candidates,
    )
    runner._checkpoint_write = AsyncMock()
    runner._semaphore = asyncio.Semaphore(1)

    example = _FakeRecord(id=1, text="some text", label=1)
    results: list[ResultRow | None] = [None]
    await runner._process_example(MagicMock(), example, 0, results)

    assert runner._stats["ok"] == 1
    assert runner._stats["classify_ok"] == 1


# --------------------------------------------------------------------------- #
#  Rule extraction tests
# --------------------------------------------------------------------------- #
def test_default_prompt_rules_extracted() -> None:
    """The default classification prompt has no rules (empty list)."""
    layers = CLASSIFICATION_PROMPT_V0.layers
    expected = tuple(layers.rules) if layers else ()
    assert expected == ()


def test_non_default_prompt_with_valid_layers_rules_extracted() -> None:
    """A non-default prompt with valid layers has its rules extracted."""
    custom_layers = PromptLayer(
        role="role",
        task="task",
        rules=["custom rule 1", "custom rule 2", "custom rule 3"],
        output_contract="contract",
        fallback="fallback",
    )
    custom_prompt = PromptArtifact(
        version="classify-v1",
        layers=custom_layers,
        text="rendered text",
    )
    runner = BaselineRunner(
        classifier=FakeClassifier(_default_classification(_make_candidates())),
        extractor=FakeExtractor(_make_extract_result()),
        config=RunnerConfig(),
        split="dev",
        candidates=_make_candidates(),
        classify_prompt=custom_prompt,
    )
    assert runner._rules == ClassificationRules(
        rules=("custom rule 1", "custom rule 2", "custom rule 3"),
    )


@pytest.mark.asyncio
async def test_non_default_prompt_with_none_layers_uses_empty_rules() -> None:
    """A non-default prompt with None layers yields empty rules; the runner
    sets predicted_decision=0 and classify_status=ok without calling the
    classifier."""
    malformed_prompt = PromptArtifact(
        version="classify-v1",
        layers=None,
        text="some text",
    )
    classifier = FakeClassifier(_default_classification(_make_candidates()))
    extractor = FakeExtractor(_make_extract_result())
    runner = BaselineRunner(
        classifier=classifier,
        extractor=extractor,
        config=RunnerConfig(),
        split="dev",
        candidates=_make_candidates(),
        classify_prompt=malformed_prompt,
    )
    runner._checkpoint_write = AsyncMock()
    runner._semaphore = asyncio.Semaphore(1)
    assert runner._rules.rules == ()

    example = _FakeRecord(id=1, text="text", label=0)
    results: list[ResultRow | None] = [None]
    await runner._process_example(MagicMock(), example, 0, results)

    classifier.classify.assert_not_called()
    row = results[0]
    assert row is not None
    assert row.predicted_decision == 0
    assert row.selected_candidate == "0"
    assert row.classify_status == "ok"


@pytest.mark.asyncio
async def test_rules_passed_to_classifier_on_classify() -> None:
    """The runner passes extracted rules to classifier.classify()."""
    candidates = _make_candidates()
    classification = _default_classification(candidates)
    classifier = FakeClassifier(classification)
    runner = _make_runner(classifier=classifier, candidates=candidates)
    runner._semaphore = asyncio.Semaphore(1)
    example = _FakeRecord(id=1, text="text", label=1)
    results: list[ResultRow | None] = [None]

    await runner._process_example(MagicMock(), example, 0, results)

    call_args = classifier.classify.call_args
    assert call_args.args[1] == runner._rules
