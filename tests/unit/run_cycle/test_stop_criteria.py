"""Unit tests for the orchestrator's content-based stop-criteria integration."""
from __future__ import annotations

from typing import Any

from dynamic_prompt_core.application.ports.outbound.stop_criteria import (
    StopDecision,
    StopEvaluationContext,
)
from dynamic_prompt_core.application.services.classification_policy import (
    ArgmaxClassificationPolicy,
)
from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.deps import (
    EvaluateStopCriteriaDeps,
)
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import (
    _extract_metric_summary,
    _extract_rule_set,
    check_stop,
)
from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState
from dynamic_prompt_core.domain.prompts.base import PromptArtifact, PromptLayer


class _StubRunRepository:
    def save_results(self, results: list[dict[str, Any]], run_id: str, path: str) -> str:
        return path

    def load_results(self, path: str) -> list[dict[str, Any]]:
        return []


class _AlwaysStop:
    """StopCriteria implementation that always returns stop."""

    async def evaluate(self, context: StopEvaluationContext) -> StopDecision:
        return StopDecision(
            should_stop=True,
            reason="plateau_detected",
            triggered_criteria=["plateau_detected"],
            round_number=context.round_counter,
        )


class _AlwaysContinue:
    """StopCriteria implementation that always returns continue."""

    async def evaluate(self, context: StopEvaluationContext) -> StopDecision:
        return StopDecision(
            should_stop=False,
            reason=None,
            triggered_criteria=[],
            round_number=context.round_counter,
        )


def test_extract_metric_summary():
    metrics = {
        "accuracy": 0.82,
        "f1": {"macro": 0.74, "minority": 0.61, "weighted": 0.70, "0": 0.8, "1": 0.7},
    }
    summary = _extract_metric_summary(metrics)
    assert summary == {"accuracy": 0.82, "macro_f1": 0.74, "minority_f1": 0.61}


def test_extract_metric_summary_empty():
    summary = _extract_metric_summary({})
    assert summary == {"accuracy": 0.0, "macro_f1": 0.0, "minority_f1": 0.0}


def test_extract_rule_set_with_layers():
    layers = PromptLayer(
        role="r", task="t", rules=["rule-a", "rule-b"], output_contract="o", fallback="f"
    )
    artifact = PromptArtifact(version="v1", layers=layers, text="text")
    state = CycleState(active_version=artifact)
    rule_set = _extract_rule_set(state)
    assert rule_set == frozenset({"rule-a", "rule-b"})


def test_extract_rule_set_no_layers():
    artifact = PromptArtifact(version="v0", layers=None, text="text")
    state = CycleState(active_version=artifact)
    rule_set = _extract_rule_set(state)
    assert rule_set == frozenset()


def test_check_stop_max_rounds():
    state = CycleState(round_counter=5)
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig

    config = CycleConfig(max_rounds=5)
    should_stop, reason = check_stop(state, config)
    assert should_stop is True
    assert reason == "max_rounds_reached"


def test_check_stop_continue():
    state = CycleState(round_counter=2, candidate_queue=[{"id": 1}])
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig

    config = CycleConfig(max_rounds=5, max_consecutive_rollbacks=3)
    should_stop, reason = check_stop(state, config)
    assert should_stop is False
    assert reason is None


def test_stop_criteria_deps_optional():
    """RunCycleDeps constructs without stop_criteria_deps (backward compatible)."""
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
        RunCycleDeps,
    )
    from tests.unit.run_cycle.conftest import (
        FAKE_CANDIDATES,
        FakeCandidateScorer,
        MockDatasetRepository,
        MockEmbeddingClient,
        MockLLMClient,
        MockNormalizer,
        MockPromptRepository,
        MockRunRepository,
    )

    deps = RunCycleDeps(
        llm_client=MockLLMClient(),
        prompt_repository=MockPromptRepository(),
        run_repository=MockRunRepository(),
        dataset_repository=MockDatasetRepository(),
        embedding_client=MockEmbeddingClient(),
        normalizer=MockNormalizer(),
        candidate_scorer=FakeCandidateScorer(),
        classification_policy=ArgmaxClassificationPolicy(),
        candidates=FAKE_CANDIDATES,
    )
    assert deps.stop_criteria_deps is None


def test_stop_criteria_deps_provided():
    """RunCycleDeps accepts stop_criteria_deps when provided."""
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
        RunCycleDeps,
    )
    from tests.unit.run_cycle.conftest import (
        FAKE_CANDIDATES,
        FakeCandidateScorer,
        MockDatasetRepository,
        MockEmbeddingClient,
        MockLLMClient,
        MockNormalizer,
        MockPromptRepository,
        MockRunRepository,
    )

    stop_deps = EvaluateStopCriteriaDeps(run_repository=_StubRunRepository())
    deps = RunCycleDeps(
        llm_client=MockLLMClient(),
        prompt_repository=MockPromptRepository(),
        run_repository=MockRunRepository(),
        dataset_repository=MockDatasetRepository(),
        embedding_client=MockEmbeddingClient(),
        normalizer=MockNormalizer(),
        candidate_scorer=FakeCandidateScorer(),
        classification_policy=ArgmaxClassificationPolicy(),
        candidates=FAKE_CANDIDATES,
        stop_criteria_deps=stop_deps,
    )
    assert deps.stop_criteria_deps is not None


def test_valid_stop_reasons_extended():
    from dynamic_prompt_core.application.use_cases.run_cycle.report import (
        VALID_STOP_REASONS,
    )

    assert "plateau_detected" in VALID_STOP_REASONS
    assert "metric_degradation" in VALID_STOP_REASONS
    assert "no_candidates_available" in VALID_STOP_REASONS
    assert "budget_exhausted" in VALID_STOP_REASONS
    assert "rule_stagnation" in VALID_STOP_REASONS
    assert "rollback_streak" in VALID_STOP_REASONS
    assert "max_rounds_reached" in VALID_STOP_REASONS
