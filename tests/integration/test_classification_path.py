"""Integration tests for the end-to-end candidate-scoring classification path."""
from __future__ import annotations

from dynamic_prompt_core.application.services.classification_policy import (
    ArgmaxClassificationPolicy,
)
from dynamic_prompt_core.application.use_cases.classify_input import (
    ClassifyInputDeps,
    classify_input,
)
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment


class _FakeScorer:
    """Deterministic! fake scorer returning fixed judgments."""

    def __init__(self, judgments: list[Judgment]) -> None:
        self._judgments = judgments

    async def score(
        self,
        text: str,
        candidates: list[Candidate],
    ) -> list[Judgment]:
        return list(self._judgments)


class TestEndToEndClassificationPath:
    """Verify the full path: text → scorer.score → judgments → policy.classify → Classification."""

    async def test_selected_candidate_derived_from_scores(self) -> None:
        """The selected candidate is derived from candidate scores, not generated text."""
        candidates = [Candidate("0"), Candidate("1")]
        scorer = _FakeScorer([
            Judgment(candidates[0], 0.2),
            Judgment(candidates[1], 0.8),
        ])
        deps = ClassifyInputDeps(scorer=scorer, policy=ArgmaxClassificationPolicy())

        result = await classify_input(deps, "example input", candidates)

        assert result.selected == Candidate("1")

    async def test_no_generated_json_response_involved(self) -> None:
        """The classification path does not involve a generated JSON response.

        The fake scorer returns Judgment objects directly — there is no text
        generation, JSON parsing, or structured output involved.
        """
        candidates = [Candidate("0"), Candidate("1")]
        scorer = _FakeScorer([
            Judgment(candidates[0], 0.2),
            Judgment(candidates[1], 0.8),
        ])
        deps = ClassifyInputDeps(scorer=scorer, policy=ArgmaxClassificationPolicy())

        result = await classify_input(deps, "example input", candidates)

        assert result.selected == Candidate("1")
        assert all(isinstance(j, Judgment) for j in result.judgments)
        assert all(isinstance(j.score, float) for j in result.judgments)

    async def test_candidate_ordering_preserved_through_scorer_policy(self) -> None:
        """Candidate ordering is preserved through the full scorer → policy path."""
        candidates = [Candidate("a"), Candidate("b"), Candidate("c")]
        scorer = _FakeScorer([
            Judgment(candidates[0], 0.5),
            Judgment(candidates[1], 0.5),
            Judgment(candidates[2], 0.2),
        ])
        deps = ClassifyInputDeps(scorer=scorer, policy=ArgmaxClassificationPolicy())

        result = await classify_input(deps, "text", candidates)

        for i, c in enumerate(candidates):
            assert result.judgments[i].candidate == c
        assert result.selected == Candidate("a")


class TestMetricBoundaryInvariant:
    """Verify that Judgment.score magnitude does not independently influence metrics.

    The metric is computed from predicted_decision vs true_label. Changing
    Judgment.score without changing the selected candidate must not change
    predicted_decision or the resulting metric.
    """

    async def test_score_change_without_selection_change_is_metric_invariant(self) -> None:
        candidates = [Candidate("0"), Candidate("1")]
        true_label = 1

        scorer_low = _FakeScorer([
            Judgment(candidates[0], 0.1),
            Judgment(candidates[1], 0.8),
        ])
        scorer_high = _FakeScorer([
            Judgment(candidates[0], 0.2),
            Judgment(candidates[1], 0.9),
        ])

        deps_low = ClassifyInputDeps(scorer=scorer_low, policy=ArgmaxClassificationPolicy())
        deps_high = ClassifyInputDeps(scorer=scorer_high, policy=ArgmaxClassificationPolicy())

        result_low = await classify_input(deps_low, "text", candidates)
        result_high = await classify_input(deps_high, "text", candidates)

        assert result_low.selected == Candidate("1")
        assert result_high.selected == Candidate("1")

        predicted_low = int(result_low.selected.value)
        predicted_high = int(result_high.selected.value)

        assert predicted_low == predicted_high == 1

        metric_low = 1 if predicted_low == true_label else 0
        metric_high = 1 if predicted_high == true_label else 0

        assert metric_low == metric_high

    async def test_different_scores_same_selection_same_predicted_decision(self) -> None:
        """Two scoring scenarios with different score magnitudes but the same
        selected candidate produce the same predicted_decision."""
        candidates = [Candidate("0"), Candidate("1")]

        scorer_a = _FakeScorer([
            Judgment(candidates[0], -5.0),
            Judgment(candidates[1], -0.3),
        ])
        scorer_b = _FakeScorer([
            Judgment(candidates[0], -2.0),
            Judgment(candidates[1], -0.1),
        ])

        deps_a = ClassifyInputDeps(scorer=scorer_a, policy=ArgmaxClassificationPolicy())
        deps_b = ClassifyInputDeps(scorer=scorer_b, policy=ArgmaxClassificationPolicy())

        result_a = await classify_input(deps_a, "text", candidates)
        result_b = await classify_input(deps_b, "text", candidates)

        predicted_a = int(result_a.selected.value)
        predicted_b = int(result_b.selected.value)

        assert predicted_a == predicted_b == 1
