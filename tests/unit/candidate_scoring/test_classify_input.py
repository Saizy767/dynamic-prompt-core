"""Tests for the ClassifyInput use case."""
from __future__ import annotations

import pytest

from dynamic_prompt_core.application.services.classification_policy import (
    ArgmaxClassificationPolicy,
)
from dynamic_prompt_core.application.use_cases.classify_input import (
    ClassifyInputDeps,
    classify_input,
)
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.judgment import Judgment


class StubScorer:
    """Deterministic stub scorer for testing."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores = scores or {}
        self.calls: list[tuple[str, list[Candidate]]] = []

    async def score(
        self,
        text: str,
        candidates: list[Candidate],
    ) -> list[Judgment]:
        self.calls.append((text, candidates))
        return [
            Judgment(c, self._scores.get(c.value, 0.0)) for c in candidates
        ]


class StubPolicy:
    """Deterministic stub policy for testing."""

    def __init__(self) -> None:
        self.calls: list[list[Judgment]] = []

    def classify(self, judgments: list[Judgment]) -> object:
        self.calls.append(judgments)
        return ArgmaxClassificationPolicy().classify(judgments)


def _candidates(*names: str) -> list[Candidate]:
    return [Candidate(n) for n in names]


class TestClassifyInput:
    async def test_happy_path(self) -> None:
        scorer = StubScorer({"a": 0.2, "b": 0.8})
        policy = StubPolicy()
        deps = ClassifyInputDeps(scorer=scorer, policy=policy)

        result = await classify_input(deps, "some text", _candidates("a", "b"))

        assert result.selected == Candidate("b")
        assert len(scorer.calls) == 1
        assert scorer.calls[0] == ("some text", _candidates("a", "b"))
        assert len(policy.calls) == 1

    async def test_empty_candidates_rejected_before_scoring(self) -> None:
        scorer = StubScorer()
        policy = StubPolicy()
        deps = ClassifyInputDeps(scorer=scorer, policy=policy)

        with pytest.raises(ValueError):
            await classify_input(deps, "text", [])

        assert scorer.calls == []
        assert policy.calls == []

    async def test_duplicate_candidates_rejected_before_scoring(self) -> None:
        scorer = StubScorer()
        policy = StubPolicy()
        deps = ClassifyInputDeps(scorer=scorer, policy=policy)

        with pytest.raises(ValueError):
            await classify_input(deps, "text", _candidates("a", "a"))

        assert scorer.calls == []
        assert policy.calls == []

    async def test_scorer_receives_only_text_and_candidates(self) -> None:
        scorer = StubScorer({"a": 0.5})
        policy = StubPolicy()
        deps = ClassifyInputDeps(scorer=scorer, policy=policy)

        await classify_input(deps, "input text", _candidates("a"))

        assert scorer.calls[0][0] == "input text"
        assert scorer.calls[0][1] == _candidates("a")

    async def test_policy_receives_only_judgments(self) -> None:
        scorer = StubScorer({"a": 0.5})
        policy = StubPolicy()
        deps = ClassifyInputDeps(scorer=scorer, policy=policy)

        await classify_input(deps, "text", _candidates("a"))

        assert len(policy.calls[0]) == 1
        assert policy.calls[0][0].candidate == Candidate("a")

    async def test_scorer_mechanism_agnostic(self) -> None:
        policy = StubPolicy()

        scorer1 = StubScorer({"a": 0.2, "b": 0.8})
        scorer2 = StubScorer({"a": 0.9, "b": 0.1})

        deps1 = ClassifyInputDeps(scorer=scorer1, policy=policy)
        deps2 = ClassifyInputDeps(scorer=scorer2, policy=policy)

        result1 = await classify_input(deps1, "text", _candidates("a", "b"))
        result2 = await classify_input(deps2, "text", _candidates("a", "b"))

        assert result1.selected == Candidate("b")
        assert result2.selected == Candidate("a")
