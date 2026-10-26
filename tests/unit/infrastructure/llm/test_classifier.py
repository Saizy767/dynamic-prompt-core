"""Tests for the LLMLogitClassifier with deterministic doubles."""
from __future__ import annotations

import math

import pytest

from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification_rules import ClassificationRules
from dynamic_prompt_core.domain.models.judgment import Judgment
from dynamic_prompt_core.infrastructure.llm.classifier import LLMLogitClassifier
from dynamic_prompt_core.infrastructure.llm.scoring.classifier_prompt_builder import (
    ClassifierPromptBuilder,
)
from dynamic_prompt_core.infrastructure.llm.scoring.errors import LLMScoringError
from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import LogitScorer
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedLogits,
    SequenceLogits,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenization,
)

_SYSTEM_PROMPT = "You are a classifier.\n\n"
_RULES = ClassificationRules(rules=("rule a", "rule b", "rule c"))
_CANDIDATES = [Candidate("0"), Candidate("1")]


class StubTokenizer:
    PAD_TOKEN_ID = 0

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tid) for tid in token_ids)

    def encode_batch(
        self, prefixes: list[str], candidates: list[str],
    ) -> BatchTokenization:
        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []
        all_pcounts: list[int] = []
        all_cids: list[list[int]] = []
        for p, c in zip(prefixes, candidates, strict=True):
            pids = [ord(ch) for ch in p]
            cids = [ord(ch) for ch in c]
            full = pids + cids
            all_ids.append(full)
            all_masks.append([1] * len(full))
            all_pcounts.append(len(pids))
            all_cids.append(cids)
        max_len = max(len(ids) for ids in all_ids) if all_ids else 0
        for i in range(len(all_ids)):
            pad = max_len - len(all_ids[i])
            all_ids[i] = all_ids[i] + [0] * pad
            all_masks[i] = all_masks[i] + [0] * pad
        return BatchTokenization(
            input_ids=all_ids, attention_mask=all_masks,
            prefix_token_counts=all_pcounts, candidate_token_ids=all_cids,
        )


VOCAB = 256


def _uniform_lp(vocab: int = VOCAB) -> list[float]:
    lp = math.log(1.0 / vocab)
    return [lp] * vocab


def _valid_lp(p0: float) -> list[float]:
    return [math.log(p0), math.log(1.0 - p0)]


class StubModel:
    def __init__(self) -> None:
        self._lp = _uniform_lp()

    def forward_batch(
        self, input_ids: list[list[int]], attention_mask: list[list[int]], **kw: object,
    ) -> BatchedLogits:
        items = [
            SequenceLogits(logits=tuple(tuple(self._lp) for _ in range(len(seq))))
            for seq in input_ids
        ]
        return BatchedLogits(items=tuple(items))


class StubModelWithProb:
    """Model double that favors specific candidate token IDs."""

    def __init__(self, prob_for_token: dict[int, float]) -> None:
        self._pt = prob_for_token

    def forward_batch(
        self, input_ids: list[list[int]], attention_mask: list[list[int]], **kw: object,
    ) -> BatchedLogits:
        prefix_counts = kw.get("prefix_token_counts", [])
        candidate_ids = kw.get("candidate_token_ids", [])
        items: list[SequenceLogits] = []
        for seq_idx, seq in enumerate(input_ids):
            n = len(seq)
            positions: list[list[float]] = []
            pcount = prefix_counts[seq_idx] if seq_idx < len(prefix_counts) else n
            cids = candidate_ids[seq_idx] if seq_idx < len(candidate_ids) else []
            for i in range(n):
                if i == pcount - 1 and cids:
                    tid = cids[0]
                    p = self._pt.get(tid, 0.5)
                else:
                    tid = seq[i] if i < n else -1
                    p = self._pt.get(tid, 0.5)
                lp = math.log(p) if p > 0 else -700.0
                rest = math.log((1.0 - p) / (VOCAB - 1)) if p < 1.0 else -700.0
                logits = [rest] * VOCAB
                if 0 <= tid < VOCAB:
                    logits[tid] = lp
                positions.append(logits)
            items.append(SequenceLogits(logits=tuple(tuple(p) for p in positions)))
        return BatchedLogits(items=tuple(items))


class FailingModel:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def forward_batch(
        self, input_ids: list[list[int]], attention_mask: list[list[int]], **kw: object,
    ) -> BatchedLogits:
        raise self._exc


def _make_classifier(
    model: object,
    candidates: list[Candidate] | None = None,
) -> LLMLogitClassifier:
    return LLMLogitClassifier(
        prompt_builder=ClassifierPromptBuilder(system_prompt=_SYSTEM_PROMPT),
        tokenizer=StubTokenizer(),  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        logit_scorer=LogitScorer(),
        candidates=candidates or _CANDIDATES,
    )


class TestClassifierReturnsClassification:
    async def test_returns_classification_with_selected(self) -> None:
        clf = _make_classifier(StubModel())
        result = await clf.classify("some text", _RULES)
        assert result.selected in _CANDIDATES
        assert len(result.judgments) == 2

    async def test_judgment_order_preserved(self) -> None:
        cands = [Candidate("0"), Candidate("1"), Candidate("2")]
        clf = _make_classifier(StubModel(), candidates=cands)
        result = await clf.classify("text", _RULES)
        assert [j.candidate for j in result.judgments] == cands

    async def test_highest_score_selected(self) -> None:
        prob = {ord("0"): 0.1, ord("1"): 0.9}
        clf = _make_classifier(StubModelWithProb(prob))
        result = await clf.classify("text", _RULES)
        assert result.selected == Candidate("1")

    async def test_tie_breaks_by_input_order(self) -> None:
        clf = _make_classifier(StubModel())
        result = await clf.classify("text", _RULES)
        assert result.selected == Candidate("0")


class TestClassifierSystemPrompt:
    async def test_system_prompt_used_as_prefix(self) -> None:
        builder = ClassifierPromptBuilder(system_prompt="MY_CUSTOM_PROMPT\n")
        clf = LLMLogitClassifier(
            prompt_builder=builder,
            tokenizer=StubTokenizer(),  # type: ignore[arg-type]
            model=StubModel(),  # type: ignore[arg-type]
            logit_scorer=LogitScorer(),
            candidates=_CANDIDATES,
        )
        await clf.classify("text", _RULES)
        prefix = builder.build_prefix("text", _RULES.rules)
        assert prefix.startswith("MY_CUSTOM_PROMPT\n")


class TestClassifierErrorPropagation:
    async def test_model_failure_raises_candidate_scoring_error(self) -> None:
        clf = _make_classifier(FailingModel(LLMScoringError("boom")))
        with pytest.raises(CandidateScoringError):
            await clf.classify("text", _RULES)

    async def test_generic_exception_raises_candidate_scoring_error(self) -> None:
        clf = _make_classifier(FailingModel(RuntimeError("boom")))
        with pytest.raises(CandidateScoringError):
            await clf.classify("text", _RULES)

    async def test_no_partial_results(self) -> None:
        clf = _make_classifier(FailingModel(LLMScoringError("boom")))
        with pytest.raises(CandidateScoringError, match="classification failed"):
            await clf.classify("text", _RULES)
