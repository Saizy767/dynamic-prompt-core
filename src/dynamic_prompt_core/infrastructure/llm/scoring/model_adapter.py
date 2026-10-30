"""Model seam for candidate scoring.

The ``BatchedCausalLanguageModel`` protocol decouples the scoring algorithm
from a concrete model library: one ``forward_batch`` call evaluates multiple
padded sequences and returns ``BatchedLogits`` with semantic shape
``[B, S, V]``.  ``SparseLogitVector`` avoids materializing a full
vocabulary-size logit list when only one entry's ``log_softmax`` is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dynamic_prompt_core.infrastructure.llm.scoring.instrumentation import (
    ScoringProfile,
)


@dataclass(frozen=True)
class SequenceLogits:
    """Per-position logits for a token sequence.

    ``logits[i]`` is the logit vector at position ``i``, which for a causal LM
    predicts the token at position ``i + 1``.
    """

    logits: tuple[Sequence[float], ...]


class SparseLogitVector(Sequence[float]):
    """Logit vector with one distinguished token and a shared fill value.

    Avoids materializing a full vocab-size list when only one entry's
    log_softmax is needed (e.g. logits reconstructed from API log-probs).
    ``actual_token < 0`` marks a uniform/floor vector where every entry
    equals ``other_logit``.
    """

    __slots__ = ("actual_token", "actual_logit", "other_logit", "vocab_size")

    def __init__(
        self,
        actual_token: int,
        actual_logit: float,
        other_logit: float,
        vocab_size: int,
    ) -> None:
        self.actual_token = actual_token
        self.actual_logit = actual_logit
        self.other_logit = other_logit
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            raise TypeError("SparseLogitVector does not support slice indexing")
        if index < 0 or index >= self.vocab_size:
            raise IndexError(index)
        if index == self.actual_token:
            return self.actual_logit
        return self.other_logit


@dataclass(frozen=True)
class BatchedLogits:
    """Per-item, per-position logits for a batch of token sequences.

    Has the semantic shape ``[batch_size, sequence_length, vocabulary_size]``.
    ``items[i]`` is the ``SequenceLogits`` for batch item ``i``, preserving the
    input batch order without reordering.  ``profile`` carries optional
    per-stage performance instrumentation produced by the model adapter; it is
    ``None`` for adapters that do not instrument their forward pass.
    """

    items: tuple[SequenceLogits, ...]
    profile: ScoringProfile | None = None


@runtime_checkable
class BatchedCausalLanguageModel(Protocol):
    """Infrastructure seam: execute batched model inference.

    Accepts padded ``input_ids`` and ``attention_mask`` and returns
    ``BatchedLogits`` where ``items[i]`` corresponds to input batch item ``i``
    without reordering.
    """

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        """Run the model on a padded batch and return per-item logits."""
        ...
