"""Model seam for candidate scoring.

The ``CausalLanguageModel`` protocol decouples the scoring algorithm from a
concrete model library.  ``TorchCausalLMAdapter`` wraps a PyTorch causal LM
and runs inference under ``torch.no_grad()`` / ``model.eval()``.

The ``BatchedCausalLanguageModel`` protocol extends the seam to batched
inference: one ``forward_batch`` call evaluates multiple padded sequences
and returns ``BatchedLogits`` with semantic shape ``[B, S, V]``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SequenceLogits:
    """Per-position logits for a token sequence.

    ``logits[i]`` is the logit vector at position ``i``, which for a causal LM
    predicts the token at position ``i + 1``.
    """

    logits: tuple[Sequence[float], ...]


@dataclass(frozen=True)
class BatchedLogits:
    """Per-item, per-position logits for a batch of token sequences.

    Has the semantic shape ``[batch_size, sequence_length, vocabulary_size]``.
    ``items[i]`` is the ``SequenceLogits`` for batch item ``i``, preserving the
    input batch order without reordering.
    """

    items: tuple[SequenceLogits, ...]


@runtime_checkable
class CausalLanguageModel(Protocol):
    """Infrastructure seam: execute model inference and expose logits."""

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        """Run the model on ``input_ids`` and return per-position logits."""
        ...


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


class TorchCausalLMAdapter:
    """Concrete adapter for a PyTorch causal language model.

    The model is expected to be constructed externally and passed in.  The
    adapter does not own the model lifecycle.  Inference runs under
    ``torch.no_grad()`` with the model in ``eval()`` mode.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        import torch

        self._model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            outputs = self._model(input_tensor)
            raw_logits = outputs.logits[0]
            per_position: list[Sequence[float]] = [
                raw_logits[i].tolist() for i in range(raw_logits.shape[0])
            ]
        return SequenceLogits(logits=tuple(per_position))


class TorchBatchedCausalLMAdapter:
    """Concrete batched adapter for a PyTorch causal language model.

    Runs the model in ``eval()`` mode under ``torch.no_grad()`` with batched
    input tensors and attention masks, returning per-item ``SequenceLogits``.
    The model lifecycle is owned externally.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        import torch

        self._model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            attention_tensor = torch.tensor(attention_mask, dtype=torch.long)
            outputs = self._model(
                input_tensor,
                attention_mask=attention_tensor,
            )
            batch_logits: list[SequenceLogits] = []
            for b in range(input_tensor.shape[0]):
                raw_logits = outputs.logits[b]
                per_position: list[Sequence[float]] = [
                    raw_logits[i].tolist() for i in range(raw_logits.shape[0])
                ]
                batch_logits.append(SequenceLogits(logits=tuple(per_position)))
        return BatchedLogits(items=tuple(batch_logits))


class SequentialBatchCompatibilityAdapter:
    """Temporary compatibility adapter: delegates ``forward_batch`` to ``forward``.

    For model adapters that do not support native batching, this adapter
    implements ``BatchedCausalLanguageModel`` by calling ``forward`` once per
    batch item (after stripping padding via the attention mask).  This is a
    transitional mechanism only — it preserves the semantic contract but does
    not provide the performance benefit of native tensor batching.
    """

    def __init__(self, model: CausalLanguageModel) -> None:
        self._model = model

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        items: list[SequenceLogits] = []
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            unpadded = [
                token_id
                for token_id, m in zip(ids, mask, strict=True)
                if m == 1
            ]
            items.append(self._model.forward(unpadded))
        return BatchedLogits(items=tuple(items))
