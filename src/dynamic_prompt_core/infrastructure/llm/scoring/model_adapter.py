"""Model seam for candidate scoring.

The ``CausalLanguageModel`` protocol decouples the scoring algorithm from a
concrete model library.  ``TorchCausalLMAdapter`` wraps a PyTorch causal LM
and runs inference under ``torch.no_grad()`` / ``model.eval()``.
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


@runtime_checkable
class CausalLanguageModel(Protocol):
    """Infrastructure seam: execute model inference and expose logits."""

    def forward(self, input_ids: list[int]) -> SequenceLogits:
        """Run the model on ``input_ids`` and return per-position logits."""
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
