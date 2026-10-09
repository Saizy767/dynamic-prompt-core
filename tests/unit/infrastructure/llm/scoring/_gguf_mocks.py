"""Shared mock doubles for GGUF adapter unit tests.

These doubles stand in for ``llama_cpp.Llama`` so the test suite runs without
the optional ``llama-cpp-python`` runtime installed.
"""
from __future__ import annotations

from collections.abc import Iterator


class CountingLogits:
    """Wraps a list of per-position logit lists and counts iterations.

    ``list(model.eval_logits)`` calls ``__iter__`` once.  Asserting
    ``iter_count == 1`` per evaluation proves the adapter reads the buffer
    once rather than once per position.
    """

    def __init__(self, positions: list[list[float]]) -> None:
        self._positions = positions
        self.iter_count = 0

    def __iter__(self) -> Iterator[list[float]]:
        self.iter_count += 1
        return iter(self._positions)


class FakeLlama:
    """Deterministic double for ``llama_cpp.Llama``.

    Character-level tokenization (each byte → its int value) so the prefix is
    always an exact prefix of the full sequence.  ``eval_logits`` is a
    ``CountingLogits`` so tests can assert it is read once per evaluation.
    """

    def __init__(
        self,
        n_vocab: int = 8,
        logits_per_position: list[float] | None = None,
    ) -> None:
        self.n_vocab = n_vocab
        self.reset_count = 0
        self.eval_calls: list[list[int]] = []
        self._logits_per_position = logits_per_position
        self.eval_logits: CountingLogits = CountingLogits([])

    def reset(self) -> None:
        self.reset_count += 1
        self.eval_logits = CountingLogits([])

    def eval(self, tokens: list[int]) -> None:
        self.eval_calls.append(list(tokens))
        n = len(tokens)
        base = self._logits_per_position or [0.0] * self.n_vocab
        positions = [list(base) for _ in range(n)]
        self.eval_logits = CountingLogits(positions)

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        ids = list(text)
        if add_bos:
            ids = [self.n_vocab, *ids]
        return ids


class MergingFakeLlama:
    """Double whose tokenizer merges the bigram ``ab`` into one token.

    When ``ab`` appears consecutively it is emitted as a single merge token
    (999) instead of two separate tokens.  Used to exercise the boundary
    mismatch path: encoding ``"xa"`` and ``"by"`` separately yields
    ``[x, a]`` for the prefix, but encoding ``"xaby"`` yields
    ``[x, merge, y]`` so the prefix is NOT an exact prefix of the full
    sequence.
    """

    def __init__(self, n_vocab: int = 8) -> None:
        self.n_vocab = n_vocab

    def reset(self) -> None:
        pass

    def eval(self, tokens: list[int]) -> None:
        pass

    eval_logits: list[list[float]] = []

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        decoded = text.decode("utf-8")
        ids: list[int] = []
        i = 0
        while i < len(decoded):
            if decoded[i : i + 2] == "ab":
                ids.append(999)
                i += 2
            else:
                ids.append(ord(decoded[i]))
                i += 1
        return ids
