"""Tokenizer seam for candidate scoring.

The ``TokenizerAdapter`` protocol decouples the scoring algorithm from a
concrete tokenizer library.  ``HuggingFaceTokenizerAdapter`` wraps
``transformers.AutoTokenizer`` for production use.

The ``BatchTokenizerAdapter`` protocol extends tokenization to padded batches
with per-item candidate-token boundaries, enabling batched model inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from transformers import AutoTokenizer


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Infrastructure seam: convert text into model tokens and back."""

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token IDs."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode ``token_ids`` back into text."""
        ...


class HuggingFaceTokenizerAdapter:
    """Concrete adapter wrapping ``transformers.AutoTokenizer``."""

    def __init__(self, model_name_or_path: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True))


@dataclass(frozen=True)
class BatchTokenization:
    """Result of batch tokenization with right padding and per-item boundaries.

    ``input_ids[i]`` is the padded token sequence for batch item ``i``.
    ``attention_mask[i]`` is ``1`` for real tokens and ``0`` for padding.
    ``prefix_token_counts[i]`` is the number of prefix tokens for item ``i``.
    ``candidate_token_ids[i]`` are the candidate continuation token IDs for
    item ``i`` — exactly the token IDs at candidate continuation positions in
    ``input_ids[i]`` (before padding).
    """

    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    prefix_token_counts: list[int]
    candidate_token_ids: list[list[int]]


@runtime_checkable
class BatchTokenizerAdapter(Protocol):
    """Infrastructure seam: encode multiple prompts as a padded batch.

    For each ``(prefix, candidate)`` pair, the adapter produces the complete
    model input token sequence, the exact positions of candidate continuation
    tokens, and the candidate token IDs at those positions.  The correctness
    invariant: candidate token IDs used for scoring MUST be exactly the token
    IDs occupying the candidate continuation positions in the model input
    sequence.  The implementation MUST NOT assume that
    ``encode(prefix) + encode(candidate)`` equals
    ``encode(prefix + candidate)``.
    """

    def encode_batch(
        self,
        prefixes: list[str],
        candidates: list[str],
    ) -> BatchTokenization:
        """Encode ``(prefix, candidate)`` pairs into a padded batch."""
        ...


class HuggingFaceBatchTokenizerAdapter:
    """Concrete batch adapter wrapping ``transformers.AutoTokenizer``.

    Uses right padding (padding tokens appended after real content).  The
    boundary between prefix and candidate tokens is determined using offset
    mapping from the tokenizer, not by assuming separate encoding is
    equivalent to encoding the concatenated prompt.
    """

    def __init__(self, model_name_or_path: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._pad_token_id = self._tokenizer.pad_token_id
        if self._pad_token_id is None:
            self._pad_token_id = 0

    def encode_batch(
        self,
        prefixes: list[str],
        candidates: list[str],
    ) -> BatchTokenization:
        if len(prefixes) != len(candidates):
            raise ValueError(
                f"prefixes ({len(prefixes)}) and candidates "
                f"({len(candidates)}) must have equal length"
            )

        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_prefix_counts: list[int] = []
        all_candidate_ids: list[list[int]] = []

        for prefix, candidate in zip(prefixes, candidates, strict=True):
            full_text = prefix + candidate
            encoding = self._tokenizer.encode(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids: list[int] = list(encoding["input_ids"])
            offsets = encoding["offset_mapping"]

            prefix_char_len = len(prefix)
            prefix_count = 0
            for idx, (_start, end) in enumerate(offsets):
                if end <= prefix_char_len:
                    prefix_count = idx + 1
                else:
                    break

            candidate_ids = token_ids[prefix_count:]

            all_input_ids.append(token_ids)
            all_attention_masks.append([1] * len(token_ids))
            all_prefix_counts.append(prefix_count)
            all_candidate_ids.append(candidate_ids)

        max_len = max(len(ids) for ids in all_input_ids) if all_input_ids else 0
        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            all_input_ids[i] = all_input_ids[i] + [self._pad_token_id] * pad_len
            all_attention_masks[i] = all_attention_masks[i] + [0] * pad_len

        return BatchTokenization(
            input_ids=all_input_ids,
            attention_mask=all_attention_masks,
            prefix_token_counts=all_prefix_counts,
            candidate_token_ids=all_candidate_ids,
        )
