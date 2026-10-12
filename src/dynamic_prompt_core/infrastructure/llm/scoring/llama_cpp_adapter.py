"""GGUF-backed adapters for candidate scoring.

``LlamaCppBatchedCausalLMAdapter`` implements the ``BatchedCausalLanguageModel``
seam and ``LlamaCppBatchTokenizerAdapter`` implements the ``BatchTokenizerAdapter``
seam using ``llama-cpp-python``.  Both are wired into the shared
``LLMLogitCandidateScorer`` so the scoring algorithm, logit scorer, and prompt
builder are reused unchanged.

``llama_cpp`` is imported lazily inside ``LlamaCppBatchedCausalLMAdapter.__init__``
so the core package imports and installs without the optional GGUF runtime.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from collections.abc import Sequence
from typing import Any

from dynamic_prompt_core.infrastructure.llm.scoring.errors import GGUFProviderError
from dynamic_prompt_core.infrastructure.llm.scoring.instrumentation import (
    ScoringProfile,
    timed_stage,
)
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedLogits,
    SequenceLogits,
)
from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
    BatchTokenization,
)

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "the GGUF backend requires the optional 'llama-cpp-python' package; "
    "install it with `pip install 'dynamic-prompt-core[gguf]'`"
)


def _resolve_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Get an attribute that may be a value, a property, or a method.

    ``llama-cpp-python`` exposes ``n_vocab`` and ``n_ctx`` as methods in some
    versions and as properties/attributes in others.  This resolves both.
    """
    value = getattr(obj, name, default)
    if callable(value):
        return value()
    return value


def _import_llama() -> Any:
    """Lazy-import ``llama_cpp`` and return the ``Llama`` class.

    Raises ``GGUFProviderError`` with an actionable install hint if the
    optional dependency is missing.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - exercised via mock
        raise GGUFProviderError(_INSTALL_HINT) from exc
    return Llama


def _compute_file_checksum(path: str) -> str | None:
    """Return the SHA-256 checksum of ``path``, or ``None`` if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def scored_positions(prefix_token_count: int, candidate_token_count: int) -> range:
    """Return the logit positions needed to score a candidate.

    For a causal LM where logits at position ``i`` predict token ``i+1``,
    candidate tokens occupy positions ``m .. m+n-1`` (``m`` = prefix token
    count, ``n`` = candidate token count).  The logits that predict them are
    at positions ``m-1 .. m+n-2`` inclusive.  This is the causal-alignment
    convention used by ``LogitScorer`` (``position = prefix_token_count + k - 1``).
    """
    if candidate_token_count <= 0:
        return range(0)
    start = prefix_token_count - 1
    stop = prefix_token_count + candidate_token_count - 1
    return range(start, stop)


def _probe_selective_logits(model: Any) -> bool:
    """Probe whether the backend supports restricting saved logit positions.

    The probe is based on verified API behavior, not mere attribute presence.
    ``llama-cpp-python`` 0.3.x exposes no per-position logit-restriction API:
    ``logits_all=True`` saves all positions and ``logits_all=False`` saves only
    the last position, neither of which yields an arbitrary scored subset
    ``m-1 .. m+n-2``.  A future version exposing a verified selective-retrieval
    mechanism (e.g. an explicit position-mask API with tested semantics) would
    return ``True`` here.  Until then this returns ``False`` so the adapter
    falls back to the full ``eval_logits`` read and indexes into it.
    """
    return False


def shared_prefix_length(
    input_ids: list[list[int]],
    prefix_token_counts: list[int],
) -> int:
    """Return the length of the common token-level prefix across batch items.

    Compares the actual tokenized sequences, not text.  The shared prefix is
    capped at the minimum ``prefix_token_count`` so it never extends into
    candidate-specific tokens.  Returns 0 for an empty batch or when items
    share no common prefix.
    """
    if not input_ids:
        return 0
    cap = min(prefix_token_counts) if prefix_token_counts else 0
    if cap == 0:
        return 0
    shared = 0
    for pos in range(cap):
        token = input_ids[0][pos]
        for item in input_ids[1:]:
            if pos >= len(item) or item[pos] != token:
                return shared
        shared += 1
    return shared


def _probe_kv_cache_reuse(model: Any) -> bool:
    """Probe whether the backend supports a verified KV-cache save/restore.

    The probe is based on validated API behavior, not mere attribute presence.
    ``llama-cpp-python`` 0.3.x exposes no verified KV-cache save/restore or
    safe context-branching mechanism that guarantees per-candidate state
    isolation under the adapter's single mutable context.  A future version
    exposing a verified mechanism (e.g. ``kv_cache_seq_cp`` with tested
    semantics) would return ``True`` here.  Until then this returns ``False``
    so the adapter falls back to per-candidate reset-and-evaluate.
    """
    return False


class LlamaCppBatchedCausalLMAdapter:
    """Concrete ``BatchedCausalLanguageModel`` backed by ``llama-cpp-python``.

    The GGUF model is loaded once per instance from ``model_path``.  Because
    ``llama-cpp-python`` evaluates one token sequence per context (no native
    tensor batching), ``forward_batch`` evaluates each batch item sequentially:
    reset the KV cache, eval, read ``eval_logits`` once, and index into it.
    A ``threading.Lock`` serializes the eval+read critical section so the
    single mutable context is safe for concurrent thread calls.

    The loaded ``Llama`` instance is exposed via the ``llama`` property so the
    composition root can reuse it for the tokenizer adapter without reloading.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_batch: int = 512,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        seed: int = 42,
        verbose: bool = False,
        logits_all: bool = True,
        max_concurrent: int = 1,
        *,
        _llama: Any = None,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._seed = seed
        self._verbose = verbose
        self._logits_all = logits_all
        self._lock = threading.Lock()
        self._checksum: str | None = None
        self._last_profile: ScoringProfile | None = None
        self._max_concurrent = max_concurrent
        self._concurrency_semaphore = threading.BoundedSemaphore(max_concurrent)

        if _llama is not None:
            log.info("GGUF adapter: using injected model (skip load)")
            self._model = _llama
        else:
            llama_cls = _import_llama()
            if not os.path.isfile(model_path):
                raise GGUFProviderError(f"GGUF model file not found: {model_path}")
            log.info(
                "GGUF adapter: loading model %s (n_ctx=%d, n_batch=%d, "
                "n_threads=%s, n_gpu_layers=%d, seed=%d, logits_all=%s)",
                model_path, n_ctx, n_batch, n_threads, n_gpu_layers, seed, logits_all,
            )
            try:
                kwargs: dict[str, Any] = {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "n_batch": n_batch,
                    "n_gpu_layers": n_gpu_layers,
                    "seed": seed,
                    "verbose": verbose,
                    "logits_all": logits_all,
                }
                if n_threads is not None:
                    kwargs["n_threads"] = n_threads
                self._model = llama_cls(**kwargs)
            except GGUFProviderError:
                raise
            except Exception as exc:
                raise GGUFProviderError(
                    f"failed to load GGUF model {model_path!r}: {exc}"
                ) from exc
        self._n_vocab = int(_resolve_attr(self._model, "n_vocab", 0))
        self._selective_supported = _probe_selective_logits(self._model)
        self._kv_cache_reuse_supported = _probe_kv_cache_reuse(self._model)
        self._n_threads_configured = n_threads
        self._n_threads_effective = self._resolve_effective_threads()
        log.info(
            "GGUF adapter: model loaded — n_vocab=%d, selective_logits=%s, "
            "kv_cache_reuse=%s, n_threads(configured=%s, effective=%s)",
            self._n_vocab, self._selective_supported,
            self._kv_cache_reuse_supported,
            self._n_threads_configured, self._n_threads_effective,
        )

    @property
    def llama(self) -> Any:
        """The loaded ``Llama`` instance (infrastructure-internal reuse)."""
        return self._model

    @property
    def last_profile(self) -> ScoringProfile | None:
        """Most recent per-stage profile (available even when a call raised)."""
        return self._last_profile

    @property
    def selective_logits_supported(self) -> bool:
        """Whether the backend supports restricting saved logit positions."""
        return self._selective_supported

    @property
    def kv_cache_reuse_supported(self) -> bool:
        """Whether the backend supports a verified KV-cache save/restore."""
        return self._kv_cache_reuse_supported

    def _resolve_effective_threads(self) -> int | None:
        """Resolve the effective thread count, distinguishing source.

        If ``n_threads`` was explicitly configured, that value is used.  If it
        was ``None`` (backend default), attempt to read the backend's actual
        value; if unreadable, return ``None`` (meaning "backend default,
        not directly observable").  Actual CPU usage is NOT inferred solely
        from a configured value.
        """
        if self._n_threads is not None:
            return self._n_threads
        observed = _resolve_attr(self._model, "n_threads", None)
        if isinstance(observed, int):
            return observed
        return None

    @property
    def thread_info(self) -> dict[str, int | str | None]:
        """Thread-count resolution: configured, effective, and source."""
        if self._n_threads_configured is not None:
            source = "configured"
        elif self._n_threads_effective is not None:
            source = "backend_observed"
        else:
            source = "backend_default_unobservable"
        return {
            "n_threads_configured": self._n_threads_configured,
            "n_threads_effective": self._n_threads_effective,
            "n_threads_source": source,
        }

    def forward_batch(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> BatchedLogits:
        """Evaluate each batch item sequentially and return per-item logits.

        For each item: strip padding, acquire the lock, reset the context,
        eval, and read ``eval_logits`` once (materialized, not per-position).
        Validates vocabulary dimension and rejects NaN/inf.  Returns
        ``BatchedLogits`` in input order with plain-float elements matching
        the existing seam contract.

        Each stage is timed into a ``ScoringProfile`` (lock wait, reset, eval,
        logit read, logit convert, logit validate) so the profile separates
        ``model.eval(...)`` from lazy ``eval_logits`` materialization and
        Python post-processing.  The profile is attached to the result and
        also stored on ``self.last_profile`` so it remains available when a
        later stage raises.
        """
        items: list[SequenceLogits] = []
        profile = ScoringProfile()
        self._last_profile = profile
        log.debug("GGUF forward_batch: %d items", len(input_ids))
        for idx, (ids, mask) in enumerate(zip(input_ids, attention_mask, strict=True)):
            unpadded = [
                token_id
                for token_id, m in zip(ids, mask, strict=True)
                if m == 1
            ]
            log.debug(
                "GGUF forward_batch item %d: %d tokens (padded %d)",
                idx, len(unpadded), len(ids),
            )
            with timed_stage(profile, "lock_wait_ms"):
                self._concurrency_semaphore.acquire()
                self._lock.acquire()
            try:
                with timed_stage(profile, "reset_ms"):
                    try:
                        self._model.reset()
                    except Exception as exc:
                        raise GGUFProviderError(f"GGUF inference failed: {exc}") from exc
                with timed_stage(profile, "eval_ms"):
                    try:
                        self._model.eval(unpadded)
                    except Exception as exc:
                        raise GGUFProviderError(f"GGUF inference failed: {exc}") from exc
                with timed_stage(profile, "logits_read_ms"):
                    try:
                        raw_logits = list(self._model.eval_logits)
                    except Exception as exc:
                        raise GGUFProviderError(f"GGUF inference failed: {exc}") from exc
            finally:
                self._lock.release()
                self._concurrency_semaphore.release()
            log.debug(
                "GGUF forward_batch item %d: %d logit positions read",
                idx, len(raw_logits),
            )
            with timed_stage(profile, "logits_convert_ms"):
                per_position: list[Sequence[float]] = [
                    tuple(float(x) for x in position) for position in raw_logits
                ]
            with timed_stage(profile, "logits_validate_ms"):
                for position_values in per_position:
                    if self._n_vocab and len(position_values) != self._n_vocab:
                        raise GGUFProviderError(
                            f"logit vocabulary dimension {len(position_values)} "
                            f"does not match model vocabulary {self._n_vocab}"
                        )
                    if any(math.isnan(v) or math.isinf(v) for v in position_values):
                        raise GGUFProviderError(
                            "GGUF logits contain NaN or infinite values"
                        )
            profile.input_tokens += len(unpadded)
            profile.vocab_size = self._n_vocab
            profile.candidate_count += 1
            items.append(SequenceLogits(logits=tuple(per_position)))
        return BatchedLogits(items=tuple(items), profile=profile)

    def describe(self) -> dict[str, Any]:
        """Return backend metadata for experiment recording and resume identity."""
        if self._checksum is None:
            self._checksum = _compute_file_checksum(self._model_path)
        return {
            "backend": "gguf",
            "model_path": self._model_path,
            "model_checksum": self._checksum,
            "n_vocab": self._n_vocab,
            "n_ctx": self._n_ctx,
            "n_batch": self._n_batch,
            "n_threads": self._n_threads,
            "n_threads_effective": self._n_threads_effective,
            "n_threads_source": self.thread_info["n_threads_source"],
            "n_gpu_layers": self._n_gpu_layers,
            "seed": self._seed,
            "logits_all": self._logits_all,
            "selective_logits_supported": self._selective_supported,
            "kv_cache_reuse_supported": self._kv_cache_reuse_supported,
            "max_concurrent": self._max_concurrent,
        }


class LlamaCppBatchTokenizerAdapter:
    """Concrete ``BatchTokenizerAdapter`` backed by the GGUF tokenizer.

    Determines the prefix/candidate boundary with a full-encode prefix-prefix
    compatibility check: encode ``prefix + candidate`` and ``prefix``
    separately using the same BOS/special-token policy, verify the prefix
    token sequence is an exact prefix of the full sequence, and use the
    remainder as the candidate continuation.  On a cross-boundary merge
    (prefix not an exact prefix), reject with a ``GGUFProviderError`` rather
    than silently scoring a different token sequence.
    """

    def __init__(
        self,
        llama: Any,
        add_bos: bool = False,
        pad_token_id: int = 0,
    ) -> None:
        self._model = llama
        self._add_bos = add_bos
        self._pad_token_id = pad_token_id

    def _tokenize(self, text: str) -> list[int]:
        return list(
            self._model.tokenize(text.encode("utf-8"), add_bos=self._add_bos)
        )

    def encode_batch(
        self,
        prefixes: list[str],
        candidates: list[str],
    ) -> BatchTokenization:
        if len(prefixes) != len(candidates):
            raise GGUFProviderError(
                f"prefixes ({len(prefixes)}) and candidates "
                f"({len(candidates)}) must have equal length"
            )

        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_prefix_counts: list[int] = []
        all_candidate_ids: list[list[int]] = []

        log.debug("GGUF tokenizer encode_batch: %d items", len(prefixes))
        for prefix, candidate in zip(prefixes, candidates, strict=True):
            full_ids = self._tokenize(prefix + candidate)
            prefix_ids = self._tokenize(prefix)

            if full_ids[: len(prefix_ids)] == prefix_ids:
                prefix_count = len(prefix_ids)
                candidate_ids = full_ids[prefix_count:]
                log.debug(
                    "GGUF tokenizer: prefix=%d tokens, candidate=%d tokens "
                    "(full=%d)",
                    prefix_count, len(candidate_ids), len(full_ids),
                )
            else:
                log.warning(
                    "GGUF tokenizer: boundary mismatch for prompt length %d "
                    "(prefix %d tokens, full %d tokens)",
                    len(prefix + candidate), len(prefix_ids), len(full_ids),
                )
                raise GGUFProviderError(
                    f"tokenization boundary mismatch: the prefix token "
                    f"sequence is not an exact prefix of the full token "
                    f"sequence (prompt length {len(prefix + candidate)}); "
                    f"the GGUF tokenizer merges tokens across the "
                    f"prefix/candidate boundary"
                )

            all_input_ids.append(full_ids)
            all_attention_masks.append([1] * len(full_ids))
            all_prefix_counts.append(prefix_count)
            all_candidate_ids.append(candidate_ids)

        max_len = max((len(ids) for ids in all_input_ids), default=0)
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
