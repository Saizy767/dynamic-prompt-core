"""Tests for runtime configuration: thread resolution and GPU-offload opt-in."""
from __future__ import annotations

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
)

from ._gguf_mocks import FakeLlama

VOCAB = 8


class TestThreadResolution:
    def test_explicit_thread_count(self) -> None:
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="unused.gguf", n_ctx=64, n_threads=4,
            _llama=FakeLlama(n_vocab=VOCAB),
        )
        info = adapter.thread_info
        assert info["n_threads_configured"] == 4
        assert info["n_threads_effective"] == 4
        assert info["n_threads_source"] == "configured"

    def test_none_falls_back_to_backend_default(self) -> None:
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="unused.gguf", n_ctx=64, n_threads=None,
            _llama=FakeLlama(n_vocab=VOCAB),
        )
        info = adapter.thread_info
        assert info["n_threads_configured"] is None
        assert info["n_threads_source"] == "backend_default_unobservable"

    def test_describe_includes_thread_info(self) -> None:
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="unused.gguf", n_ctx=64, n_threads=2,
            _llama=FakeLlama(n_vocab=VOCAB),
        )
        meta = adapter.describe()
        assert meta["n_threads"] == 2
        assert meta["n_threads_effective"] == 2
        assert meta["n_threads_source"] == "configured"

    def test_describe_includes_all_runtime_params(self) -> None:
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="unused.gguf", n_ctx=128, n_batch=256, n_threads=4,
            n_gpu_layers=0, _llama=FakeLlama(n_vocab=VOCAB),
        )
        meta = adapter.describe()
        assert meta["n_ctx"] == 128
        assert meta["n_batch"] == 256
        assert meta["n_threads"] == 4
        assert meta["n_gpu_layers"] == 0
        assert "logits_all" in meta


class TestGPUOffload:
    @pytest.mark.gguf
    def test_gpu_offload_opt_in(self) -> None:
        """Opt-in GPU-offload test; skipped without a compatible backend.

        This test is marked ``gguf`` so it only runs when the optional
        ``llama-cpp-python`` runtime and a GGUF fixture are available.  It is
        skipped by default in CI (no GPU backend).
        """
        pytest.skip("GPU-offload test requires a compatible backend and hardware")
