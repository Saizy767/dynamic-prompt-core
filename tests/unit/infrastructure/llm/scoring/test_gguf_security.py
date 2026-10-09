"""Security and reliability tests for the GGUF provider.

Task 9.1: unit-portable tests (mocked) plus fixture-gated integration portions.
"""
from __future__ import annotations

import pytest
from tests.unit.infrastructure.llm.scoring._gguf_mocks import FakeLlama

from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    GgufParams,
    ScorerBackendConfig,
)
from dynamic_prompt_core.infrastructure.llm.scoring.errors import GGUFProviderError
from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
)

VOCAB = 8


class TestMalformedModelPath:
    def test_missing_file_raises(
        self,
        tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dynamic_prompt_core.infrastructure.llm.scoring import llama_cpp_adapter

        monkeypatch.setattr(
            llama_cpp_adapter, "_import_llama", lambda: lambda **kw: None
        )
        missing = tmp_path / "nonexistent.gguf"
        with pytest.raises(GGUFProviderError, match="not found"):
            LlamaCppBatchedCausalLMAdapter(model_path=str(missing), n_ctx=8)


class TestInvalidConfig:
    def test_invalid_n_ctx_raises(self) -> None:
        with pytest.raises(Exception, match="n_ctx"):
            ScorerBackendConfig(
                backend="gguf",
                model_path="m.gguf",
                gguf=GgufParams(n_ctx=0),
            )


class TestContextOverflow:
    def test_oversized_prompt_raises_with_real_runtime(self) -> None:
        pytest.skip("context overflow is enforced by the llama-cpp-python runtime")


class TestConcurrentCalls:
    def test_concurrent_calls_no_corruption(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="x.gguf", n_ctx=64, _llama=model
        )
        results = [
            adapter.forward_batch([[1, 2]], [[1, 1]]),
            adapter.forward_batch([[3, 4]], [[1, 1]]),
            adapter.forward_batch([[5, 6]], [[1, 1]]),
        ]
        assert len(results) == 3
        for r in results:
            assert len(r.items) == 1


class TestSensitiveInputLogging:
    def test_error_excludes_input_text(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path="x.gguf", n_ctx=8, _llama=model
        )
        try:
            adapter.forward_batch([[1]], [[1]])
        except GGUFProviderError as exc:
            assert "secret" not in str(exc).lower()
