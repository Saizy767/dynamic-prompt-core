"""Unit tests for the GGUF model adapter (``LlamaCppBatchedCausalLMAdapter``)."""
from __future__ import annotations

import typing

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring.errors import GGUFProviderError
from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
    LlamaCppBatchedCausalLMAdapter,
)
from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
    BatchedCausalLanguageModel,
    BatchedLogits,
)

from ._gguf_mocks import FakeLlama

VOCAB = 8


def _adapter(model: FakeLlama | None = None) -> LlamaCppBatchedCausalLMAdapter:
    return LlamaCppBatchedCausalLMAdapter(
        model_path="unused.gguf",
        n_ctx=64,
        _llama=model or FakeLlama(n_vocab=VOCAB),
    )


def _adapter_with(**kwargs: object) -> LlamaCppBatchedCausalLMAdapter:
    llama = kwargs.pop("llama", FakeLlama(n_vocab=VOCAB))
    return LlamaCppBatchedCausalLMAdapter(
        model_path="unused.gguf",
        n_ctx=64,
        _llama=llama,
        **typing.cast("dict[str, object]", kwargs),
    )


class TestProtocolSatisfaction:
    def test_satisfies_batched_protocol(self) -> None:
        adapter = _adapter()
        assert isinstance(adapter, BatchedCausalLanguageModel)

    def test_has_describe(self) -> None:
        adapter = _adapter()
        assert callable(getattr(adapter, "describe", None))


class TestForwardBatchContract:
    def test_returns_batched_logits_in_input_order(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch(
            input_ids=[[1, 2, 3], [4, 5]],
            attention_mask=[[1, 1, 1], [1, 1]],
        )
        assert isinstance(result, BatchedLogits)
        assert len(result.items) == 2
        assert len(result.items[0].logits) == 3
        assert len(result.items[1].logits) == 2

    def test_semantic_shape_b_s_v(self) -> None:
        adapter = _adapter()
        b, s = 2, 4
        result = adapter.forward_batch(
            input_ids=[[1] * s for _ in range(b)],
            attention_mask=[[1] * s for _ in range(b)],
        )
        assert len(result.items) == b
        for item in result.items:
            assert len(item.logits) == s
            for pos in item.logits:
                assert len(pos) == VOCAB

    def test_elements_are_plain_floats(self) -> None:
        adapter = _adapter()
        result = adapter.forward_batch([[1, 2]], [[1, 1]])
        for pos in result.items[0].logits:
            for v in pos:
                assert isinstance(v, float)

    def test_padding_tokens_stripped(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter_with(llama=model)
        adapter.forward_batch(
            input_ids=[[1, 2, 3, 0, 0]],
            attention_mask=[[1, 1, 1, 0, 0]],
        )
        assert model.eval_calls[0] == [1, 2, 3]

    def test_context_reset_per_item(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter_with(llama=model)
        adapter.forward_batch([[1], [2]], [[1], [1]])
        assert model.reset_count == 2


class TestNumericalValidation:
    def test_vocab_dimension_mismatch_raises(self) -> None:
        model = FakeLlama(n_vocab=VOCAB, logits_per_position=[0.0] * (VOCAB + 1))
        adapter = _adapter_with(llama=model)
        with pytest.raises(GGUFProviderError, match="vocabulary dimension"):
            adapter.forward_batch([[1, 2]], [[1, 1]])

    def test_nan_logits_rejected(self) -> None:
        bad = [float("nan")] + [0.0] * (VOCAB - 1)
        model = FakeLlama(n_vocab=VOCAB, logits_per_position=bad)
        adapter = _adapter_with(llama=model)
        with pytest.raises(GGUFProviderError, match="NaN"):
            adapter.forward_batch([[1]], [[1]])

    def test_inf_logits_rejected(self) -> None:
        bad = [float("inf")] + [0.0] * (VOCAB - 1)
        model = FakeLlama(n_vocab=VOCAB, logits_per_position=bad)
        adapter = _adapter_with(llama=model)
        with pytest.raises(GGUFProviderError, match="NaN"):
            adapter.forward_batch([[1]], [[1]])


class TestLogitBufferReadOnce:
    def test_eval_logits_read_once_per_item(self) -> None:
        model = FakeLlama(n_vocab=VOCAB)
        adapter = _adapter_with(llama=model)
        adapter.forward_batch([[1, 2, 3, 4]], [[1, 1, 1, 1]])
        assert model.eval_logits.iter_count == 1


class TestMissingDependency:
    def test_missing_llama_cpp_raises_with_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def blocking_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llama_cpp":
                raise ImportError("no module named 'llama_cpp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)
        with pytest.raises(GGUFProviderError) as exc_info:
            LlamaCppBatchedCausalLMAdapter(model_path="x.gguf", n_ctx=8)
        assert "llama-cpp-python" in str(exc_info.value)


class TestDescribe:
    def test_returns_required_fields(self, tmp_path: pytest.TempPathFactory) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake")
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path=str(model_file),
            n_ctx=32,
            _llama=FakeLlama(n_vocab=VOCAB),
        )
        meta = adapter.describe()
        assert meta["backend"] == "gguf"
        assert meta["model_path"] == str(model_file)
        assert meta["model_checksum"] is not None
        assert meta["n_vocab"] == VOCAB
        assert meta["n_ctx"] == 32

    def test_checksum_cached(self, tmp_path: pytest.TempPathFactory) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake")
        adapter = LlamaCppBatchedCausalLMAdapter(
            model_path=str(model_file),
            n_ctx=32,
            _llama=FakeLlama(n_vocab=VOCAB),
        )
        first = adapter.describe()
        second = adapter.describe()
        assert first["model_checksum"] == second["model_checksum"]
