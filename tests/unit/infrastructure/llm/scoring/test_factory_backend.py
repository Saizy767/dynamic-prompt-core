"""Unit tests for the scorer factory backend@backend selection."""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from dynamic_prompt_core.infrastructure.llm.scoring import factory as factory_module
from dynamic_prompt_core.infrastructure.llm.scoring import llama_cpp_adapter
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    GgufParams,
    ScorerBackendConfig,
)
from dynamic_prompt_core.infrastructure.llm.scoring.errors import GGUFProviderError


class _FakeLlama:
    n_vocab = 8

    def reset(self) -> None:
        pass

    def eval(self, tokens: list[int]) -> None:
        pass

    eval_logits: Sequence[list[float]] = []

    def tokenize(self, text: bytes, add_bos: bool = False) -> list[int]:
        return list(text)


class _FakeLlamaCls:
    def __call__(self, **kwargs: object) -> _FakeLlama:
        return _FakeLlama()


class TestFactoryHuggingFace:
    def test_hf_backend_returns_scorer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeHFModel:
            def eval(self) -> None:
                pass

        class FakeTokenizer:
            pad_token_id = 0

            def encode(self, text: str, **kwargs: object) -> dict[str, list[int]]:
                return {"input_ids": list(text.encode()), "offset_mapping": []}

            def decode(self, ids: list[int], **kwargs: object) -> str:
                return ""

        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda path: FakeHFModel(),
        )
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda path: FakeTokenizer(),
        )
        cfg = ScorerBackendConfig(backend="huggingface", model_path="hf-model")
        scorer = factory_module.build_candidate_scorer(cfg)
        assert callable(getattr(scorer, "score", None))


class TestFactoryGguf:
    def test_gguf_backend_returns_scorer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        model_file = tmp_path / "m.gguf"
        model_file.write_bytes(b"fake")
        monkeypatch.setattr(
            llama_cpp_adapter, "_import_llama", lambda: _FakeLlamaCls()
        )
        cfg = ScorerBackendConfig(
            backend="gguf",
            model_path=str(model_file),
            gguf=GgufParams(n_ctx=64),
        )
        scorer = factory_module.build_candidate_scorer(cfg)
        assert callable(getattr(scorer, "score", None))


class TestFactoryMissingDependency:
    def test_missing_llama_cpp_raises_with_hint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        model_file = tmp_path / "m.gguf"
        model_file.write_bytes(b"fake")

        def raising_import() -> object:
            raise GGUFProviderError(
                "the GGUF backend requires the optional 'llama-cpp-python' package"
            )

        monkeypatch.setattr(llama_cpp_adapter, "_import_llama", raising_import)
        cfg = ScorerBackendConfig(
            backend="gguf",
            model_path=str(model_file),
            gguf=GgufParams(n_ctx=64),
        )
        with pytest.raises(GGUFProviderError, match="llama-cpp-python"):
            factory_module.build_candidate_scorer(cfg)
