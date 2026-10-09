"""Unit tests for scorer backend configuration."""
from __future__ import annotations

import pytest

from dynamic_prompt_core.infrastructure.config.loader import ConfigError
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    GgufParams,
    ScorerBackendConfig,
    scorer_backend_config_from_toml,
)


class TestScorerBackendConfigValidation:
    def test_valid_huggingface_config(self) -> None:
        cfg = ScorerBackendConfig(backend="huggingface", model_path="model")
        assert cfg.backend == "huggingface"
        assert cfg.gguf is None

    def test_valid_gguf_config_defaults_params(self) -> None:
        cfg = ScorerBackendConfig(backend="gguf", model_path="m.gguf")
        assert cfg.gguf is not None
        assert cfg.gguf.n_ctx == 2048

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ConfigError, match="Invalid scorer backend"):
            ScorerBackendConfig(backend="vllm", model_path="m")

    def test_missing_model_path_raises(self) -> None:
        with pytest.raises(ConfigError, match="model_path"):
            ScorerBackendConfig(backend="huggingface", model_path="")

    def test_invalid_n_ctx_raises(self) -> None:
        with pytest.raises(ConfigError, match="n_ctx"):
            ScorerBackendConfig(
                backend="gguf",
                model_path="m.gguf",
                gguf=GgufParams(n_ctx=0),
            )

    def test_invalid_n_batch_raises(self) -> None:
        with pytest.raises(ConfigError, match="n_batch"):
            ScorerBackendConfig(
                backend="gguf",
                model_path="m.gguf",
                gguf=GgufParams(n_batch=-1),
            )

    def test_invalid_n_gpu_layers_raises(self) -> None:
        with pytest.raises(ConfigError, match="n_gpu_layers"):
            ScorerBackendConfig(
                backend="gguf",
                model_path="m.gguf",
                gguf=GgufParams(n_gpu_layers=-5),
            )


class TestScorerBackendConfigFromToml:
    def test_absent_model_section_defaults_to_hf(self) -> None:
        cfg = scorer_backend_config_from_toml(
            {"llm": {"model_path": "hf"}},
            default_model_path="hf",
        )
        assert cfg.backend == "huggingface"
        assert cfg.model_path == "hf"

    def test_huggingface_section(self) -> None:
        cfg = scorer_backend_config_from_toml(
            {"model": {"backend": "huggingface", "model_path": "hf"}}
        )
        assert cfg.backend == "huggingface"
        assert cfg.model_path == "hf"

    def test_gguf_section_with_params(self) -> None:
        cfg = scorer_backend_config_from_toml(
            {
                "model": {
                    "backend": "gguf",
                    "model_path": "m.gguf",
                    "gguf": {
                        "n_ctx": 4096,
                        "n_batch": 256,
                        "n_threads": 8,
                        "n_gpu_layers": 10,
                        "seed": 123,
                        "verbose": True,
                        "logits_all": True,
                    },
                }
            }
        )
        assert cfg.backend == "gguf"
        assert cfg.gguf is not None
        assert cfg.gguf.n_ctx == 4096
        assert cfg.gguf.n_threads == 8
        assert cfg.gguf.n_gpu_layers == 10

    def test_gguf_section_minimal(self) -> None:
        cfg = scorer_backend_config_from_toml(
            {"model": {"backend": "gguf", "model_path": "m.gguf"}}
        )
        assert cfg.gguf is not None
        assert cfg.gguf.n_ctx == 2048

    def test_invalid_backend_in_toml_raises(self) -> None:
        with pytest.raises(ConfigError, match="Invalid scorer backend"):
            scorer_backend_config_from_toml(
                {"model": {"backend": "unknown", "model_path": "m"}}
            )
