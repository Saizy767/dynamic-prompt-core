"""Unit tests for scorer backend configuration (single llama_server backend)."""

from __future__ import annotations

import pytest

from dynamic_prompt_core.infrastructure.config.loader import ConfigError
from dynamic_prompt_core.infrastructure.llm.scoring.config import (
    LlamaServerParams,
    ScorerBackendConfig,
    scorer_backend_config_from_toml,
)


class TestScorerBackendConfigValidation:
    def test_valid_llama_server_config_defaults(self) -> None:
        cfg = ScorerBackendConfig(backend="llama_server")
        assert cfg.llama_server is not None
        assert cfg.llama_server.endpoint == "http://127.0.0.1:8080"
        assert cfg.llama_server.timeout == 30

    def test_llama_server_custom_endpoint(self) -> None:
        cfg = ScorerBackendConfig(
            backend="llama_server",
            llama_server=LlamaServerParams(
                endpoint="http://192.168.1.10:9090",
                timeout=60,
            ),
        )
        assert cfg.llama_server is not None
        assert cfg.llama_server.endpoint == "http://192.168.1.10:9090"
        assert cfg.llama_server.timeout == 60

    def test_removed_backends_rejected(self) -> None:
        for backend in ("gguf", "huggingface", "vllm"):
            with pytest.raises(ConfigError, match="Invalid scorer backend"):
                ScorerBackendConfig(backend=backend)

    def test_llama_server_invalid_endpoint_raises(self) -> None:
        with pytest.raises(ConfigError, match="invalid endpoint"):
            ScorerBackendConfig(
                backend="llama_server",
                llama_server=LlamaServerParams(endpoint="ftp://bad"),
            )

    def test_llama_server_invalid_timeout_raises(self) -> None:
        with pytest.raises(ConfigError, match="invalid timeout"):
            ScorerBackendConfig(
                backend="llama_server",
                llama_server=LlamaServerParams(timeout=0),
            )


class TestScorerBackendConfigFromToml:
    def test_absent_model_section_defaults_to_llama_server(self) -> None:
        cfg = scorer_backend_config_from_toml({"llm": {"model_path": "model"}})
        assert cfg.backend == "llama_server"
        assert cfg.llama_server is not None
        assert cfg.llama_server.endpoint == "http://127.0.0.1:8080"
        assert cfg.llama_server.timeout == 30

    def test_llama_server_section_with_params(self) -> None:
        cfg = scorer_backend_config_from_toml(
            {
                "model": {
                    "backend": "llama_server",
                    "llama_server": {
                        "endpoint": "http://10.0.0.1:7070",
                        "timeout": 45,
                    },
                }
            }
        )
        assert cfg.backend == "llama_server"
        assert cfg.llama_server is not None
        assert cfg.llama_server.endpoint == "http://10.0.0.1:7070"
        assert cfg.llama_server.timeout == 45

    def test_llama_server_section_defaults(self) -> None:
        cfg = scorer_backend_config_from_toml({"model": {"backend": "llama_server"}})
        assert cfg.llama_server is not None
        assert cfg.llama_server.endpoint == "http://127.0.0.1:8080"
        assert cfg.llama_server.timeout == 30

    def test_invalid_backend_in_toml_raises(self) -> None:
        with pytest.raises(ConfigError, match="Invalid scorer backend"):
            scorer_backend_config_from_toml({"model": {"backend": "gguf"}})

    def test_unknown_backend_in_toml_raises(self) -> None:
        with pytest.raises(ConfigError, match="Invalid scorer backend"):
            scorer_backend_config_from_toml({"model": {"backend": "unknown"}})

    def test_gguf_path_taken_from_llm_section(self) -> None:
        raw = {
            "llm": {"gguf_path": "model/qwen.gguf"},
            "model": {
                "backend": "llama_server",
                "llama_server": {"endpoint": "http://10.0.0.1:7070"},
            },
        }
        cfg = scorer_backend_config_from_toml(raw)
        assert cfg.llama_server is not None
        assert cfg.llama_server.gguf_path == "model/qwen.gguf"

    def test_gguf_path_defaults_without_llm_section(self) -> None:
        cfg = scorer_backend_config_from_toml(
            {"model": {"backend": "llama_server", "llama_server": {}}}
        )
        assert cfg.llama_server is not None
        assert cfg.llama_server.gguf_path is None

    def test_max_concurrency_follows_runner_concurrency(self) -> None:
        cfg = scorer_backend_config_from_toml({"runner": {"concurrency": 4}})
        assert cfg.llama_server is not None
        assert cfg.llama_server.max_concurrency == 4

    def test_max_concurrency_override_beats_runner(self) -> None:
        cfg = scorer_backend_config_from_toml(
            {
                "runner": {"concurrency": 4},
                "model": {
                    "backend": "llama_server",
                    "llama_server": {"max_concurrency": 8},
                },
            }
        )
        assert cfg.llama_server is not None
        assert cfg.llama_server.max_concurrency == 8

    def test_max_concurrency_defaults_to_two(self) -> None:
        cfg = scorer_backend_config_from_toml({})
        assert cfg.llama_server is not None
        assert cfg.llama_server.max_concurrency == 2

    def test_invalid_max_concurrency_raises(self) -> None:
        with pytest.raises(ConfigError, match="invalid max_concurrency"):
            ScorerBackendConfig(
                backend="llama_server",
                llama_server=LlamaServerParams(max_concurrency=0),
            )
