"""Command-line argv builders for LLM server backends."""
from __future__ import annotations

import sys
from typing import Any

from dynamic_prompt_core.infrastructure.config.loader import ConfigError


class VllmNotSupportedError(RuntimeError):
    """Raised when vLLM is selected on an unsupported platform."""


def _check_platform(backend: str) -> None:
    if backend == "vllm" and sys.platform == "darwin":
        raise VllmNotSupportedError(
            "vLLM is not supported on macOS (darwin). vLLM requires a Linux "
            'environment with a GPU. Use backend = "llamacpp" on macOS, or '
            "run this launcher on Linux for vLLM."
        )


def build_argv(backend: str, llm_cfg: dict[str, Any]) -> list[str]:
    """Build the command-line argv for the given backend from config."""
    _check_platform(backend)
    if backend == "llamacpp":
        return _build_llamacpp_argv(llm_cfg)
    if backend == "vllm":
        return _build_vllm_argv(llm_cfg)
    raise ConfigError(f"Unknown backend {backend!r}")


def _build_llamacpp_argv(cfg: dict[str, Any]) -> list[str]:
    if "gguf_path" not in cfg:
        raise ConfigError(
            "Missing required key [llm].gguf_path for llamacpp backend "
            "(path to the .gguf model file)"
        )
    argv = [
        "llama-server",
        "-m", str(cfg["gguf_path"]),
        "--host", str(cfg["host"]),
        "--port", str(cfg["port"]),
        "-c", str(cfg.get("ctx_size", 4096)),
        "-ngl", str(cfg.get("gpu_layers", 0)),
        "-t", str(cfg.get("threads", 4)),
        "--alias", str(cfg["served_model_name"]),
        "--jinja",
    ]
    extra = cfg.get("llamacpp", {}).get("extra_args", [])
    if extra:
        argv.extend(str(a) for a in extra)
    return argv


def _build_vllm_argv(cfg: dict[str, Any]) -> list[str]:
    argv = [
        "vllm", "serve",
        str(cfg["model_path"]),
        "--host", str(cfg["host"]),
        "--port", str(cfg["port"]),
        "--max-model-len", str(cfg.get("ctx_size", 4096)),
    ]
    vllm_opts = cfg.get("vllm", {})
    if "gpu_memory_utilization" in vllm_opts:
        argv += ["--gpu-memory-utilization", str(vllm_opts["gpu_memory_utilization"])]
    if "tensor_parallel_size" in vllm_opts:
        argv += ["--tensor-parallel-size", str(vllm_opts["tensor_parallel_size"])]
    return argv
