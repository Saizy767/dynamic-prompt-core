"""Backend selection configuration for the candidate scorer.

``ScorerBackendConfig`` selects between the Hugging Face and GGUF backends and
carries the parameters needed to construct the corresponding adapters.  This
extends the existing configuration mechanism rather than introducing a
separate config file.
"""
from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.infrastructure.config.loader import ConfigError

VALID_SCORER_BACKENDS = ("huggingface", "gguf", "llama_server")

DEFAULT_LLAMA_SERVER_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_LLAMA_SERVER_TIMEOUT = 30


@dataclass(frozen=True)
class GgufParams:
    """GGUF inference parameters."""

    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int | None = None
    n_gpu_layers: int = 0
    seed: int = 42
    verbose: bool = False
    logits_all: bool = True


@dataclass(frozen=True)
class LlamaServerParams:
    """HTTP connection parameters for the llama-server scoring backend."""

    endpoint: str = DEFAULT_LLAMA_SERVER_ENDPOINT
    timeout: int = DEFAULT_LLAMA_SERVER_TIMEOUT


@dataclass(frozen=True)
class ScorerBackendConfig:
    """Backend selection for the candidate scorer factory."""

    backend: str
    model_path: str
    gguf: GgufParams | None = None
    llama_server: LlamaServerParams | None = None

    def __post_init__(self) -> None:
        if self.backend not in VALID_SCORER_BACKENDS:
            raise ConfigError(
                f"Invalid scorer backend {self.backend!r}; "
                f"must be one of {VALID_SCORER_BACKENDS}"
            )
        if not self.model_path:
            raise ConfigError("scorer backend requires a non-empty model_path")
        if self.backend == "gguf":
            if self.gguf is None:
                object.__setattr__(self, "gguf", GgufParams())
            assert self.gguf is not None
            p = self.gguf
            if p.n_ctx <= 0:
                raise ConfigError(f"invalid n_ctx {p.n_ctx}; must be positive")
            if p.n_batch <= 0:
                raise ConfigError(f"invalid n_batch {p.n_batch}; must be positive")
            if p.n_gpu_layers < 0:
                raise ConfigError(
                    f"invalid n_gpu_layers {p.n_gpu_layers}; must be >= 0"
                )
        if self.backend == "llama_server":
            if self.llama_server is None:
                object.__setattr__(
                    self, "llama_server", LlamaServerParams()
                )
            assert self.llama_server is not None
            ls = self.llama_server
            if not ls.endpoint.startswith(("http://", "https://")):
                raise ConfigError(
                    f"invalid endpoint {ls.endpoint!r}; "
                    f"must start with 'http://' or 'https://'"
                )
            if ls.timeout <= 0:
                raise ConfigError(
                    f"invalid timeout {ls.timeout}; must be positive"
                )


def scorer_backend_config_from_toml(
    raw: dict[str, object],
    *,
    default_model_path: str = "model",
) -> ScorerBackendConfig:
    """Build a ``ScorerBackendConfig`` from a parsed TOML dict.

    Reads the ``[model]`` section with ``backend`` and ``model_path`` and the
    ``[model.gguf]``     sub-table.  When ``[model]`` is absent, defaults to
    the Hugging Face backend with ``default_model_path`` (preserving existing
    behavior).
    """
    model_section = raw.get("model")
    if not isinstance(model_section, dict):
        return ScorerBackendConfig(
            backend="huggingface",
            model_path=default_model_path,
        )

    backend = str(model_section.get("backend", "huggingface"))
    model_path = str(model_section.get("model_path", default_model_path))

    gguf: GgufParams | None = None
    gguf_section = model_section.get("gguf")
    if isinstance(gguf_section, dict):
        gguf = GgufParams(
            n_ctx=int(gguf_section.get("n_ctx", 2048)),
            n_batch=int(gguf_section.get("n_batch", 512)),
            n_threads=_opt_int(gguf_section.get("n_threads")),
            n_gpu_layers=int(gguf_section.get("n_gpu_layers", 0)),
            seed=int(gguf_section.get("seed", 42)),
            verbose=bool(gguf_section.get("verbose", False)),
            logits_all=bool(gguf_section.get("logits_all", True)),
        )

    llama_server: LlamaServerParams | None = None
    ls_section = model_section.get("llama_server")
    if isinstance(ls_section, dict):
        llama_server = LlamaServerParams(
            endpoint=str(ls_section.get("endpoint", DEFAULT_LLAMA_SERVER_ENDPOINT)),
            timeout=int(ls_section.get("timeout", DEFAULT_LLAMA_SERVER_TIMEOUT)),
        )

    return ScorerBackendConfig(
        backend=backend,
        model_path=model_path,
        gguf=gguf,
        llama_server=llama_server,
    )


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)  # type: ignore[call-overload,no-any-return]
