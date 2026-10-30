"""Backend selection configuration for the candidate scorer.

``ScorerBackendConfig`` selects the llama-server HTTP backend — the single
scoring backend — and carries the connection parameters needed to construct
the scoring adapters.  This extends the existing configuration mechanism
rather than introducing a separate config file.
"""

from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.infrastructure.config.loader import ConfigError

VALID_SCORER_BACKENDS = ("llama_server",)

DEFAULT_LLAMA_SERVER_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_LLAMA_SERVER_TIMEOUT = 30
DEFAULT_LLAMA_SERVER_MAX_CONCURRENCY = 2


@dataclass(frozen=True)
class LlamaServerParams:
    """HTTP connection parameters for the llama-server scoring backend."""

    endpoint: str = DEFAULT_LLAMA_SERVER_ENDPOINT
    timeout: int = DEFAULT_LLAMA_SERVER_TIMEOUT
    gguf_path: str | None = None
    max_concurrency: int = DEFAULT_LLAMA_SERVER_MAX_CONCURRENCY


@dataclass(frozen=True)
class ScorerBackendConfig:
    """Backend selection for the candidate scorer factory."""

    backend: str
    llama_server: LlamaServerParams | None = None

    def __post_init__(self) -> None:
        if self.backend not in VALID_SCORER_BACKENDS:
            raise ConfigError(
                f"Invalid scorer backend {self.backend!r}; must be one of {VALID_SCORER_BACKENDS}"
            )
        if self.backend == "llama_server":
            if self.llama_server is None:
                object.__setattr__(self, "llama_server", LlamaServerParams())
            assert self.llama_server is not None
            ls = self.llama_server
            if not ls.endpoint.startswith(("http://", "https://")):
                raise ConfigError(
                    f"invalid endpoint {ls.endpoint!r}; must start with 'http://' or 'https://'"
                )
            if ls.timeout <= 0:
                raise ConfigError(f"invalid timeout {ls.timeout}; must be positive")
            if ls.max_concurrency <= 0:
                raise ConfigError(f"invalid max_concurrency {ls.max_concurrency}; must be positive")


def scorer_backend_config_from_toml(
    raw: dict[str, object],
) -> ScorerBackendConfig:
    """Build a ``ScorerBackendConfig`` from a parsed TOML dict.

    Reads the ``[model]`` section with ``backend`` and the
    ``[model.llama_server]`` sub-table.  When ``[model]`` is absent, defaults
    to the llama_server backend with the default endpoint.  ``gguf_path`` is
    taken from the ``[llm]`` server-launch section so run-identity manifests
    can checksum the model file.
    """
    model_section = raw.get("model")
    llm_section = raw.get("llm")
    runner_section = raw.get("runner")
    gguf_path = (
        str(llm_section["gguf_path"])
        if isinstance(llm_section, dict) and llm_section.get("gguf_path")
        else None
    )
    # Scoring concurrency follows [runner].concurrency unless the scorer
    # section overrides it, so both stages use the same parallelism budget.
    max_concurrency = (
        int(runner_section["concurrency"])
        if isinstance(runner_section, dict) and runner_section.get("concurrency")
        else DEFAULT_LLAMA_SERVER_MAX_CONCURRENCY
    )
    if not isinstance(model_section, dict):
        return ScorerBackendConfig(
            backend="llama_server",
            llama_server=LlamaServerParams(gguf_path=gguf_path, max_concurrency=max_concurrency),
        )

    backend = str(model_section.get("backend", "llama_server"))

    llama_server: LlamaServerParams | None = None
    ls_section = model_section.get("llama_server")
    if isinstance(ls_section, dict):
        llama_server = LlamaServerParams(
            endpoint=str(ls_section.get("endpoint", DEFAULT_LLAMA_SERVER_ENDPOINT)),
            timeout=int(ls_section.get("timeout", DEFAULT_LLAMA_SERVER_TIMEOUT)),
            gguf_path=gguf_path,
            max_concurrency=int(ls_section.get("max_concurrency", max_concurrency)),
        )
    elif backend == "llama_server":
        llama_server = LlamaServerParams(gguf_path=gguf_path, max_concurrency=max_concurrency)

    return ScorerBackendConfig(
        backend=backend,
        llama_server=llama_server,
    )
