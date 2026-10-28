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


@dataclass(frozen=True)
class LlamaServerParams:
    """HTTP connection parameters for the llama-server scoring backend."""

    endpoint: str = DEFAULT_LLAMA_SERVER_ENDPOINT
    timeout: int = DEFAULT_LLAMA_SERVER_TIMEOUT


@dataclass(frozen=True)
class ScorerBackendConfig:
    """Backend selection for the candidate scorer factory."""

    backend: str
    llama_server: LlamaServerParams | None = None

    def __post_init__(self) -> None:
        if self.backend not in VALID_SCORER_BACKENDS:
            raise ConfigError(
                f"Invalid scorer backend {self.backend!r}; "
                f"must be one of {VALID_SCORER_BACKENDS}"
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
) -> ScorerBackendConfig:
    """Build a ``ScorerBackendConfig`` from a parsed TOML dict.

    Reads the ``[model]`` section with ``backend`` and the
    ``[model.llama_server]`` sub-table.  When ``[model]`` is absent, defaults
    to the llama_server backend with the default endpoint.
    """
    model_section = raw.get("model")
    if not isinstance(model_section, dict):
        return ScorerBackendConfig(backend="llama_server")

    backend = str(model_section.get("backend", "llama_server"))

    llama_server: LlamaServerParams | None = None
    ls_section = model_section.get("llama_server")
    if isinstance(ls_section, dict):
        llama_server = LlamaServerParams(
            endpoint=str(ls_section.get("endpoint", DEFAULT_LLAMA_SERVER_ENDPOINT)),
            timeout=int(ls_section.get("timeout", DEFAULT_LLAMA_SERVER_TIMEOUT)),
        )

    return ScorerBackendConfig(
        backend=backend,
        llama_server=llama_server,
    )
