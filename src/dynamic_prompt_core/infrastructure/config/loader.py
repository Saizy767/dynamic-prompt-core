"""Configuration loader for the LLM server and client."""
from __future__ import annotations

import tomllib
from typing import Any, cast

DEFAULT_CONFIG_PATH = "config.toml"
VALID_BACKENDS = ("llamacpp", "vllm")
REQUIRED_KEYS = ("backend", "model_path", "served_model_name", "host", "port")


class ConfigError(ValueError):
    """Raised when config.toml is missing, invalid, or incomplete."""


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the [llm] section of a TOML config file.

    Returns the [llm] table (with nested backend sub-tables preserved).
    Raises ConfigError naming the missing or invalid key on failure.
    """
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    if "llm" not in raw:
        raise ConfigError(f"Missing [llm] section in {path}")
    llm = raw["llm"]

    for key in REQUIRED_KEYS:
        if key not in llm:
            raise ConfigError(f"Missing required key [llm].{key} in {path}")

    backend = llm["backend"]
    if backend not in VALID_BACKENDS:
        raise ConfigError(
            f"Invalid [llm].backend {backend!r} in {path}; "
            f"must be one of {VALID_BACKENDS}"
        )
    return cast(dict[str, Any], llm)
