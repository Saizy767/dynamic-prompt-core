"""
LLM server launcher: start an OpenAI-compatible inference server
(llama-server or vllm) from config.toml, wait for it to be ready, and
hand off the live endpoint.

Usage:
    python server_launcher.py --config config.toml
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from typing import Any, Optional

log = logging.getLogger(__name__)

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
    return llm


# --------------------------------------------------------------------------- #
#  Backend argv builders
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Readiness polling
# --------------------------------------------------------------------------- #
def wait_for_ready(
    endpoint: str, timeout: float, interval: float = 0.5
) -> bool:
    """Poll GET {endpoint}/models until HTTP 200 or timeout elapses.

    Returns True if the server became ready, False on timeout.
    """
    url = endpoint.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# --------------------------------------------------------------------------- #
#  ServerLauncher
# --------------------------------------------------------------------------- #
class ServerLauncher:
    """Start and manage the lifecycle of an inference server subprocess."""

    def __init__(self, llm_cfg: dict[str, Any]) -> None:
        self._cfg = llm_cfg
        self._backend = llm_cfg["backend"]
        self.endpoint = f"http://{llm_cfg['host']}:{llm_cfg['port']}/v1"
        self._proc: Optional[subprocess.Popen] = None
        self._stop_grace = 10.0

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    def start(self, startup_timeout: Optional[float] = None) -> str:
        """Spawn the server, wait for readiness, return the endpoint URL."""
        argv = build_argv(self._backend, self._cfg)
        timeout = (
            startup_timeout
            if startup_timeout is not None
            else self._cfg.get("startup_timeout", 60)
        )
        log.info("Starting server: %s", " ".join(argv))
        self._proc = subprocess.Popen(argv, start_new_session=True)
        log.info(
            "Server PID %d, waiting for readiness (timeout=%ss)",
            self._proc.pid, timeout,
        )
        if not wait_for_ready(self.endpoint, timeout):
            self.stop()
            raise RuntimeError(
                f"Server did not become ready within {timeout}s "
                f"at {self.endpoint}. The port may already be in use."
            )
        log.info("Server ready at %s", self.endpoint)
        return self.endpoint

    def stop(self) -> None:
        """Terminate the server process group (SIGTERM, then SIGKILL)."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return
        try:
            pgid = os.getpgid(self._proc.pid)
        except ProcessLookupError:
            self._proc = None
            return
        log.info("Stopping server (PID %d, PGID %d)", self._proc.pid, pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=self._stop_grace)
        except subprocess.TimeoutExpired:
            log.warning("SIGTERM grace expired, sending SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._proc.wait()
        self._proc = None
        log.info("Server stopped")

    def install_signal_handlers(self) -> None:
        """Wire SIGINT/SIGTERM to stop the server."""
        def _handler(signum: int, frame: Any) -> None:
            log.info("Received signal %d, stopping server", signum)
            self.stop()
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch an LLM inference server from config.toml."
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    launcher = ServerLauncher(cfg)
    try:
        endpoint = launcher.start()
    except VllmNotSupportedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Server ready at {endpoint} (PID {launcher.pid}). "
        "Press Ctrl+C to stop."
    )
    launcher.install_signal_handlers()
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        launcher.stop()


if __name__ == "__main__":
    main()
