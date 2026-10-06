"""Server launcher for managing inference server subprocess lifecycle."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
from typing import Any

from dynamic_prompt_core.infrastructure.server.argv import build_argv
from dynamic_prompt_core.infrastructure.server.readiness import wait_for_ready

log = logging.getLogger(__name__)


class ServerLauncher:
    """Start and manage the lifecycle of an inference server subprocess."""

    def __init__(self, llm_cfg: dict[str, Any]) -> None:
        self._cfg = llm_cfg
        self._backend = llm_cfg["backend"]
        self.endpoint = f"http://{llm_cfg['host']}:{llm_cfg['port']}/v1"
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop_grace = 10.0

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def start(self, startup_timeout: float | None = None) -> str:
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
