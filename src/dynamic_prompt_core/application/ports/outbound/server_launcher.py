from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ServerLauncher(Protocol):
    """Outbound port: contract for inference server lifecycle."""

    def start(self, startup_timeout: float | None = None) -> str:
        """Start the server and return the endpoint URL."""
        ...

    def stop(self) -> None:
        """Stop the server gracefully."""
        ...

    def health(self) -> bool:
        """Return True if the server is healthy and ready."""
        ...
