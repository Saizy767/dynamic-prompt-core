from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunRepository(Protocol):
    """Outbound port: contract for run result storage."""

    def save_results(
        self, results: list[dict[str, Any]], run_id: str, path: str
    ) -> str:
        """Persist run results to a file. Returns the path written."""
        ...

    def load_results(self, path: str) -> list[dict[str, Any]]:
        """Load run results from a file."""
        ...
