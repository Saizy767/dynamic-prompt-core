from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PromptRepository(Protocol):
    """Outbound port: contract for prompt version storage."""

    def save(
        self,
        prompt_version_dict: dict[str, Any],
        reason: str = "composed",
        base_version: int | None = None,
    ) -> dict[str, Any]:
        """Persist a new prompt version."""
        ...

    def get(self, version_number: int) -> dict[str, Any]:
        """Return the full record for a version."""
        ...

    def get_active(self) -> dict[str, Any]:
        """Return the active version record."""
        ...

    def activate(self, version_number: int) -> dict[str, Any]:
        """Mark a version as active."""
        ...

    def list_versions(self) -> list[dict[str, Any]]:
        """Return metadata for all versions."""
        ...

    def lineage(self, version_number: int) -> list[dict[str, Any]]:
        """Return the ancestry chain for a version."""
        ...
