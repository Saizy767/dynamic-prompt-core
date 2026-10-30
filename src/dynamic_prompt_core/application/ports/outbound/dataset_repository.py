"""Outbound port: dataset artifact loading."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dynamic_prompt_core.domain.models.dataset import Dataset


@runtime_checkable
class DatasetRepository(Protocol):
    """Loads prepared dataset artifacts (jsonl or parquet)."""

    def load_artifact(self, path: str) -> Dataset:
        """Load a prepared dataset artifact and re-derive the three-way split."""
        ...
