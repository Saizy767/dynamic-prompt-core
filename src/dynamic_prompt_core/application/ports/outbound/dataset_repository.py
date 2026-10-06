from __future__ import annotations

from typing import Protocol, runtime_checkable

from dynamic_prompt_core.domain.models.dataset import Dataset, Record


@runtime_checkable
class DatasetRepository(Protocol):
    """Outbound port: contract for dataset loading and saving."""

    def load_dataset(self, path: str) -> list[Record]:
        """Load and validate a dataset from a file."""
        ...

    def load_artifact(self, path: str) -> Dataset:
        """Reload a prepared dataset artifact."""
        ...

    def write_artifact(
        self,
        dataset: Dataset,
        output_dir: str = "data",
        fmt: str = "jsonl",
        notes: str = "",
    ) -> str:
        """Persist a prepared dataset artifact. Returns the path written."""
        ...
