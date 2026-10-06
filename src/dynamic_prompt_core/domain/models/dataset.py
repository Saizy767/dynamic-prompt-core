"""Domain models for datasets: Record, Dataset, DatasetConfig."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_SEED = 42
DEFAULT_HOLDOUT_RATIO = 0.2

REQUIRED_FIELDS = ("id", "text", "label")


@dataclass
class Record:
    id: Any
    text: str
    label: int
    ambiguous: bool = False
    notes: str = ""

    def to_dict(self, split: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "split": split,
            "notes": self.notes,
        }


@dataclass
class Dataset:
    dev: list[Record] = field(default_factory=list)
    holdout: list[Record] = field(default_factory=list)
    ambiguous: list[Record] = field(default_factory=list)
    seed: int = DEFAULT_SEED
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO
    source_path: str = ""

    @property
    def all_records(self) -> list[Record]:
        return self.dev + self.holdout + self.ambiguous

    def ids(self, group: str) -> set[Any]:
        return {r.id for r in getattr(self, group)}


@dataclass
class DatasetConfig:
    path: str = ""
    seed: int = DEFAULT_SEED
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> DatasetConfig:
        """Read [dataset] section from config.toml with defaults."""
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        ds = config.get("dataset", {})
        return cls(
            path=ds.get("path", ""),
            seed=int(ds.get("seed", DEFAULT_SEED)),
            holdout_ratio=float(ds.get("holdout_ratio", DEFAULT_HOLDOUT_RATIO)),
        )
