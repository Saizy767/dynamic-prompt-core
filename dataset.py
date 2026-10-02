"""
Dataset loader, validator, splitter, and artifact persistence for the
binary-classification prompt-optimization pipeline.

Load  -> validate -> split (dev / holdout / ambiguous) -> persist artifact.

The split is reproducible: same source file + same seed yields identical
dev and holdout. Ambiguous records (ambiguous=true) are excluded from dev
and holdout. Holdout metrics are logged by the runner but MUST NOT drive
accept/reject decisions on Stage 1 (enforced by convention in the runner,
not here).
"""
from __future__ import annotations

import json
import os
import random
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_SEED = 42
DEFAULT_HOLDOUT_RATIO = 0.2

REQUIRED_FIELDS = ("id", "text", "label")


# --------------------------------------------------------------------------- #
#  Errors
# --------------------------------------------------------------------------- #
class DatasetError(ValueError):
    """Raised when a dataset file is invalid or cannot be loaded."""


# --------------------------------------------------------------------------- #
#  Record
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    id: Any
    text: str
    label: int
    ambiguous: bool = False
    notes: str = ""

    def to_dict(self, split: str) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "split": split,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
#  Dataset (split result)
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    dev: List[Record] = field(default_factory=list)
    holdout: List[Record] = field(default_factory=list)
    ambiguous: List[Record] = field(default_factory=list)
    seed: int = DEFAULT_SEED
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO
    source_path: str = ""

    @property
    def all_records(self) -> List[Record]:
        return self.dev + self.holdout + self.ambiguous

    def ids(self, group: str) -> set:
        return {r.id for r in getattr(self, group)}


# --------------------------------------------------------------------------- #
#  Loader
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> List[Record]:
    """Load and validate a dataset from a jsonl or parquet file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        rows = _read_jsonl(path)
    elif ext == ".parquet":
        rows = _read_parquet(path)
    else:
        raise DatasetError(
            f"unsupported file extension '{ext}' for '{path}' "
            "(expected .jsonl or .parquet)"
        )
    return _validate(rows)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_parquet(path: str) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError:
            raise DatasetError(
                f"reading parquet requires 'pyarrow' or 'pandas'; "
                f"install with: pip install pyarrow"
            )
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    table = pq.read_table(path)
    return table.to_pylist()


def _validate(rows: List[Dict[str, Any]]) -> List[Record]:
    records: List[Record] = []
    for row in rows:
        rid = row.get("id")
        for fname in REQUIRED_FIELDS:
            if fname not in row or row[fname] is None:
                raise DatasetError(
                    f"record id={rid!r} is missing required field '{fname}'"
                )
        label = row["label"]
        if label not in (0, 1):
            raise DatasetError(
                f"record id={rid!r} has invalid label {label!r} "
                "(expected 0 or 1)"
            )
        records.append(Record(
            id=rid,
            text=str(row["text"]),
            label=int(label),
            ambiguous=bool(row.get("ambiguous", False)),
            notes=str(row.get("notes", "")),
        ))
    return records


# --------------------------------------------------------------------------- #
#  Split
# --------------------------------------------------------------------------- #
def split_dataset(
    records: List[Record],
    seed: int = DEFAULT_SEED,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    source_path: str = "",
) -> Dataset:
    """Split records into dev, holdout, and ambiguous groups.

    The split is reproducible: records are sorted by id (deterministic order
    independent of file row order), then shuffled with random.Random(seed).
    Ambiguous records are separated before the shuffle.
    """
    ambiguous = [r for r in records if r.ambiguous]
    non_ambiguous = [r for r in records if not r.ambiguous]

    ordered = sorted(non_ambiguous, key=lambda r: str(r.id))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    cut = int(n * holdout_ratio)
    if n > 0 and not (0 < cut < n):
        raise DatasetError(
            f"holdout_ratio {holdout_ratio} produces an invalid split: "
            f"cut={cut} for {n} non-ambiguous records. "
            f"Choose a ratio strictly between 0 and 1."
        )

    holdout = ordered[:cut]
    dev = ordered[cut:]

    return Dataset(
        dev=dev,
        holdout=holdout,
        ambiguous=ambiguous,
        seed=seed,
        holdout_ratio=holdout_ratio,
        source_path=source_path,
    )


# --------------------------------------------------------------------------- #
#  Artifact persistence
# --------------------------------------------------------------------------- #
def write_artifact(
    dataset: Dataset,
    output_dir: str = "data",
    fmt: str = "jsonl",
    notes: str = "",
) -> str:
    """Persist the prepared dataset as a reproducible artifact.

    Returns the path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ext = "jsonl" if fmt == "jsonl" else "parquet"
    filename = f"prepared_{dataset.seed}_{ts}.{ext}"
    path = os.path.join(output_dir, filename)

    rows = []
    for r in dataset.dev:
        rows.append(_artifact_row(r, "dev", notes, dataset))
    for r in dataset.holdout:
        rows.append(_artifact_row(r, "holdout", notes, dataset))
    for r in dataset.ambiguous:
        rows.append(_artifact_row(r, "ambiguous", notes, dataset))

    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        _write_parquet(rows, path)

    return path


def _artifact_row(
    r: Record, split: str, notes: str, dataset: Dataset
) -> Dict[str, Any]:
    return {
        "id": r.id,
        "text": r.text,
        "label": r.label,
        "split": split,
        "notes": notes or r.notes,
    }


def _write_parquet(rows: List[Dict[str, Any]], path: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise DatasetError(
            "writing parquet requires 'pyarrow'; "
            "install with: pip install pyarrow"
        )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def load_artifact(path: str) -> Dataset:
    """Reload a prepared artifact and re-derive the three-way split."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        rows = _read_jsonl(path)
    elif ext == ".parquet":
        rows = _read_parquet(path)
    else:
        raise DatasetError(f"unsupported artifact extension '{ext}'")

    dev: List[Record] = []
    holdout: List[Record] = []
    ambiguous: List[Record] = []
    for row in rows:
        rec = Record(
            id=row["id"],
            text=str(row["text"]),
            label=int(row["label"]),
            ambiguous=False,
            notes=str(row.get("notes", "")),
        )
        split = row.get("split", "")
        if split == "dev":
            dev.append(rec)
        elif split == "holdout":
            holdout.append(rec)
        elif split == "ambiguous":
            rec.ambiguous = True
            ambiguous.append(rec)

    return Dataset(
        dev=dev, holdout=holdout, ambiguous=ambiguous,
        source_path=path,
    )


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    path: str = ""
    seed: int = DEFAULT_SEED
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO

    @classmethod
    def from_config(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> "DatasetConfig":
        """Read [dataset] section from config.toml with defaults."""
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        ds = config.get("dataset", {})
        return cls(
            path=ds.get("path", ""),
            seed=int(ds.get("seed", DEFAULT_SEED)),
            holdout_ratio=float(ds.get("holdout_ratio", DEFAULT_HOLDOUT_RATIO)),
        )


def prepare_from_config(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Tuple[Dataset, str]:
    """Load, split, and persist a dataset using config.toml settings."""
    cfg = DatasetConfig.from_config(config_path)
    if not cfg.path:
        raise DatasetError(
            "[dataset].path is not set in config.toml"
        )
    records = load_dataset(cfg.path)
    dataset = split_dataset(
        records, seed=cfg.seed, holdout_ratio=cfg.holdout_ratio,
        source_path=cfg.path,
    )
    artifact_path = write_artifact(
        dataset, output_dir=os.path.dirname(cfg.path) or "data",
        notes=f"source={os.path.basename(cfg.path)} ratio={cfg.holdout_ratio}",
    )
    return dataset, artifact_path
