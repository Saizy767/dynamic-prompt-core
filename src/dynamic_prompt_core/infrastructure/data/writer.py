"""Dataset artifact persistence (write and reload prepared datasets)."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.domain.errors.dataset import DatasetError
from dynamic_prompt_core.domain.models.dataset import Dataset, Record
from dynamic_prompt_core.infrastructure.data.loader import _read_jsonl, _read_parquet


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
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
) -> dict[str, Any]:
    return {
        "id": r.id,
        "text": r.text,
        "label": r.label,
        "split": split,
        "notes": notes or r.notes,
    }


def _write_parquet(rows: list[dict[str, Any]], path: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError(
            "writing parquet requires 'pyarrow'; "
            "install with: pip install pyarrow"
        ) from exc
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

    dev: list[Record] = []
    holdout: list[Record] = []
    ambiguous: list[Record] = []
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


def prepare_from_config(
    config_path: str = "config.toml",
) -> tuple[Dataset, str]:
    """Load, split, and persist a dataset using config.toml settings."""
    from dynamic_prompt_core.domain.models.dataset import DatasetConfig
    from dynamic_prompt_core.domain.services.dataset_split import split_dataset
    from dynamic_prompt_core.infrastructure.data.loader import load_dataset

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
