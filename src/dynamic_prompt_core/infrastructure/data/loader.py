"""Dataset file loading from jsonl and parquet formats."""
from __future__ import annotations

import json
import os
from typing import Any, cast

from dynamic_prompt_core.domain.errors.dataset import DatasetError
from dynamic_prompt_core.domain.models.dataset import REQUIRED_FIELDS, Record


def load_dataset(path: str) -> list[Record]:
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


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_parquet(path: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:
            raise DatasetError(
                "reading parquet requires 'pyarrow' or 'pandas'; "
                "install with: pip install pyarrow"
            ) from exc
        df = pd.read_parquet(path)
        return cast(list[dict[str, Any]], df.to_dict(orient="records"))
    table = pq.read_table(path)
    return cast(list[dict[str, Any]], table.to_pylist())


def _validate(rows: list[dict[str, Any]]) -> list[Record]:
    records: list[Record] = []
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
