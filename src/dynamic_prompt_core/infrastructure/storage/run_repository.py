"""File-based RunRepository implementation."""
from __future__ import annotations

import json
import os
from typing import Any


class FileRunRepository:
    """Implements RunRepository via JSONL file I/O."""

    def save_results(
        self,
        results: list[dict[str, Any]],
        run_id: str,
        path: str,
    ) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def load_results(self, path: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
