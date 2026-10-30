"""File-based RunRepository implementation with crash-safe atomic writes."""

from __future__ import annotations

import itertools
import json
import os
import threading
from typing import Any

from dynamic_prompt_core.domain.errors.io import AtomicWriteError
from dynamic_prompt_core.domain.io.atomic import atomic_write


class RunRepositoryError(ValueError):
    """Base error for run repository operations."""


class RunValidationError(RunRepositoryError):
    """Raised when result data fails validation before write."""


class RunWriteError(RunRepositoryError):
    """Raised when a disk write or atomic replace fails."""


class RunCorruptedError(RunRepositoryError):
    """Raised when a results file is corrupted."""


SCHEMA_VERSION = 1


def _atomic_write(
    path: str, payload: str, counter: itertools.count[int], *, jsonl: bool = False
) -> None:
    """Write *payload* atomically via the shared domain helper.

    When *jsonl* is True, the read-back validation parses each line as a
    separate JSON document (JSONL format).  Otherwise it parses the whole
    file as a single JSON document.
    """
    try:
        atomic_write(path, payload, counter, jsonl=jsonl)
    except AtomicWriteError as exc:
        raise RunWriteError(str(exc)) from exc


class FileRunRepository:
    """Implements RunRepository via JSONL file I/O with atomic writes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._write_counter = itertools.count()

    def save_results(
        self,
        results: list[dict[str, Any]],
        run_id: str,
        path: str,
    ) -> str:
        """Persist results as JSONL atomically. Returns the path written."""
        with self._lock:
            for rec in results:
                if not isinstance(rec, dict):
                    raise RunValidationError(f"result record is not a dict for run {run_id}")
                rec.setdefault("schema_version", SCHEMA_VERSION)

            payload = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in results)
            _atomic_write(path, payload, self._write_counter, jsonl=True)
            return path

    def save_artifact(self, path: str, payload: str) -> str:
        """Persist a JSON document artifact atomically. Returns the path."""
        with self._lock:
            _atomic_write(path, payload, self._write_counter)
            return path

    def load_results(self, path: str) -> list[dict[str, Any]]:
        """Load JSONL results. Raises RunCorruptedError on bad lines."""
        with self._lock:
            if not os.path.exists(path):
                raise RunCorruptedError(f"results file not found: {path}")
            records: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise RunCorruptedError(
                            f"corrupted JSONL at {path} line {i}: {exc}"
                        ) from exc
            return records

    def load_results_safe(self, path: str) -> tuple[list[dict[str, Any]], str | None]:
        """Load results up to the first corrupt line.

        Returns ``(valid_records, corruption_info)`` where *corruption_info*
        is ``None`` when the file is clean, or a diagnostic string describing
        the first corruption found.
        """
        with self._lock:
            if not os.path.exists(path):
                return [], f"results file not found: {path}"
            records: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        return records, f"corrupted JSONL at {path} line {i}: {exc}"
            return records, None
