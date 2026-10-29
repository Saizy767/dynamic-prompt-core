"""File-based RunRepository implementation with crash-safe atomic writes."""

from __future__ import annotations

import itertools
import json
import os
import threading
from typing import Any


class RunRepositoryError(ValueError):
    """Base error for run repository operations."""


class RunValidationError(RunRepositoryError):
    """Raised when result data fails validation before write."""


class RunWriteError(RunRepositoryError):
    """Raised when a disk write or atomic replace fails."""


class RunCorruptedError(RunRepositoryError):
    """Raised when a results file is corrupted."""


SCHEMA_VERSION = 1


def _fsync_directory(directory: str) -> None:
    """Best-effort fsync on the parent directory for crash durability."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _atomic_write(
    path: str, payload: str, counter: itertools.count[int], *, jsonl: bool = False
) -> None:
    """Write *payload* to *path* atomically via a unique temp file.

    When *jsonl* is True, the read-back validation parses each line as a
    separate JSON document (JSONL format).  Otherwise it parses the whole
    file as a single JSON document.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    counter_val = next(counter)
    tmp = os.path.join(
        directory,
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{counter_val}",
    )
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, encoding="utf-8") as vf:
            if jsonl:
                for line in vf:
                    line = line.strip()
                    if line:
                        json.loads(line)
            else:
                json.load(vf)
        os.replace(tmp, path)
        _fsync_directory(directory)
    except OSError as exc:
        raise RunWriteError(f"failed to write file {path}: {exc}") from exc
    except Exception as exc:
        raise RunWriteError(f"failed to write file {path}: {exc}") from exc
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


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
            directory = os.path.dirname(path) or "."

            for rec in results:
                if not isinstance(rec, dict):
                    raise RunValidationError(f"result record is not a dict for run {run_id}")
                rec.setdefault("schema_version", SCHEMA_VERSION)

            payload = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in results)

            if payload:
                _atomic_write(path, payload, self._write_counter, jsonl=True)
            else:
                os.makedirs(directory, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
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
