"""File-based RunRepository implementation with crash-safe atomic writes."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
from datetime import UTC, datetime
from typing import Any, cast


class RunRepositoryError(ValueError):
    """Base error for run repository operations."""


class RunValidationError(RunRepositoryError):
    """Raised when result data fails validation before write."""


class RunWriteError(RunRepositoryError):
    """Raised when a disk write or atomic replace fails."""


class RunCorruptedError(RunRepositoryError):
    """Raised when a results file is corrupted."""


class RunConflictError(RunRepositoryError):
    """Raised when a save conflicts with an existing run_id."""


class RunRecoveryError(RunRepositoryError):
    """Raised when recovery encounters an unexpected problem."""


SCHEMA_VERSION = 1

_VALID_STATUSES = frozenset({"running", "completed", "failed", "interrupted"})


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
            status_path = os.path.join(directory, f"{run_id}.status.json")

            for rec in results:
                if not isinstance(rec, dict):
                    raise RunValidationError(f"result record is not a dict for run {run_id}")
                rec.setdefault("schema_version", SCHEMA_VERSION)

            payload = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in results)
            content_hash = _sha256_hex(payload)

            if os.path.exists(status_path):
                try:
                    with open(status_path, encoding="utf-8") as f:
                        existing_status = json.load(f)
                except (json.JSONDecodeError, OSError):
                    existing_status = {}
                existing_hash = existing_status.get("content_hash")
                if existing_hash == content_hash:
                    return path
                if existing_hash is not None:
                    raise RunConflictError(f"run {run_id} already exists with different data")

            self._set_status_internal(
                run_id, "running", status_path, started_at=datetime.now(UTC).isoformat()
            )

            try:
                if payload:
                    _atomic_write(path, payload, self._write_counter, jsonl=True)
                else:
                    os.makedirs(directory, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("")
            except Exception:
                raise

            self._set_status_internal(
                run_id,
                "completed",
                status_path,
                finished_at=datetime.now(UTC).isoformat(),
                content_hash=content_hash,
            )
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

    def _set_status_internal(
        self,
        run_id: str,
        status: str,
        status_path: str,
        *,
        started_at: str | None = None,
        finished_at: str | None = None,
        error: str | None = None,
        content_hash: str | None = None,
    ) -> None:
        if status not in _VALID_STATUSES:
            raise RunValidationError(f"invalid status: {status}")
        existing: dict[str, Any] = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        manifest = {
            "run_id": run_id,
            "status": status,
            "started_at": started_at or existing.get("started_at"),
            "finished_at": finished_at or existing.get("finished_at"),
            "error": error,
            "content_hash": content_hash or existing.get("content_hash"),
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2)
        _atomic_write(status_path, payload, self._write_counter)

    def set_status(
        self,
        run_id: str,
        status: str,
        results_dir: str,
        error: str | None = None,
    ) -> None:
        """Set the status of a run, persisting it atomically."""
        with self._lock:
            status_path = os.path.join(results_dir, f"{run_id}.status.json")
            self._set_status_internal(
                run_id,
                status,
                status_path,
                finished_at=datetime.now(UTC).isoformat() if status != "running" else None,
                error=error,
            )

    def get_status(self, run_id: str, results_dir: str) -> dict[str, Any]:
        """Read the status manifest for a run."""
        with self._lock:
            status_path = os.path.join(results_dir, f"{run_id}.status.json")
            if not os.path.exists(status_path):
                raise RunRepositoryError(f"no status found for run {run_id}")
            try:
                with open(status_path, encoding="utf-8") as f:
                    return cast(dict[str, Any], json.load(f))
            except json.JSONDecodeError as exc:
                raise RunCorruptedError(f"corrupted status file for run {run_id}: {exc}") from exc

    def mark_failed(self, run_id: str, results_dir: str, error: str) -> None:
        """Mark a run as failed with diagnostic information."""
        self.set_status(run_id, "failed", results_dir, error=error)

    def recover_interrupted(self, results_dir: str) -> list[str]:
        """Find runs left in ``running`` status and mark them ``interrupted``.

        Returns a list of recovered run IDs.  Does not modify result files.
        Idempotent: a second call returns an empty list.
        """
        with self._lock:
            recovered: list[str] = []
            if not os.path.isdir(results_dir):
                return recovered
            for name in os.listdir(results_dir):
                if not name.endswith(".status.json"):
                    continue
                status_path = os.path.join(results_dir, name)
                try:
                    with open(status_path, encoding="utf-8") as f:
                        manifest = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    raise RunRecoveryError(f"cannot read status file {status_path}: {exc}") from exc
                if manifest.get("status") == "running":
                    run_id = manifest.get("run_id", name.removesuffix(".status.json"))
                    self._set_status_internal(
                        run_id,
                        "interrupted",
                        status_path,
                        error="detected as interrupted on restart",
                    )
                    recovered.append(run_id)
            return recovered
