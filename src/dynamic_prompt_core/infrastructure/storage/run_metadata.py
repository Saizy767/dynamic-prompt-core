"""Run metadata manifest: recording and resume identity verification.

A ``RunMetadata`` captures the provenance of an evaluation run — backend,
model content identity, dataset fingerprint, prompt version, and inference
settings — so resume operations can verify that the model, dataset, and
scoring configuration match the saved run.

Result-changing (semantic) settings are distinguished from non-semantic
settings (e.g. logging verbosity).  A mismatch in a semantic setting either
rejects the resume or explicitly records the run as a new experiment; it does
not silently continue with a different configuration.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any

SEMANTIC_FIELDS: frozenset[str] = frozenset(
    {
        "backend",
        "model_checksum",
        "dataset_fingerprint",
        "prompt_sha256",
        "prompt_version",
    }
)


@dataclass(frozen=True)
class RunMetadata:
    """Provenance metadata for an evaluation run."""

    run_id: str
    prompt_version: str
    prompt_sha256: str | None = None
    split: str = "dev"
    dataset_path: str | None = None
    dataset_fingerprint: str | None = None
    backend: str | None = None
    model_path: str | None = None
    model_checksum: str | None = None
    n_vocab: int | None = None
    inference_settings: dict[str, Any] = field(default_factory=dict)
    runtime_versions: dict[str, str] = field(default_factory=dict)
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunMetadata:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass(frozen=True)
class ResumeVerificationResult:
    """Outcome of comparing recorded metadata against current metadata."""

    matches: bool
    mismatched_semantic_fields: list[str]

    @property
    def should_reject(self) -> bool:
        return not self.matches


def verify_resume_identity(
    recorded: RunMetadata,
    current: RunMetadata,
) -> ResumeVerificationResult:
    """Compare semantic fields of two ``RunMetadata`` instances.

    Only result-changing settings (model contents, tokenizer, prompt version,
    dataset fingerprint, backend) are compared.  Non-semantic settings
    (timestamp, logging verbosity) are ignored.  Returns a result indicating
    whether the resume is safe and which semantic fields mismatched.
    """
    mismatched: list[str] = []
    for field_name in sorted(SEMANTIC_FIELDS):
        recorded_val = getattr(recorded, field_name, None)
        current_val = getattr(current, field_name, None)
        if recorded_val is not None and current_val is not None:
            if recorded_val != current_val:
                mismatched.append(field_name)
    return ResumeVerificationResult(
        matches=len(mismatched) == 0,
        mismatched_semantic_fields=mismatched,
    )


class RunMetadataError(ValueError):
    """Raised when a manifest file is corrupted or cannot be read."""


_manifest_counter = itertools.count()


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


def write_manifest(path: str, metadata: RunMetadata) -> str:
    """Write a run metadata manifest as JSON to ``path`` atomically."""
    payload = json.dumps(metadata.to_dict(), indent=2, sort_keys=True)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    counter_val = next(_manifest_counter)
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
            json.load(vf)
        os.replace(tmp, path)
        _fsync_directory(directory)
    except OSError as exc:
        raise RunMetadataError(f"failed to write manifest {path}: {exc}") from exc
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def load_manifest(path: str) -> RunMetadata:
    """Load a run metadata manifest from JSON at ``path``."""
    try:
        with open(path, encoding="utf-8") as f:
            return RunMetadata.from_dict(json.load(f))
    except json.JSONDecodeError as exc:
        raise RunMetadataError(f"corrupted manifest {path}: {exc}") from exc
    except FileNotFoundError as exc:
        raise RunMetadataError(f"manifest not found: {path}") from exc


def build_run_metadata(
    run_id: str,
    prompt_version: str,
    backend_metadata: dict[str, Any],
    *,
    prompt_sha256: str | None = None,
    split: str = "dev",
    dataset_path: str | None = None,
    dataset_fingerprint: str | None = None,
    timestamp: str | None = None,
) -> RunMetadata:
    """Build ``RunMetadata`` from a backend adapter's ``describe()`` dict."""
    inference_settings = {
        k: v
        for k, v in backend_metadata.items()
        if k
        not in {
            "backend",
            "model_path",
            "model_checksum",
            "n_vocab",
        }
    }
    return RunMetadata(
        run_id=run_id,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        split=split,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        backend=backend_metadata.get("backend"),
        model_path=backend_metadata.get("model_path"),
        model_checksum=backend_metadata.get("model_checksum"),
        n_vocab=backend_metadata.get("n_vocab"),
        inference_settings=inference_settings,
        timestamp=timestamp,
    )
