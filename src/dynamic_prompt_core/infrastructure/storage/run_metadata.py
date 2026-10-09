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
import json
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


def write_manifest(path: str, metadata: RunMetadata) -> str:
    """Write a run metadata manifest as JSON to ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_manifest(path: str) -> RunMetadata:
    """Load a run metadata manifest from JSON at ``path``."""
    with open(path, encoding="utf-8") as f:
        return RunMetadata.from_dict(json.load(f))


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
