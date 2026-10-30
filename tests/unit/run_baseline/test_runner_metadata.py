"""Tests for BaselineRunner run-metadata manifest and resume identity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
    BaselineRunner,
    RunnerConfig,
)
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.data.writer import FileDatasetRepository


class _ClassifierWithDescribe:
    """Stub classifier carrying a ``describe()`` dict."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.classify = AsyncMock()
        self._metadata = metadata

    def describe(self) -> dict[str, Any]:
        return dict(self._metadata)


def _make_extractor() -> AsyncMock:
    return AsyncMock()


def _make_runner(classifier: Any, tmp_path: Path) -> BaselineRunner:
    config = RunnerConfig(output_dir=str(tmp_path))
    runner = BaselineRunner(
        dataset_repository=FileDatasetRepository(),
        classifier=classifier,
        extractor=_make_extractor(),
        config=config,
        split="dev",
        candidates=[Candidate("0"), Candidate("1")],
        run_id="test-run",
    )
    runner._timestamp = "20260101T000000Z"
    return runner


_BACKEND_META = {
    "backend": "gguf",
    "model_path": "m.gguf",
    "model_checksum": "abc",
    "n_vocab": 32000,
    "n_ctx": 2048,
}


class TestManifestWrite:
    def test_manifest_written_from_describe(self, tmp_path: Path) -> None:
        classifier = _ClassifierWithDescribe(_BACKEND_META)
        runner = _make_runner(classifier, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)
        with open(runner._manifest_path()) as f:
            data = json.load(f)
        assert data["backend"] == "gguf"
        assert data["model_checksum"] == "abc"
        assert data["run_id"] == "test-run"

    def test_no_manifest_without_describe(self, tmp_path: Path) -> None:
        classifier = MagicMock()
        runner = _make_runner(classifier, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)
        assert not os.path.exists(runner._manifest_path())


class TestResumeIdentity:
    def test_matching_metadata_no_error(self, tmp_path: Path) -> None:
        classifier = _ClassifierWithDescribe(_BACKEND_META)
        runner = _make_runner(classifier, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)
        runner._verify_resume_identity()

    def test_changed_checksum_raises(self, tmp_path: Path) -> None:
        classifier = _ClassifierWithDescribe(_BACKEND_META)
        runner = _make_runner(classifier, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)

        changed = _ClassifierWithDescribe({**_BACKEND_META, "model_checksum": "different"})
        runner2 = _make_runner(changed, tmp_path)
        with pytest.raises(ValueError, match="identity mismatch"):
            runner2._verify_resume_identity()

    def test_changed_dataset_fingerprint_raises(self, tmp_path: Path) -> None:
        classifier = _ClassifierWithDescribe(_BACKEND_META)
        runner = _make_runner(classifier, tmp_path)
        runner._write_manifest(dataset_fingerprint="aaa")

        runner2 = _make_runner(classifier, tmp_path)
        with pytest.raises(ValueError, match="dataset_fingerprint"):
            runner2._verify_resume_identity(dataset_fingerprint="bbb")

    def test_dataset_fingerprint_hashes_file_bytes(self, tmp_path: Path) -> None:
        import hashlib

        dataset = tmp_path / "dataset.jsonl"
        dataset.write_bytes(b"hello")
        classifier = _ClassifierWithDescribe(_BACKEND_META)
        runner = _make_runner(classifier, tmp_path)
        runner._dataset_artifact = str(dataset)
        assert runner._dataset_fingerprint() == hashlib.sha256(b"hello").hexdigest()
