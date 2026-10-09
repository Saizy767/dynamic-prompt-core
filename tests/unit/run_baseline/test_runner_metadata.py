"""Tests for BaselineRunner run-metadata manifest and resume identity."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamic_prompt_core.application.services.classification_policy import (
    ArgmaxClassificationPolicy,
)
from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
    BaselineRunner,
    RunnerConfig,
)
from dynamic_prompt_core.domain.models.candidate import Candidate


class _ScorerWithDescribe:
    """Stub scorer carrying a ``describe()`` dict."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.score = AsyncMock()
        self._metadata = metadata

    def describe(self) -> dict[str, Any]:
        return dict(self._metadata)


def _make_runner(scorer: Any, tmp_path: Path) -> BaselineRunner:
    config = RunnerConfig(output_dir=str(tmp_path))
    runner = BaselineRunner(
        task=MagicMock(),
        config=config,
        split="dev",
        scorer=scorer,
        policy=ArgmaxClassificationPolicy(),
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
        scorer = _ScorerWithDescribe(_BACKEND_META)
        runner = _make_runner(scorer, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)
        with open(runner._manifest_path()) as f:
            data = json.load(f)
        assert data["backend"] == "gguf"
        assert data["model_checksum"] == "abc"
        assert data["run_id"] == "test-run"

    def test_no_manifest_without_describe(self, tmp_path: Path) -> None:
        scorer = AsyncMock()
        runner = _make_runner(scorer, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)
        assert not os.path.exists(runner._manifest_path())


class TestResumeIdentity:
    def test_matching_metadata_no_error(self, tmp_path: Path) -> None:
        scorer = _ScorerWithDescribe(_BACKEND_META)
        runner = _make_runner(scorer, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)
        runner._verify_resume_identity()

    def test_changed_checksum_raises(self, tmp_path: Path) -> None:
        scorer = _ScorerWithDescribe(_BACKEND_META)
        runner = _make_runner(scorer, tmp_path)
        runner._write_manifest(dataset_fingerprint=None)

        changed = _ScorerWithDescribe({**_BACKEND_META, "model_checksum": "different"})
        runner2 = _make_runner(changed, tmp_path)
        with pytest.raises(ValueError, match="identity mismatch"):
            runner2._verify_resume_identity()
