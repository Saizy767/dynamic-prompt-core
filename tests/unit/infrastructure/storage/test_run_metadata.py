"""Unit tests for run metadata recording and resume identity verification."""
from __future__ import annotations

import json

import pytest

from dynamic_prompt_core.infrastructure.storage.run_metadata import (
    RunMetadata,
    build_run_metadata,
    load_manifest,
    verify_resume_identity,
    write_manifest,
)


def _metadata(**overrides: object) -> RunMetadata:
    defaults: dict[str, object] = {
        "run_id": "run-1",
        "prompt_version": "classify-v0",
        "prompt_sha256": "abc123",
        "split": "dev",
        "dataset_fingerprint": "ds-hash",
        "backend": "gguf",
        "model_checksum": "model-hash",
    }
    defaults.update(overrides)
    return RunMetadata(**defaults)


class TestRunMetadataSerialization:
    def test_to_dict_is_json_serializable(self) -> None:
        meta = _metadata(inference_settings={"n_ctx": 2048})
        json.dumps(meta.to_dict())

    def test_round_trip(self) -> None:
        meta = _metadata(inference_settings={"n_ctx": 2048})
        restored = RunMetadata.from_dict(meta.to_dict())
        assert restored == meta

    def test_from_dict_ignores_unknown_keys(self) -> None:
        raw = _metadata().to_dict()
        raw["unknown_field"] = "x"
        restored = RunMetadata.from_dict(raw)
        assert restored.run_id == "run-1"


class TestManifestWriteLoad:
    def test_write_and_load_round_trip(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "manifest.json"
        meta = _metadata()
        write_manifest(str(path), meta)
        loaded = load_manifest(str(path))
        assert loaded == meta


class TestBuildRunMetadata:
    def test_builds_from_backend_describe(self) -> None:
        meta = build_run_metadata(
            run_id="run-1",
            prompt_version="v0",
            backend_metadata={
                "backend": "gguf",
                "model_path": "m.gguf",
                "model_checksum": "hash",
                "n_vocab": 32000,
                "n_ctx": 2048,
                "seed": 42,
            },
            dataset_fingerprint="ds-hash",
        )
        assert meta.backend == "gguf"
        assert meta.model_checksum == "hash"
        assert meta.n_vocab == 32000
        assert meta.inference_settings["n_ctx"] == 2048
        assert meta.inference_settings["seed"] == 42


class TestResumeIdentityVerification:
    def test_matching_metadata_passes(self) -> None:
        recorded = _metadata()
        current = _metadata()
        result = verify_resume_identity(recorded, current)
        assert result.matches
        assert not result.should_reject

    def test_changed_model_checksum_rejects(self) -> None:
        recorded = _metadata()
        current = _metadata(model_checksum="different")
        result = verify_resume_identity(recorded, current)
        assert not result.matches
        assert "model_checksum" in result.mismatched_semantic_fields

    def test_changed_dataset_fingerprint_rejects(self) -> None:
        recorded = _metadata()
        current = _metadata(dataset_fingerprint="different")
        result = verify_resume_identity(recorded, current)
        assert not result.matches
        assert "dataset_fingerprint" in result.mismatched_semantic_fields

    def test_changed_backend_rejects(self) -> None:
        recorded = _metadata()
        current = _metadata(backend="huggingface")
        result = verify_resume_identity(recorded, current)
        assert not result.matches
        assert "backend" in result.mismatched_semantic_fields

    def test_non_semantic_change_does_not_block(self) -> None:
        recorded = _metadata()
        current = _metadata(timestamp="2025-01-01", run_id="run-2")
        result = verify_resume_identity(recorded, current)
        assert result.matches

    def test_none_semantic_field_not_compared(self) -> None:
        recorded = _metadata(prompt_sha256=None)
        current = _metadata(prompt_sha256="new-sha")
        result = verify_resume_identity(recorded, current)
        assert result.matches
