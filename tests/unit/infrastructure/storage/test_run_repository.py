"""Unit tests for the crash-safe FileRunRepository."""

from __future__ import annotations

import json
import os
import threading
from typing import Any
from unittest.mock import patch

import pytest

from dynamic_prompt_core.infrastructure.storage.run_repository import (
    FileRunRepository,
    RunCorruptedError,
    RunRepositoryError,
    RunValidationError,
    RunWriteError,
)


def _make_results(n: int = 3) -> list[dict[str, Any]]:
    return [{"sample_id": i, "label": f"label-{i}", "score": 0.5 * i} for i in range(n)]


@pytest.fixture
def repo() -> FileRunRepository:
    return FileRunRepository()


@pytest.fixture
def results_path(tmp_path: pytest.TempPathFactory) -> str:
    return str(tmp_path / "results.jsonl")


@pytest.fixture
def results_dir(tmp_path: pytest.TempPathFactory) -> str:
    return str(tmp_path)


class TestHappyPath:
    def test_save_and_load_round_trip(self, repo: FileRunRepository, results_path: str) -> None:
        results = _make_results()
        path = repo.save_results(results, "run-1", results_path)
        assert path == results_path
        loaded = repo.load_results(results_path)
        assert len(loaded) == 3
        assert loaded[0]["sample_id"] == 0
        assert loaded[2]["score"] == 1.0

    def test_save_empty_results(self, repo: FileRunRepository, results_path: str) -> None:
        repo.save_results([], "run-empty", results_path)
        loaded = repo.load_results(results_path)
        assert loaded == []

    def test_save_creates_parent_dir(
        self, repo: FileRunRepository, tmp_path: pytest.TempPathFactory
    ) -> None:
        path = str(tmp_path / "sub" / "deep" / "results.jsonl")
        repo.save_results(_make_results(), "run-1", path)
        assert os.path.exists(path)


class TestWriteFailurePreservesState:
    def test_replace_failure_preserves_target(
        self, repo: FileRunRepository, results_path: str
    ) -> None:
        repo.save_results(_make_results(), "run-1", results_path)
        original = open(results_path, encoding="utf-8").read()

        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(RunRepositoryError):
                repo.save_results(_make_results(5), "run-2", results_path)

        assert open(results_path, encoding="utf-8").read() == original

    def test_write_failure_preserves_target(
        self, repo: FileRunRepository, results_path: str
    ) -> None:
        repo.save_results(_make_results(), "run-1", results_path)
        original = open(results_path, encoding="utf-8").read()

        real_open = open

        def failing_open(path: Any, *args: Any, **kwargs: Any) -> Any:
            if "w" in str(args) and ".tmp" in str(path):
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            with pytest.raises(RunRepositoryError):
                repo.save_results(_make_results(5), "run-2", results_path)

        assert open(results_path, encoding="utf-8").read() == original

    def test_no_partial_file_after_error(self, repo: FileRunRepository, results_path: str) -> None:
        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(RunRepositoryError):
                repo.save_results(_make_results(), "run-1", results_path)

        if os.path.exists(results_path):
            content = open(results_path, encoding="utf-8").read()
            json.loads(content.strip().split("\n")[0]) if content.strip() else None


class TestCorruptedFileDetection:
    def test_corrupted_last_line_raises(self, repo: FileRunRepository, results_path: str) -> None:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write(json.dumps({"b": 2}) + "\n")
            f.write("{truncated")

        with pytest.raises(RunCorruptedError) as exc_info:
            repo.load_results(results_path)
        assert "line 3" in str(exc_info.value)

    def test_absent_file_raises_corrupted(self, repo: FileRunRepository, results_path: str) -> None:
        with pytest.raises(RunCorruptedError) as exc_info:
            repo.load_results(results_path)
        assert "not found" in str(exc_info.value)

    def test_load_safe_returns_valid_before_corruption(
        self, repo: FileRunRepository, results_path: str
    ) -> None:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write(json.dumps({"b": 2}) + "\n")
            f.write("{bad")

        records, info = repo.load_results_safe(results_path)
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert info is not None
        assert "line 3" in info

    def test_load_safe_clean_file(self, repo: FileRunRepository, results_path: str) -> None:
        repo.save_results(_make_results(), "run-1", results_path)
        records, info = repo.load_results_safe(results_path)
        assert len(records) == 3
        assert info is None


class TestErrorHierarchy:
    def test_subclasses(self) -> None:
        assert issubclass(RunValidationError, RunRepositoryError)
        assert issubclass(RunWriteError, RunRepositoryError)
        assert issubclass(RunCorruptedError, RunRepositoryError)
        assert issubclass(RunRepositoryError, ValueError)


class TestSchemaVersion:
    def test_schema_version_added_on_save(self, repo: FileRunRepository, results_path: str) -> None:
        results = [{"sample_id": 0, "label": "x"}]
        repo.save_results(results, "run-1", results_path)
        loaded = repo.load_results(results_path)
        assert loaded[0]["schema_version"] == 1

    def test_schema_version_not_overwritten(
        self, repo: FileRunRepository, results_path: str
    ) -> None:
        results = [{"sample_id": 0, "schema_version": 99}]
        repo.save_results(results, "run-1", results_path)
        loaded = repo.load_results(results_path)
        assert loaded[0]["schema_version"] == 99


class TestConcurrency:
    def test_concurrent_saves_different_files(
        self, repo: FileRunRepository, results_dir: str
    ) -> None:
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                path = os.path.join(results_dir, f"results_{thread_id}.jsonl")
                repo.save_results(_make_results(), f"run-{thread_id}", path)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for i in range(4):
            path = os.path.join(results_dir, f"results_{i}.jsonl")
            loaded = repo.load_results(path)
            assert len(loaded) == 3

    def test_concurrent_read_during_save(self, repo: FileRunRepository, results_path: str) -> None:
        repo.save_results(_make_results(), "run-1", results_path)
        stop = threading.Event()
        observed: list[int] = []

        def reader() -> None:
            while not stop.is_set():
                try:
                    records = repo.load_results(results_path)
                    observed.append(len(records))
                except RunCorruptedError:
                    pass

        def writer() -> None:
            for i in range(10):
                repo.save_results(_make_results(i + 1), f"run-w{i}", results_path)
            stop.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        writer_thread.join()
        reader_thread.join(timeout=2)

        assert len(observed) > 0


class TestRegression:
    def test_old_jsonl_loads_correctly(self, repo: FileRunRepository, results_path: str) -> None:
        with open(results_path, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps({"sample_id": i, "label": f"l{i}"}) + "\n")

        loaded = repo.load_results(results_path)
        assert len(loaded) == 3
        assert loaded[0]["sample_id"] == 0


class TestManifestHardening:
    def test_manifest_write_and_load_round_trip(self, tmp_path: pytest.TempPathFactory) -> None:
        from dynamic_prompt_core.infrastructure.storage.run_metadata import (
            RunMetadata,
            load_manifest,
            write_manifest,
        )

        meta = RunMetadata(
            run_id="run-1",
            prompt_version="v0",
            backend="gguf",
        )
        path = str(tmp_path / "manifest.json")
        write_manifest(path, meta)
        loaded = load_manifest(path)
        assert loaded == meta

    def test_failed_manifest_write_preserves_existing(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from dynamic_prompt_core.infrastructure.storage.run_metadata import (
            RunMetadata,
            RunMetadataError,
            load_manifest,
            write_manifest,
        )

        meta1 = RunMetadata(run_id="run-1", prompt_version="v0")
        path = str(tmp_path / "manifest.json")
        write_manifest(path, meta1)
        original = open(path, encoding="utf-8").read()

        meta2 = RunMetadata(run_id="run-2", prompt_version="v1")
        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(RunMetadataError):
                write_manifest(path, meta2)

        assert open(path, encoding="utf-8").read() == original
        assert load_manifest(path).run_id == "run-1"

    def test_corrupted_manifest_raises_error(self, tmp_path: pytest.TempPathFactory) -> None:
        from dynamic_prompt_core.infrastructure.storage.run_metadata import (
            RunMetadataError,
            load_manifest,
        )

        path = str(tmp_path / "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{corrupted")

        with pytest.raises(RunMetadataError):
            load_manifest(path)
