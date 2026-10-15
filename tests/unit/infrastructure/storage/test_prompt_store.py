"""Unit tests for the JSON-backed prompt version store."""

from __future__ import annotations

import json
import os
import threading
from typing import Any
from unittest.mock import patch

import pytest

from dynamic_prompt_core.infrastructure.storage.prompt_store import (
    PromptStore,
    PromptStoreConfig,
    PromptStoreError,
    PromptStoreValidationError,
    PromptStoreWriteError,
    _sha256_hex,
    _validate_document,
)


def _make_prompt_version_dict(
    text: str = "You are a classifier. Output JSON.",
    rules: list[str] | None = None,
    source_candidates: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "rules": rules or [],
        "source_candidates": source_candidates or [],
        "created_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture
def store_config(tmp_path: pytest.TempPathFactory) -> PromptStoreConfig:
    store_path = tmp_path / "prompt_store.json"
    log_path = tmp_path / "prompt_store.jsonl"
    return PromptStoreConfig(
        store_path=str(store_path),
        log_path=str(log_path),
    )


@pytest.fixture
def store(store_config: PromptStoreConfig) -> PromptStore:
    return PromptStore(store_config)


@pytest.fixture
def store_with_v0(store: PromptStore) -> PromptStore:
    pv = _make_prompt_version_dict(text="Baseline prompt.")
    store.save(pv, reason="baseline")
    store.activate(0)
    return store


class TestHappyPath:
    def test_save_and_get(self, store: PromptStore) -> None:
        pv = _make_prompt_version_dict(text="v1 prompt")
        record = store.save(pv, reason="test")
        assert record["version"] == 0
        retrieved = store.get(0)
        assert retrieved["text"] == "v1 prompt"
        assert retrieved["hash"] == _sha256_hex("v1 prompt")

    def test_save_assigns_monotonic_versions(self, store: PromptStore) -> None:
        for i in range(3):
            rec = store.save(_make_prompt_version_dict(text=f"v{i}"))
            assert rec["version"] == i

    def test_get_active(self, store_with_v0: PromptStore) -> None:
        active = store_with_v0.get_active()
        assert active["version"] == 0
        assert active["active"] is True

    def test_activate_archives_previous(self, store_with_v0: PromptStore) -> None:
        pv = _make_prompt_version_dict(text="v1 prompt")
        store_with_v0.save(pv, reason="test")
        store_with_v0.activate(1)
        v0 = store_with_v0.get(0)
        assert v0["active"] is False
        assert v0["archived"] is True
        v1 = store_with_v0.get(1)
        assert v1["active"] is True

    def test_list_versions_sorted(self, store: PromptStore) -> None:
        for i in range(3):
            store.save(_make_prompt_version_dict(text=f"v{i}"))
        versions = store.list_versions()
        assert [v["version"] for v in versions] == [0, 1, 2]
        for v in versions:
            assert "text" not in v

    def test_lineage(self, store: PromptStore) -> None:
        r0 = store.save(_make_prompt_version_dict(text="v0"))
        r1 = store.save(
            _make_prompt_version_dict(text="v1"),
            base_version=r0["version"],
        )
        r2 = store.save(
            _make_prompt_version_dict(text="v2"),
            base_version=r1["version"],
        )
        lineage = store.lineage(r2["version"])
        assert [r["version"] for r in lineage] == [0, 1, 2]

    def test_reload_preserves_state(
        self, store_with_v0: PromptStore, store_config: PromptStoreConfig
    ) -> None:
        pv = _make_prompt_version_dict(text="v1")
        store_with_v0.save(pv, reason="test")
        store_with_v0.activate(1)
        reloaded = PromptStore(store_config)
        assert len(reloaded._versions) == 2
        assert reloaded.get(1)["active"] is True
        assert reloaded.get(0)["archived"] is True


class TestWriteFailurePreservesState:
    """UT-02/03/04/05: failures must not corrupt file or diverge memory."""

    def test_replace_failure_preserves_file_and_memory(
        self, store_with_v0: PromptStore, store_config: PromptStoreConfig
    ) -> None:
        original_file = json.loads(open(store_config.store_path, encoding="utf-8").read())
        original_memory = dict(store_with_v0._versions)

        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(PromptStoreError):
                store_with_v0.save(_make_prompt_version_dict(text="v1"))

        file_now = json.loads(open(store_config.store_path, encoding="utf-8").read())
        assert file_now == original_file
        assert store_with_v0._versions == original_memory

    def test_write_failure_preserves_file_and_memory(
        self, store_with_v0: PromptStore, store_config: PromptStoreConfig
    ) -> None:
        original_file = json.loads(open(store_config.store_path, encoding="utf-8").read())
        original_memory = dict(store_with_v0._versions)

        real_open = open

        def failing_open(path: Any, *args: Any, **kwargs: Any) -> Any:
            if "w" in str(args) and ".tmp" in str(path):
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            with pytest.raises(PromptStoreError):
                store_with_v0.save(_make_prompt_version_dict(text="v1"))

        file_now = json.loads(open(store_config.store_path, encoding="utf-8").read())
        assert file_now == original_file
        assert store_with_v0._versions == original_memory

    def test_memory_unchanged_on_disk_write_failure(self, store_with_v0: PromptStore) -> None:
        original_keys = set(store_with_v0._versions.keys())

        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(PromptStoreError):
                store_with_v0.save(_make_prompt_version_dict(text="v1"))

        assert set(store_with_v0._versions.keys()) == original_keys

    def test_corrupted_json_raises_without_silent_reinit(
        self, store_config: PromptStoreConfig
    ) -> None:
        store_path = store_config.store_path
        with open(store_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")

        with pytest.raises(PromptStoreError):
            PromptStore(store_config)

    def test_no_silent_success_on_write_error(self, store_with_v0: PromptStore) -> None:
        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(PromptStoreError):
                result = store_with_v0.save(_make_prompt_version_dict(text="v1"))
                assert result is not None


class TestValidation:
    """Tests for _validate_document and error hierarchy."""

    def test_error_subclasses(self) -> None:
        assert issubclass(PromptStoreValidationError, PromptStoreError)
        assert issubclass(PromptStoreWriteError, PromptStoreError)

    def test_valid_document_passes(self) -> None:
        record = {
            "version": 0,
            "text": "hello",
            "hash": _sha256_hex("hello"),
            "rules": [],
            "source_candidates": [],
            "base_version": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "active": True,
            "archived": False,
            "reason": "test",
        }
        _validate_document({"0": record})

    def test_missing_field_raises_validation_error(self) -> None:
        record = {"version": 0, "text": "hello"}
        with pytest.raises(PromptStoreValidationError):
            _validate_document({"0": record})

    def test_bad_hash_raises_validation_error(self) -> None:
        record = {
            "version": 0,
            "text": "hello",
            "hash": "wrong",
            "rules": [],
            "source_candidates": [],
            "base_version": None,
            "created_at": "*",
            "active": True,
            "archived": False,
            "reason": "test",
        }
        with pytest.raises(PromptStoreValidationError):
            _validate_document({"0": record})


class TestAtomicWrite:
    """Tests for unique temp files, fsync, and orphan cleanup."""

    def test_unique_temp_file_names(self, store_with_v0: PromptStore) -> None:
        captured_paths: list[str] = []
        real_replace = os.replace

        def spy_replace(src: str, dst: str) -> None:
            captured_paths.append(src)
            real_replace(src, dst)

        with patch("os.replace", side_effect=spy_replace):
            store_with_v0.save(_make_prompt_version_dict(text="v1"))
            store_with_v0.save(_make_prompt_version_dict(text="v2"))

        assert len(captured_paths) == 2
        assert captured_paths[0] != captured_paths[1]

    def test_fsync_called_on_save(self, store_with_v0: PromptStore) -> None:
        fsync_calls: list[int] = []
        real_fsync = os.fsync

        def spy_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            real_fsync(fd)

        with patch("os.fsync", side_effect=spy_fsync):
            store_with_v0.save(_make_prompt_version_dict(text="v1"))

        assert len(fsync_calls) > 0

    def test_orphaned_temp_file_does_not_affect_load(
        self, store_with_v0: PromptStore, store_config: PromptStoreConfig
    ) -> None:
        directory = os.path.dirname(store_config.store_path) or "."
        basename = os.path.basename(store_config.store_path)
        orphan = os.path.join(directory, f".{basename}.tmp.orphan")
        with open(orphan, "w", encoding="utf-8") as f:
            f.write("{partial")

        reloaded = PromptStore(store_config)
        assert len(reloaded._versions) == 1
        assert reloaded.get(0)["active"] is True


class TestConcurrency:
    """CT-01/02/03: concurrent operations on a single store."""

    def test_concurrent_saves_no_lost_updates(self, store: PromptStore) -> None:
        errors: list[Exception] = []

        def worker() -> None:
            for i in range(10):
                try:
                    store.save(
                        _make_prompt_version_dict(text=f"thread-{i}"),
                        reason="concurrent",
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        versions = store.list_versions()
        assert len(versions) == 20
        version_numbers = [v["version"] for v in versions]
        assert version_numbers == list(range(20))

    def test_concurrent_activate_one_active(self, store: PromptStore) -> None:
        for i in range(5):
            store.save(_make_prompt_version_dict(text=f"v{i}"))

        errors: list[Exception] = []

        def worker(version: int) -> None:
            try:
                store.activate(version)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        active = [v for v in store.list_versions() if v["active"]]
        assert len(active) == 1
        with open(store.config.store_path, encoding="utf-8") as f:
            json.load(f)

    def test_concurrent_read_during_save(self, store: PromptStore) -> None:
        stop = threading.Event()
        observed_counts: list[int] = []

        def reader() -> None:
            while not stop.is_set():
                versions = store.list_versions()
                observed_counts.append(len(versions))

        def writer() -> None:
            for i in range(20):
                store.save(_make_prompt_version_dict(text=f"v{i}"))
            stop.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        writer_thread.join()
        reader_thread.join(timeout=2)

        assert len(observed_counts) > 0
        for count in observed_counts:
            assert 0 <= count <= 20


class TestLoadIntegrity:
    """Tests for load behavior: absent, valid, and corrupted files."""

    def test_absent_file_initializes_empty(self, store_config: PromptStoreConfig) -> None:
        assert not os.path.exists(store_config.store_path)
        store = PromptStore(store_config)
        assert store._versions == {}

    def test_valid_file_loads_without_modification(self, store_config: PromptStoreConfig) -> None:
        record = {
            "version": 0,
            "text": "test prompt",
            "hash": _sha256_hex("test prompt"),
            "rules": [],
            "source_candidates": [],
            "base_version": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "active": True,
            "archived": False,
            "reason": "test",
        }
        with open(store_config.store_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "versions": {"0": record}}, f)

        store = PromptStore(store_config)
        assert store.get(0)["text"] == "test prompt"
        assert store.get(0)["active"] is True

    def test_invalid_top_level_raises(self, store_config: PromptStoreConfig) -> None:
        with open(store_config.store_path, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        with pytest.raises(PromptStoreError):
            PromptStore(store_config)

    def test_missing_versions_dict_raises(self, store_config: PromptStoreConfig) -> None:
        with open(store_config.store_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1}, f)
        with pytest.raises(PromptStoreError):
            PromptStore(store_config)

    def test_log_entries_contain_no_prompt_text(self, store_config: PromptStoreConfig) -> None:
        store = PromptStore(store_config)
        store.save(
            _make_prompt_version_dict(text="SECRET-PROMPT-CONTENT"),
            reason="test",
        )
        with open(store_config.log_path, encoding="utf-8") as f:
            log_content = f.read()
        assert "SECRET-PROMPT-CONTENT" not in log_content
        assert "save" in log_content


class TestRegression:
    """Backward compatibility, sequential history, and immutability."""

    def test_backward_compatibility_old_format(self, store_config: PromptStoreConfig) -> None:
        versions: dict[str, Any] = {}
        for i in range(3):
            text = f"v{i} prompt"
            versions[str(i)] = {
                "version": i,
                "text": text,
                "hash": _sha256_hex(text),
                "rules": [],
                "source_candidates": [],
                "base_version": i - 1 if i > 0 else None,
                "created_at": f"2026-01-0{i + 1}T00:00:00+00:00",
                "active": i == 2,
                "archived": i < 2,
                "reason": "test",
            }
        with open(store_config.store_path, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "versions": versions}, f)

        store = PromptStore(store_config)
        assert len(store._versions) == 3
        assert store.get(2)["active"] is True
        assert store.get(0)["archived"] is True
        assert store.get(1)["archived"] is True

    def test_sequential_version_history(self, store: PromptStore) -> None:
        texts: list[str] = []
        for i in range(6):
            text = f"version-{i}"
            texts.append(text)
            store.save(_make_prompt_version_dict(text=text))

        versions = store.list_versions()
        assert len(versions) == 6
        assert [v["version"] for v in versions] == [0, 1, 2, 3, 4, 5]
        for i, text in enumerate(texts):
            record = store.get(i)
            assert record["text"] == text
            assert record["hash"] == _sha256_hex(text)

    def test_activate_does_not_change_historical_versions(self, store: PromptStore) -> None:
        store.save(_make_prompt_version_dict(text="v0"))
        store.activate(0)
        store.save(_make_prompt_version_dict(text="v1"))
        store.activate(1)

        v0_before = store.get(0)
        v1_before = store.get(1)

        store.save(_make_prompt_version_dict(text="v2"))
        store.activate(2)

        v0_after = store.get(0)
        v1_after = store.get(1)
        assert v0_after["text"] == v0_before["text"]
        assert v0_after["hash"] == v0_before["hash"]
        assert v1_after["text"] == v1_before["text"]
        assert v1_after["hash"] == v1_before["hash"]
