# Tasks

## 1. Test scaffolding and characterization tests

- [x] 1.1 Create `tests/unit/infrastructure/storage/test_run_repository.py` with
  a `tmp_path`-based fixture that builds a `FileRunRepository` pointing at a
  temporary results directory. Verify the test module is collected by
  `pytest tests/unit/infrastructure/storage/test_run_repository.py --collect-only`.
- [x] 1.2 Add characterization tests for the current happy path: save results,
  load them back, and verify round-trip integrity. Verify these pass against the
  current implementation
  (`pytest tests/unit/infrastructure/storage/test_run_repository.py -k happy`).
- [x] 1.3 Add failing tests for the desired error behavior: write failure leaves
  target intact (UT-02/UT-03), `os.replace` failure leaves target intact
  (UT-04), corrupted last JSONL line raises explicit error (UT-08/UT-09). Use
  `monkeypatch` to inject failures. Verify these tests fail against the current
  implementation, confirming the defects exist.

## 2. Error hierarchy

- [x] 2.1 Add `RunRepositoryError(ValueError)` base class and subclasses
  `RunValidationError`, `RunWriteError`, and `RunCorruptedError` to
  `run_repository.py`. Verify `ruff check` passes and all subclasses are
  confirmed by `issubclass` in a unit test.

## 3. Atomic JSONL write

- [x] 3.1 Add a per-repository `threading.RLock` and monotonic write counter
  (`itertools.count()`) to `FileRunRepository.__init__`. Add a `_fsync_directory`
  helper (same pattern as `prompt_store.py`). Verify `ruff` and `mypy` pass.
- [x] 3.2 Rewrite `save_results` to: serialize all records to a JSONL string,
  write to a unique temp file in the same directory, `flush` + `os.fsync`,
  read back and validate every line, `os.replace`, best-effort directory fsync,
  `finally` cleanup of temp file. Wrap in `self._lock`. Verify UT-01 (successful
  save) and UT-02/UT-03/UT-04 (failure preserves target) tests pass.
- [x] 3.3 Add `schema_version` to result documents on save if absent. Verify a
  saved result file includes `schema_version` in a unit test.

## 4. Corrupted file detection on load

- [x] 4.1 Rewrite `load_results` to detect a corrupted line (failed
  `json.loads`) and raise `RunCorruptedError` with the file path and line
  number. Wrap in `self._lock`. Verify UT-08/UT-09 tests pass.
- [x] 4.2 Add a `load_results_safe` method that returns
  `(valid_records, corruption_info)` — valid records up to the first corrupt
  line, plus diagnostics. Does not modify the file. Verify a test with valid
  records followed by a corrupt line returns the valid records and reports the
  corruption.

## 5. Manifest hardening in run_metadata.py

- [x] 5.1 Apply the atomic-write pattern (temp file, fsync, read-back,
  os.replace) to `write_manifest` in `run_metadata.py`. Add a
  `RunMetadataError` wrapper for corruption on `load_manifest`. Verify
  `ruff check` passes and a test confirms `write_manifest` produces a valid
  file that `load_manifest` can read back.
- [x] 5.2 Add a test that a failed manifest write leaves the existing manifest
  intact (same pattern as the run-repository write-failure tests).

## 6. Concurrency tests

- [x] 6.1 Add concurrency test CT-01: two threads simultaneously save different
  results to the same repository. Verify both saves complete without corruption
  and both result files are valid JSONL.
- [x] 6.2 Add concurrency test CT-02: a reader thread calls `load_results` in a
  loop while a writer thread saves. Verify the reader never observes a partial
  or invalid state.

## 7. Regression and compatibility tests

- [x] 7.1 Add a backward-compatibility test: write a JSONL file in the original
  format (no `schema_version`) and load it with the new `load_results`. Verify
  all records are returned correctly.
- [x] 7.2 Verify all existing project tests still pass:
  `pytest tests/ -x -m "not gguf"`.

## 8. Integration verification

- [x] 8.1 Run `ruff check src/ tests/` and `ruff format --check src/ tests/`.
  Fix any lint or formatting issues in the changed files. Verify both commands
  exit zero.
- [x] 8.2 Run `mypy src/dynamic_prompt_core/infrastructure/storage/run_repository.py
  src/dynamic_prompt_core/infrastructure/storage/run_metadata.py`. Fix any type
  errors. Verify the command exits zero.
- [x] 8.3 Run the full run-repository test suite:
  `pytest tests/unit/infrastructure/storage/test_run_repository.py -v`. Verify
  all unit, concurrency, and regression tests pass.
- [x] 8.4 Run `lint-imports` to verify architectural boundaries are not
  violated by the changes. Verify no new contract breaches compared to the
  pre-existing state.
