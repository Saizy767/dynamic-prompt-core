# Tasks

## 1. Test scaffolding and characterization tests

- [ ] 1.1 Create `tests/unit/infrastructure/storage/test_run_repository.py` with
  a `tmp_path`-based fixture that builds a `FileRunRepository` pointing at a
  temporary results directory. Verify the test module is collected by
  `pytest tests/unit/infrastructure/storage/test_run_repository.py --collect-only`.
- [ ] 1.2 Add characterization tests for the current happy path: save results,
  load them back, and verify round-trip integrity. Verify these pass against the
  current implementation
  (`pytest tests/unit/infrastructure/storage/test_run_repository.py -k happy`).
- [ ] 1.3 Add failing tests for the desired error behavior: write failure leaves
  target intact (UT-02/UT-03), `os.replace` failure leaves target intact
  (UT-04), corrupted last JSONL line raises explicit error (UT-08/UT-09), and
  duplicate `run_id` with different data raises conflict (UT-07). Use
  `monkeypatch` to inject failures. Verify these tests fail against the current
  implementation, confirming the defects exist.

## 2. Error hierarchy

- [ ] 2.1 Add `RunRepositoryError(ValueError)` base class and subclasses
  `RunValidationError`, `RunWriteError`, `RunCorruptedError`,
  `RunConflictError`, and `RunRecoveryError` to `run_repository.py`. Verify
  `ruff check` passes and all subclasses are confirmed by `issubclass` in a
  unit test.

## 3. Atomic JSONL write

- [ ] 3.1 Add a per-repository `threading.RLock` and monotonic write counter
  (`itertools.count()`) to `FileRunRepository.__init__`. Add a `_fsync_directory`
  helper (same pattern as `prompt_store.py`). Verify `ruff` and `mypy` pass.
- [ ] 3.2 Rewrite `save_results` to: serialize all records to a JSONL string,
  write to a unique temp file in the same directory, `flush` + `os.fsync`,
  read back and validate every line, `os.replace`, best-effort directory fsync,
  `finally` cleanup of temp file. Wrap in `self._lock`. Verify UT-01 (successful
  save) and UT-02/UT-03/UT-04 (failure preserves target) tests pass.
- [ ] 3.3 Add `schema_version` to result documents on save if absent. Verify a
  saved result file includes `schema_version` in a unit test.

## 4. Corrupted file detection on load

- [ ] 4.1 Rewrite `load_results` to detect a corrupted line (failed
  `json.loads`) and raise `RunCorruptedError` with the file path and line
  number. Distinguish absent file (`FileNotFoundError`) from corruption. Wrap
  in `self._lock`. Verify UT-08/UT-09 tests pass.
- [ ] 4.2 Add a `load_results_safe` method that returns
  `(valid_records, corruption_info)` — valid records up to the first corrupt
  line, plus diagnostics. Does not modify the file. Verify a test with valid
  records followed by a corrupt line returns the valid records and reports the
  corruption.

## 5. Run-status lifecycle

- [ ] 5.1 Add a `set_status(run_id, status, results_dir, error=None)` method
  that writes a `<run_id>.status.json` manifest atomically with fields `run_id`,
  `status`, `started_at`, `finished_at`, `error`. Add a `get_status(run_id,
  results_dir)` method that reads it. Wrap both in `self._lock`. Verify
  `set_status` then `get_status` round-trips in a unit test.
- [ ] 5.2 Wire status into `save_results`: before saving, set status to
  `running`; after successful save + validation, set status to `completed`; on
  failure, leave status as `running` (not `completed`). Verify a successful save
  produces `completed` status and a failed save does not (UT-05).
- [ ] 5.3 Add a `mark_failed(run_id, results_dir, error)` convenience method
  that transitions to `failed` with diagnostic info. Verify it sets `failed`
  status with the error message in a unit test.

## 6. Idempotent saves

- [ ] 6.1 Add idempotency check to `save_results`: if a status file for `run_id`
  already exists, compute the SHA-256 of the new payload and compare with the
  stored content hash. If identical, return the existing path (no-op). If
  different, raise `RunConflictError`. Store the content hash in the status
  manifest. Verify UT-06 (same data is no-op) and UT-07 (different data raises
  conflict) tests pass.

## 7. Recovery of interrupted runs

- [ ] 7.1 Add a `recover_interrupted(results_dir)` method that scans for
  `*.status.json` files, finds any with `status == "running"`, transitions them
  to `interrupted` with a diagnostic message, and returns a list of recovered
  run IDs. Does not touch result files. Wrap in `self._lock`. Verify UT-10:
  recovery detects an interrupted run, marks it, and does not delete files.
- [ ] 7.2 Add a test that recovery is idempotent: running it twice produces the
  same result with no duplicates. Verify the second call returns an empty list.

## 8. Manifest hardening in run_metadata.py

- [ ] 8.1 Apply the atomic-write pattern (temp file, fsync, read-back,
  os.replace) to `write_manifest` in `run_metadata.py`. Add a
  `RunMetadataError` wrapper for corruption on `load_manifest`. Verify
  `ruff check` passes and a test confirms `write_manifest` produces a valid
  file that `load_manifest` can read back.
- [ ] 8.2 Add a test that a failed manifest write leaves the existing manifest
  intact (same pattern as the run-repository write-failure tests).

## 9. Concurrency tests

- [ ] 9.1 Add concurrency test CT-01: two threads simultaneously save different
  results to the same repository. Verify both saves complete without corruption
  and both result files are valid JSONL.
- [ ] 9.2 Add concurrency test CT-02: a reader thread calls `load_results` in a
  loop while a writer thread saves. Verify the reader never observes a partial
  or invalid state.

## 10. Regression and compatibility tests

- [ ] 10.1 Add a backward-compatibility test: write a JSONL file in the original
  format (no `schema_version`, no status file) and load it with the new
  `load_results`. Verify all records are returned correctly.
- [ ] 10.2 Add a test for the full run lifecycle: save results, verify
  `completed` status, reload results, verify round-trip integrity. Verify all
  steps pass.
- [ ] 10.3 Verify all existing project tests still pass:
  `pytest tests/ -x -m "not gguf"` (excluding the pre-existing
  `test_stop_writes_artifact` failure).

## 11. Integration verification

- [ ] 11.1 Run `ruff check src/ tests/` and `ruff format --check src/ tests/`.
  Fix any lint or formatting issues in the changed files. Verify both commands
  exit zero.
- [ ] 11.2 Run `mypy src/dynamic_prompt_core/infrastructure/storage/run_repository.py
  src/dynamic_prompt_core/infrastructure/storage/run_metadata.py`. Fix any type
  errors. Verify the command exits zero.
- [ ] 11.3 Run the full run-repository test suite:
  `pytest tests/unit/infrastructure/storage/test_run_repository.py -v`. Verify
  all unit, concurrency, and regression tests pass.
- [ ] 11.4 Run `lint-imports` to verify architectural boundaries are not
  violated by the changes. Verify no new contract breaches compared to the
  pre-existing state.
