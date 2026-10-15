# Tasks

## 1. Test scaffolding and characterization tests

- [x] 1.1 Create `tests/unit/infrastructure/storage/test_prompt_store.py` with a
  `tmp_path`-based fixture that builds a `PromptStoreConfig` pointing at a
  temporary store file and log file. Verify the test module is collected by
  `pytest tests/unit/infrastructure/storage/test_prompt_store.py --collect-only`.
- [x] 1.2 Add characterization tests for the current happy path: save a version,
  retrieve it by number, retrieve the active version, list versions, and
  reconstruct lineage. Verify these pass against the current implementation
  (`pytest tests/unit/infrastructure/storage/test_prompt_store.py -k happy`).
- [x] 1.3 Add failing tests for the desired error behavior: save failure
  preserves previous on-disk file (UT-02/UT-03), `os.replace` failure preserves
  old state (UT-04), memory unchanged on disk write failure (UT-05), and
  corrupted JSON on load raises without silent reinit (UT-06). Use `monkeypatch`
  to inject write/replace failures. Verify these tests fail against the current
  implementation, confirming the defects exist.

## 2. Error hierarchy and input validation

- [x] 2.1 Add `PromptStoreValidationError(PromptStoreError)` and
  `PromptStoreWriteError(PromptStoreError)` subclasses to `prompt_store.py`.
  Verify `ruff check src/dynamic_prompt_core/infrastructure/storage/prompt_store.py`
  passes and existing `except PromptStoreError` handlers still match (subclass
  relationship confirmed by `issubclass`).
- [x] 2.2 Extract a `_validate_document(versions: dict) -> None` function that
  checks the top-level structure, each record's required fields, hash
  correctness, and JSON serializability. Raise `PromptStoreValidationError` on
  failure. Verify with a unit test that a valid document passes and a document
  with a missing field, bad hash, or non-serializable value raises
  `PromptStoreValidationError`.
- [x] 2.3 Wire `_validate_document` into `save` so validation runs before any
  disk write. Verify the test from 1.3 for "validation error before write"
  passes and that no temp file is created on validation failure.

## 3. Atomic write routine

- [x] 3.1 Add a per-store monotonic counter (`self._write_counter =
  itertools.count()`) and replace the temp-file naming in `_flush` with
  `.{basename}.tmp.{pid}.{next(counter)}`. Verify two consecutive saves produce
  distinct temp-file names (test with `monkeypatch` on `os.replace` to capture
  the source path).
- [x] 3.2 Rewrite the `atomic_writes=True` path of `_flush` to: open temp file
  for writing, write payload, `f.flush()`, `os.fsync(f.fileno())`, close; re-open
  and `json.load` to validate; `os.replace(temp, path)`; best-effort
  `os.fsync` on parent directory fd in `try/except OSError`. Keep the `finally`
  cleanup that removes the temp file if it still exists. Verify the UT-01 test
  (successful save writes correct file) passes and that `fsync` is called (test
  with `monkeypatch` to spy on `os.fsync`).
- [x] 3.3 Add a test (UT-08) that simulates a crash by leaving an orphaned temp
  file in the store directory, then loads the store. Verify the main file is
  still valid and the orphaned temp file does not affect loading.

## 4. Snapshot-then-persist ordering

- [x] 4.1 Refactor `save` to build `next_versions = dict(self._versions)` with
  the new record added, validate it, persist via `_flush(next_versions)`, and
  only then assign `self._versions = next_versions`. On failure,
  `self._versions` is untouched. Verify UT-02/UT-03/UT-04/UT-05 tests from group
  1 now pass: the on-disk file and in-memory state both retain the previous
  state after an injected failure.
- [x] 4.2 Refactor `activate` to build a snapshot with the flag changes applied
  to shallow-copied records, persist, then publish. Verify the "activate failure
  preserves state" test passes and that a failed activate leaves both the
  on-disk file and in-memory active/archived flags unchanged.
- [x] 4.3 Refactor `_set_field` and `prune` to the same snapshot-then-persist
  pattern. Verify existing tests for prune (if any) and a new test for
  `_set_field` failure preserving state both pass.
- [x] 4.4 Refactor `_seed_baseline` to build the snapshot, persist, then
  publish. Verify `init_store` on an empty directory still seeds v0 correctly
  and that a failure during seeding leaves no partial file.

## 5. Concurrency lock

- [x] 5.1 Add `self._lock = threading.RLock()` in `PromptStore.__init__` and
  wrap every public mutating method (`save`, `activate`, `prune`) and
  `_set_field`, `_seed_baseline` in `with self._lock:`. Wrap read methods
  (`get`, `get_active`, `list_versions`, `lineage`) in the lock as well. Verify
  `ruff` and `mypy` pass on the modified file.
- [x] 5.2 Add concurrency test CT-01: two threads simultaneously call `save` 10
  times each on the same store. Verify all 20 versions are persisted with
  distinct, sequential version numbers and no `PromptStoreError` is raised.
- [x] 5.3 Add concurrency test CT-02: multiple threads concurrently `activate`
  different versions. Verify the final on-disk state has exactly one active
  version and the file is valid JSON.
- [x] 5.4 Add concurrency test CT-03: a reader thread calls `list_versions` in a
  loop while a writer thread saves. Verify the reader never observes a partial
  or invalid state (every observed list has consistent, complete records).

## 6. Load integrity and corruption handling

- [x] 6.1 Strengthen `load` to raise `PromptStoreError` (with file path and
  parse cause in the message, no prompt text) on `json.JSONDecodeError`, invalid
  top-level type, and missing/invalid `versions` dict. Verify UT-06 passes:
  corrupted JSON raises an explicit error and is not silently reinitialized as
  empty.
- [x] 6.2 Add a test that an absent store file initializes empty (existing
  policy) and a test that a valid file with correct structure loads without
  modification. Verify both pass.
- [x] 6.3 Add a test that log entries on load/save errors contain the operation
  type and file path but not the prompt text. Verify by inspecting the log file
  content after an induced error.

## 7. Regression and compatibility tests

- [x] 7.1 Add a backward-compatibility test: write a store file in the original
  format (`{"schema_version": 1, "versions": {...}}`) with multiple versions,
  then load it with the new implementation. Verify all versions, active flags,
  and archived flags are restored correctly.
- [x] 7.2 Add a test for sequential version history (UT-07): save versions v0
  through v5, then verify `list_versions` returns all six in order with correct
  metadata and that each historical version's content is unchanged after later
  saves.
- [x] 7.3 Add a test that updating the active prompt (activate) does not change
  the content of already-saved historical versions. Verify by comparing record
  `text` and `hash` before and after activation.
- [x] 7.4 Verify all existing project tests still pass:
  `pytest tests/ -x --timeout=60` (excluding `gguf`-marked tests if the optional
  dependency is unavailable: `pytest tests/ -x -m "not gguf"`).

## 8. Integration verification

- [x] 8.1 Run `ruff check src/ tests/` and `ruff format --check src/ tests/`.
  Fix any lint or formatting issues in the changed files. Verify both commands
  exit zero.
- [x] 8.2 Run `mypy src/dynamic_prompt_core/infrastructure/storage/prompt_store.py`.
  Fix any type errors. Verify the command exits zero.
- [x] 8.3 Run the full prompt-store test suite:
  `pytest tests/unit/infrastructure/storage/test_prompt_store.py -v`. Verify all
  unit, concurrency, and regression tests pass.
- [x] 8.4 Run `lint-imports` (import-linter) to verify architectural boundaries
  are not violated by the changes. Verify the command exits zero.
