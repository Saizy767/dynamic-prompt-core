# Proposal

## Why

Experiment results are the artifacts used to compare candidates, evaluate prompt
quality, and decide when to stop the optimization cycle. The current
`FileRunRepository` writes them by opening the target file in `"w"` mode and
streaming JSONL lines directly — no temp file, no `fsync`, no atomic replace, no
status tracking, no idempotency check, no concurrency protection, and no
recovery mechanism. A crash or exception mid-write leaves a truncated file that
`load_results` cannot parse. A re-run silently overwrites prior results. There
is no way to distinguish a completed run from an interrupted one. These defects
can produce incorrect metrics, unreproducible experiments, and erroneous
stop decisions.

## What Changes

- Harden `save_results` to write atomically: serialize the full JSONL payload,
  write to a unique temp file in the same directory, `flush` + `os.fsync`,
  read back and validate every line, then `os.replace`. On any failure the
  target file is left untouched.
- Harden `load_results` to detect a corrupted last line (partial write) and
  raise an explicit error rather than silently dropping or accepting bad data.
  Correct records before the corruption boundary remain readable.
- Add a run-status lifecycle (`running`, `completed`, `failed`, `interrupted`)
  stored in a per-run manifest. `completed` is set only after all required
  result data is validated and persisted. A save failure never produces
  `completed`.
- Add idempotency: re-saving the same `run_id` with identical data is a no-op
  (returns the existing path). Re-saving with different data raises a conflict
  error instead of silently overwriting.
- Add a per-repository `threading.RLock` protecting the full read-modify-write
  cycle on `save_results` and status updates.
- Add a recovery method that scans for runs left in `running` status from a
  previous process, marks them `interrupted`, and reports diagnostics — without
  deleting or modifying result files.
- Harden `write_manifest` / `load_manifest` in `run_metadata.py` with the same
  atomic-write pattern.
- Add a `schema_version` field to result documents if not already present.
- Keep the public `RunRepository` protocol signatures (`save_results`,
  `load_results`) backward compatible. New status/recovery methods are additions,
  not signature changes. Existing JSONL files continue to load.

## Capabilities

### New Capabilities
- `run-repository`: Owns crash-safe persistence of experiment results — atomic
  JSONL writes, run-status lifecycle, idempotent saves, concurrent-write
  protection, corrupted-file detection, and recovery of interrupted runs.

### Modified Capabilities
<!-- None — no existing spec covers the run repository. The change introduces
     the capability spec for the first time. -->

## Impact

- **Modified code**:
  `src/dynamic_prompt_core/infrastructure/storage/run_repository.py` —
  `FileRunRepository.save_results` and `load_results` are rewritten for atomic
  writes and corruption detection; new methods for status management and
  recovery are added.
  `src/dynamic_prompt_core/infrastructure/storage/run_metadata.py` —
  `write_manifest` and `load_manifest` are hardened with atomic writes.
  `src/dynamic_prompt_core/application/ports/outbound/run_repository.py` —
  the `RunRepository` protocol gains optional status/recovery method signatures.
- **Public API**: `save_results` and `load_results` signatures unchanged. New
  methods (`set_status`, `get_status`, `recover_interrupted`) are additions.
- **On-disk format**: JSONL result files unchanged. New per-run status manifest
  files (`<run_id>.status.json`) are introduced alongside result files.
  `schema_version` is added to result documents if absent.
- **Dependencies**: no new third-party dependencies. Standard library only
  (`threading`, `os`, `json`, `itertools`, `dataclasses`).
- **Tests**: new test module `tests/unit/infrastructure/storage/test_run_repository.py`
  covering atomic writes, corruption detection, status lifecycle, idempotency,
  concurrency, and recovery. Existing tests must continue to pass.
- **Upstream**: `run_cycle/state.py` (`dump_state`), `run_cycle/report.py`, and
  CLI callers benefit from stronger guarantees without API changes.
- **Downstream**: stop-criteria evaluation and version comparison consume
  results via `load_results` — they gain corruption detection and never silently
  read partial data.
