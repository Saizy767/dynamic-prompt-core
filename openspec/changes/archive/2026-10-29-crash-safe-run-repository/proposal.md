# Proposal

## Why

Experiment results are the artifacts used to compare candidates, evaluate prompt
quality, and decide when to stop the optimization cycle. The original
`FileRunRepository` wrote them by opening the target file in `"w"` mode and
streaming JSONL lines directly — no temp file, no `fsync`, no atomic replace,
and no concurrency protection. A crash or exception mid-write leaves a
truncated file that `load_results` cannot parse. These defects can produce
incorrect metrics, unreproducible experiments, and erroneous stop decisions.

## What Changes

- Harden `save_results` to write atomically: serialize the full JSONL payload,
  write to a unique temp file in the same directory, `flush` + `os.fsync`,
  read back and validate every line, then `os.replace`. On any failure the
  target file is left untouched.
- Harden `load_results` to detect a corrupted line (partial write) and raise an
  explicit error rather than silently dropping or accepting bad data. Correct
  records before the corruption boundary remain readable via `load_results_safe`.
- Add a per-repository `threading.RLock` protecting the full read-modify-write
  cycle on `save_results` and loads.
- Harden `write_manifest` / `load_manifest` in `run_metadata.py` with the same
  atomic-write pattern.
- Add a `schema_version` field to result documents if not already present.
- Keep the public `RunRepository` protocol signatures (`save_results`,
  `load_results`) backward compatible. Existing JSONL files continue to load.

## Capabilities

### New Capabilities
- `run-repository`: Owns crash-safe persistence of experiment results — atomic
  JSONL writes, concurrent-write protection, and corrupted-file detection.

### Modified Capabilities
<!-- None — no existing spec covers the run repository. The change introduces
     the capability spec for the first time. -->

## Impact

- **Modified code**:
  `src/dynamic_prompt_core/infrastructure/storage/run_repository.py` —
  `FileRunRepository.save_results` and `load_results` are rewritten for atomic
  writes and corruption detection.
  `src/dynamic_prompt_core/infrastructure/storage/run_metadata.py` —
  `write_manifest` and `load_manifest` are hardened with atomic writes.
- **Public API**: `save_results` and `load_results` signatures unchanged.
  `load_results_safe` is an addition on the concrete class, not the protocol.
- **On-disk format**: JSONL result files unchanged. `schema_version` is added
  to result documents if absent.
- **Dependencies**: no new third-party dependencies. Standard library only
  (`threading`, `os`, `json`, `itertools`).
- **Tests**: `tests/unit/infrastructure/storage/test_run_repository.py` covers
  atomic writes, corruption detection, concurrency, backward compatibility,
  and manifest hardening.
- **Upstream**: `run_cycle/state.py` (`dump_state`), `run_cycle/report.py`, and
  CLI callers benefit from stronger guarantees without API changes.
- **Downstream**: stop-criteria evaluation and version comparison consume
  results via `load_results` — they gain corruption detection and never
  silently read partial data.
