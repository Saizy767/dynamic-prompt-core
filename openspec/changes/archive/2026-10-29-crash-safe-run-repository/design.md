# Design

## Context

`FileRunRepository` (`infrastructure/storage/run_repository.py`) is a 32-line
class that writes experiment results as JSONL. The current `save_results` opens
the target file in `"w"` mode and streams `json.dumps(rec) + "\n"` line by line.
The current `load_results` reads lines and calls `json.loads` on each. There is
no atomic write and no lock.

The `RunRepository` protocol (`application/ports/outbound/run_repository.py`)
defines `save_results(results, run_id, path) -> str` and
`load_results(path) -> list[dict]`. Callers include `run_cycle/state.py`
(`dump_state`), `run_cycle/report.py`, and the CLI composition root.

`RunMetadata` (`infrastructure/storage/run_metadata.py`) has
`write_manifest` / `load_manifest` with the same direct-overwrite defect.

The recently hardened `PromptStore` (`prompt_store.py`) already demonstrates the
atomic-write pattern (unique temp file, fsync, read-back, os.replace, RLock,
snapshot-then-persist) established in this project. This change applies the same
pattern to the run repository.

## Goals / Non-Goals

**Goals:**
- Make `save_results` atomic: temp file → fsync → read-back validate → replace.
- Detect corrupted JSONL on load and raise an explicit error.
- Add per-repository `RLock` for concurrent-write safety within a process.
- Harden `write_manifest` / `load_manifest` with the same atomic-write pattern.
- Keep public API and on-disk JSONL format backward compatible.

**Non-Goals:**
- Inter-process locking or distributed coordination (single-process `RLock`).
- Migrating to SQL or another storage engine.
- Automatically recalculating missing metrics or repairing corrupted data.
- Deleting corrupted result files without a backup.
- Rewriting the experiment runner or candidate evaluation logic.
- Adding message queues or distributed transactions.
- Run-status lifecycle, idempotent saves, or recovery of interrupted runs —
  deliberately dropped: the cycle orchestrator owns run identity, and re-runs
  write distinct timestamped artifacts, so status manifests and content-hash
  conflict checks added no value in practice.

## Decisions

### D1: Atomic JSONL write via temp file + read-back

**Choice:** `save_results` serializes all records to a single JSONL string,
writes it to a unique temp file (`.{basename}.tmp.{pid}.{counter}`) in the same
directory, `flush` + `os.fsync`, reads back and validates every line parses as
JSON, then `os.replace`. A `finally` block removes the temp file if it still
exists.

**Rationale:** JSONL is a line-delimited format; a partial write can leave a
truncated last line that `json.loads` rejects. Writing the full payload to a
temp file and validating before replace ensures the target is never partially
written. This mirrors the pattern already proven in `prompt_store.py`.

**Alternatives:** Write-then-rename without read-back (faster but a serialization
bug could produce a corrupt main file); per-line append with fsync after each
line (too slow for large result sets and still not atomic for the whole file).

### D2: Per-repository `threading.RLock`

**Choice:** Add `self._lock = threading.RLock()` to `FileRunRepository.__init__`.
Wrap `save_results`, `load_results`, and `load_results_safe` in the lock.

**Rationale:** Same pattern as `prompt_store.py`. The lock protects the full
read-modify-write cycle: validating records, serializing the payload, and
writing results.

### D3: Corrupted JSONL detection with recovery read

**Choice:** `load_results` reads lines and parses each with `json.loads`. If a
line fails to parse, raise `RunCorruptedError` with the file path and line
number. Add a `load_results_safe` method that returns a tuple of
`(valid_records, corruption_info)` — valid records up to the corruption
boundary, plus diagnostic info about the corrupt line. `load_results_safe` does
not truncate or modify the file.

**Rationale:** The spec requires that correct records before corruption remain
readable and that corruption is reported, not silently accepted. `load_results`
(strict) is backward compatible — it raises on any bad line, same as today but
with a clearer error. `load_results_safe` (recovery) is the new method for
callers that want to salvage partial data.

**Alternatives:** Silently skip bad lines (loses data silently — explicitly
forbidden by the spec); auto-truncate the file (modifies data without a backup
policy — forbidden).

### D4: Error hierarchy

**Choice:** Add `RunRepositoryError` as the base, with subclasses:
`RunValidationError`, `RunWriteError`, and `RunCorruptedError`. All inherit
from `RunRepositoryError` which inherits from `ValueError` (matching
`PromptStoreError`).

**Rationale:** The spec requires distinguishing error classes. Subclassing
preserves backward compatibility — callers catching `ValueError` or
`json.JSONDecodeError` still work, and new callers can catch specific types.

### D5: Manifest hardening in `run_metadata.py`

**Choice:** Apply the same atomic-write pattern (temp file, fsync, read-back,
os.replace) to `write_manifest`. `load_manifest` already raises on corrupt JSON
via `json.load`; add a wrapper error type for consistency.

**Rationale:** The manifest file has the same direct-overwrite defect. Fixing it
with the same pattern is minimal and consistent.

## Risks / Trade-offs

- **[Read-back validation doubles I/O]** → Result files are typically small
  (classification outputs, not model weights). Overhead is negligible. If it
  becomes a bottleneck for very large result sets, validation can be made
  configurable.
- **[No inter-process safety]** → Documented as a non-goal. The CLI and
  orchestrator are single-process. If multi-process support is needed later, a
  file-based lock can be added.
- **[Corrupted mid-file records]** → `load_results_safe` reads up to the first
  corrupt line. A corruption in the middle of the file limits recovery to
  records before it. This is documented in the spec — mid-file corruption is not
  automatically treated as a harmless truncation.

## Migration Plan

1. **No data migration** — existing JSONL files load without modification.
2. **No API migration** — `save_results` and `load_results` signatures are
   unchanged. New error subclasses inherit from `ValueError`, so existing
   `except` handlers are unaffected.
3. **Deploy** — drop-in replacement for `run_repository.py` and
   `run_metadata.py`, plus a new test module.
4. **Rollback** — revert to the previous files.

## Open Questions

- Should `load_results_safe` be added to the `RunRepository` protocol, or kept
  as a concrete method on `FileRunRepository`? For now, keep it on the concrete
  class and add it to the protocol if a second implementation ever needs it.
