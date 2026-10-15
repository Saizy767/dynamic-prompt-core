# Proposal

## Why

The `PromptStore` persists prompt versions to a single JSON file, but its write
path is not truly atomic, consistent, or concurrency-safe. Today the store
mutates its in-memory `_versions` dict *before* flushing to disk, shares one
temp-file name (keyed only on PID) across concurrent operations in the same
process, performs no `fsync`, never validates the temp file before `os.replace`,
and has no synchronization around the read-modify-write cycle. The result is
that a failed save leaves memory and disk diverged, concurrent saves can corrupt
the file or silently lose updates, and a crash mid-write can leave a truncated
JSON. These are correctness defects in the system's source of truth for prompt
versions, so they must be fixed now before the store is relied on more heavily.

## What Changes

- Make every save a transaction: serialize from a snapshot, write to a
  *unique* temp file in the same directory, `flush` + `os.fsync`, read back and
  validate the temp file as JSON, then `os.replace` atomically. Best-effort
  `fsync` on the parent directory for crash durability.
- Stop mutating the published in-memory state before disk write succeeds. On
  any failure before `os.replace`, the previous confirmed state is preserved in
  memory and on disk; the error propagates to the caller without masking.
- Add a per-store `threading.RLock` that protects the entire read-modify-write
  cycle (read current state, compute new version, mutate, persist) for every
  mutating operation (`save`, `activate`, `_set_field`, `prune`, `_seed_baseline`).
- Use a unique temp-file name per write operation (PID + counter or UUID-based)
  so concurrent writes in one process never collide.
- Add explicit validation of the full document structure before serialization
  (top-level shape, required record fields, hash correctness, JSON
  serializability) and distinguish corruption from absence from access errors on
  load.
- On load, a corrupted JSON raises an explicit error and is **not** silently
  replaced with an empty store. An absent file still initializes empty per
  existing policy.
- Document the recovery policy: after a crash, the main file is either the last
  confirmed state or absent; orphaned temp files are cleaned up and never
  treated as valid store data.
- Keep the public API (`PromptStore`, `PromptStoreConfig`, `init_store`,
  `save`, `get`, `get_active`, `activate`, `list_versions`, `lineage`, `prune`)
  and the on-disk JSON format backward compatible. No new version-id format.
- Inter-process safety is out of scope for this change (single-process `RLock`
  only); the design documents the boundary and what a future cross-process lock
  would require.

## Capabilities

### New Capabilities
<!-- None — this change hardens an existing capability, it does not introduce one. -->

### Modified Capabilities
- `stage3-prompt-version-store`: Strengthens the persistence, concurrency, and
  recovery requirements. The store must save atomically via a unique temp file
  with fsync and read-back validation, keep memory and disk consistent on
  failure, synchronize concurrent mutating operations within a process, and
  reject corrupted files on load rather than silently reinitializing.

## Impact

- **Modified code**: `src/dynamic_prompt_core/infrastructure/storage/prompt_store.py`
  — the `_flush` method is replaced by an atomic write routine; `save`,
  `activate`, `_set_field`, `prune`, and `_seed_baseline` are re-ordered to
  snapshot-then-persist and wrapped in a per-store lock; `load` gains stricter
  corruption handling.
- **Public API**: unchanged signatures. `PromptStoreConfig.atomic_writes` remains
  but its `True` path is hardened; the `False` path is retained for
  backward compatibility but documented as unsafe for concurrent use.
- **On-disk format**: unchanged (`{"schema_version": 1, "versions": {...}}`).
  Old correct files load without migration.
- **Dependencies**: no new third-party dependencies. Uses only the standard
  library (`threading`, `os`, `json`, `hashlib`, `uuid` or counter).
- **Tests**: new unit tests for write-failure rollback, concurrency, and
  corruption; existing tests must continue to pass. There are currently no
  dedicated `PromptStore` tests, so a test module is added.
- **Upstream/downstream**: no changes to callers in this change; the
  `stage3-cycle-orchestrator` and `cross_task_transfer` consumers benefit from
  the stronger guarantees without API changes.
