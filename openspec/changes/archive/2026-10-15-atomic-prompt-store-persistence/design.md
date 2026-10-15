# Design

## Context

`PromptStore` (`src/dynamic_prompt_core/infrastructure/storage/prompt_store.py`)
is a JSON-backed prompt-version store. It holds all versions in an in-memory
`self._versions` dict and persists them to a single JSON file via `_flush`. The
current write path has several correctness defects:

1. **Shared temp file** — `_flush` names the temp file
   `.{basename}.tmp.{os.getpid()}`. Two concurrent saves in the same process
   share the same temp path, causing write contention and potential corruption.
2. **Memory before disk** — `save`, `activate`, `_set_field`, `prune`, and
   `_seed_baseline` all mutate `self._versions` *then* call `_flush`. If `_flush`
   raises, memory has the new state but disk has the old state.
3. **No synchronization** — no lock protects the read-modify-write cycle.
   Concurrent saves can compute the same `new_version`, lose updates, or
   interleave flag changes.
4. **No fsync, no read-back** — `_flush` writes the temp file and calls
   `os.replace` without `flush`/`fsync` or validating the temp file first. A
   crash can leave a truncated temp or a partially flushed main file.
5. **Corruption policy** — `load` raises `PromptStoreError` on `JSONDecodeError`
   and structural problems, which is correct, but there are no tests enforcing
   this and no explicit policy that corruption must not silently become empty.

The store is consumed by `stage3-cycle-orchestrator` (via CLI) and
`cross_task_transfer`. The public API and on-disk format must remain compatible.

## Goals / Non-Goals

**Goals:**
- Make every save a true transaction: snapshot → unique temp → fsync →
  read-back validate → atomic replace → publish in memory.
- Guarantee memory/disk agreement after any confirmed error.
- Serialize concurrent mutating operations within a single process.
- Enforce and test corruption rejection on load.
- Add the missing dedicated test module for `PromptStore`.

**Non-Goals:**
- Inter-process locking or distributed locking (single-process `RLock` only).
  The design documents the boundary; a future change can add cross-process
  safety if the architecture requires it.
- Migrating from JSON to SQL or another storage engine.
- Changing the public API signatures or the on-disk JSON format.
- Changing the version-id format (monotonic integers remain).
- Automatic recovery of corrupted files from unconfirmed sources.
- Rewriting the experiment-results storage system.

## Decisions

### D1: Unique temp-file naming via PID + monotonic counter

**Choice:** Replace `.{basename}.tmp.{pid}` with
`.{basename}.tmp.{pid}.{counter}` where `counter` is a per-store
`itertools.count()` incremented on every `_flush` call. The temp file is created
in the same directory as the target (required for `os.replace` atomicity on the
same filesystem).

**Rationale:** PID alone is unique across processes but not across operations
within a process. A monotonic counter per store instance guarantees uniqueness
for every write in the same process. Keeping the temp file in the same directory
preserves `os.replace` atomicity guarantees (rename within a filesystem is
atomic; cross-filesystem rename is not).

**Alternatives considered:**
- `uuid.uuid4()` in the name — equally unique but adds entropy overhead and
  longer names; a counter is simpler and deterministic.
- `tempfile.NamedTemporaryFile(dir=...)` — creates the file atomically with
  `O_CREAT|O_EXCL` but returns a file descriptor that complicates the
  write-validate-replace flow; a named path is easier to reason about.

### D2: Per-store `threading.RLock` around the full read-modify-write cycle

**Choice:** Add `self._lock = threading.RLock()` in `__init__`. Every mutating
method (`save`, `activate`, `_set_field`, `prune`, `_seed_baseline`) acquires the
lock for its entire body. Read methods (`get`, `get_active`, `list_versions`,
`lineage`) also acquire the lock to prevent observing a half-updated dict during
a concurrent save.

**Rationale:** `RLock` (not `Lock`) because `save` calls `prune`, and
`_seed_baseline` is called from `init_store` — re-entrancy is needed. The lock
must cover the *entire* cycle, not just `os.replace`, because version-number
allocation and flag mutation must be atomic with respect to the persistence.

**Alternatives considered:**
- `Lock` (non-reentrant) — would deadlock on `save` → `prune` unless we
  restructured the call graph; `RLock` is simpler.
- `multiprocessing.Lock` — addresses inter-process safety but is out of scope
  and adds platform-specific complexity (file-based locks, cleanup).
- Optimistic concurrency control (version counter check before replace) —
  viable for inter-process but unnecessary within a single process where a
  mutex suffices.

### D3: Snapshot-then-persist ordering (disk first, memory second)

**Choice:** Each mutating method builds the *next* state as a separate dict
(e.g., `next_versions = dict(self._versions)` with the intended mutation
applied), validates it, persists it via the atomic write, and only then assigns
`self._versions = next_versions`. On any failure before the assignment, the
published `self._versions` is untouched.

**Rationale:** This directly satisfies the transactional-consistency
requirement. The current code mutates `self._versions` in place then flushes,
which diverges on failure. Building a separate next-state dict is a shallow copy
of the top-level dict; records themselves are only replaced, not mutated in
place, so aliasing is safe. For `activate`, the flag changes are applied to
copied records in the snapshot, not the live records.

**Alternatives considered:**
- Mutate-then-flush-then-rollback-on-error — requires saving a deep copy for
  rollback on every operation, which is more complex and error-prone than
  building the next state separately.
- Copy-on-write with immutable records — cleaner but a larger refactor that
  changes the data model; out of scope for a minimal fix.

### D4: fsync + read-back validation before `os.replace`

**Choice:** The atomic write routine:
1. Serialize the snapshot to a JSON string.
2. Open the unique temp file for writing, write the payload, `f.flush()`,
   `os.fsync(f.fileno())`, close it.
3. Re-open the temp file, `json.load` it, and run structural validation
   (top-level dict, `versions` dict, each record's required fields and hash).
4. `os.replace(temp, target)`.
5. Best-effort `os.fsync` on the parent directory's file descriptor (wrapped in
   `try/except OSError` — directory fsync is not supported on all platforms).
6. `finally`: if the temp file still exists (replace failed or wasn't reached),
   remove it.

**Rationale:** `flush` pushes Python's buffer to the OS; `fsync` forces the OS
to write to disk. Without both, a power loss can leave the temp file empty even
after a successful `write`. Read-back validation catches encoding errors and
truncation before the replace makes them visible. Directory fsync ensures the
rename is durable after a power loss; it's best-effort because Windows and some
filesystems don't support it.

**Alternatives considered:**
- Skip read-back validation — faster but defeats the purpose; a serialization
  bug or encoding issue would silently produce a corrupt main file.
- Skip directory fsync — simpler but loses the crash-durability guarantee for
  the rename itself. Best-effort with fallback is strictly better.

### D5: Explicit error hierarchy

**Choice:** Keep `PromptStoreError` as the base. Add two subclasses:
`PromptStoreValidationError(PromptStoreError)` for validation failures (missing
fields, hash mismatch, non-serializable) and `PromptStoreWriteError(
PromptStoreError)` for disk I/O failures (write, fsync, replace). `load` raises
`PromptStoreError` (or a `PromptStoreCorruptedError` subclass) for corruption.
This lets callers distinguish validation from write failures per the spec.

**Rationale:** The spec requires distinguishing validation failures from write
failures. Subclassing preserves backward compatibility — existing
`except PromptStoreError` handlers still catch everything.

**Alternatives considered:**
- Single error type with a `kind` field — less Pythonic, harder to catch
  selectively.
- Separate exception hierarchy not under `PromptStoreError` — breaks backward
  compatibility for callers that catch `PromptStoreError`.

### D6: `atomic_writes=False` path retained but documented as unsafe

**Choice:** Keep the `atomic_writes` config flag and its `False` path (direct
overwrite) for backward compatibility, but the `False` path also uses the lock
and snapshot ordering. Document in the config and docstrings that
`atomic_writes=False` is not safe for concurrent use or crash recovery.

**Rationale:** Removing the flag or its `False` path would be a breaking change
for any config that sets it. The hardening (lock, snapshot ordering) still
applies; only the atomicity guarantee is relaxed when the user explicitly opts
out.

## Risks / Trade-offs

- **[Directory fsync unsupported on some platforms]** → Best-effort with
  `try/except OSError`; document that crash durability of the rename is not
  guaranteed on platforms that reject directory fsync. The data itself is still
  durable (temp file fsync succeeds); only the rename's persistence is
  best-effort.
- **[Read-back validation doubles I/O per save]** → The store file is small
  (prompt versions, not experiment results). The overhead is negligible relative
  to the correctness guarantee. If it becomes a bottleneck, validation can be
  made configurable, but that is not needed now.
- **[Shallow copy of `_versions` may alias nested dicts]** → Mitigated by
  replacing records in the snapshot rather than mutating them in place. For
  `activate`, the affected records are shallow-copied before flag mutation. This
  is verified by tests.
- **[No inter-process safety]** → Documented as a non-goal. If the store is ever
  shared across processes, a follow-up change must add a file-based lock or
  optimistic concurrency control. The current consumers (CLI, orchestrator) are
  single-process.
- **[Orphaned temp files after crash]** → Temp files start with `.` and contain
  `.tmp.` in the name. A startup cleanup step can scan the directory for
  matching names and remove them, but this is optional — the main file is never
  a temp file, so orphans don't affect correctness, only disk hygiene.

## Migration Plan

1. **No data migration** — the on-disk format is unchanged. Old correct files
   load without modification.
2. **No API migration** — public signatures are unchanged. New error subclasses
   inherit from `PromptStoreError`, so existing `except` handlers are unaffected.
3. **Deploy** — the change is a drop-in replacement for `prompt_store.py` plus a
   new test module. No config changes required.
4. **Rollback** — revert to the previous `prompt_store.py`. Files written by the
   new code are in the same format, so the old code reads them without issue.

## Open Questions

- Should a startup cleanup pass remove orphaned `.*.tmp.*` files in the store
  directory? This is safe and improves disk hygiene but is not required for
  correctness. It can be added in a follow-up without changing specs or design.
