# Design

## Context

The pipeline produces prompt versions via `prompt_composer.write_prompt_version`
(`prompt_composer.py:553`), which writes individual `prompt_v{version}_{run_id}_{timestamp}.json`
artifacts containing `version`, `text`, `hash`, `rules`, `source_candidates`,
`base_version`, `created_at`, and `metadata`. `version_comparator.py` loads these
via `load_new_version` and builds a `PromptArtifact` via `build_prompt_artifact`
(`version_comparator.py:103`). The cycle orchestrator tracks the active version
in-memory and in its state dump (`cycle_orchestrator.py`, design D11 of that
change), with no persistent index.

Every component follows the same pattern: a `*Config` dataclass with
`from_config()` reading a TOML section via `tomllib`, a `*Error(ValueError)`
exception class, artifact load/validate helpers, and an argparse CLI. The store
component mirrors this pattern but, unlike the per-step components, it owns a
long-lived mutable index file rather than emitting one-shot artifacts.

Key existing interfaces the store reuses:

- `PromptArtifact` / `CLASSIFICATION_PROMPT_V0` from `prompts.py` — the baseline
  v0 prompt, used to seed the store on first init.
- The prompt-version dict shape from `prompt_composer.write_prompt_version` —
  the store accepts this dict and normalizes it into a store record.
- `version_comparator.build_prompt_artifact` — downstream consumers build a
  `PromptArtifact` from a store record via this existing function.

## Goals / Non-Goals

**Goals:**
- Be the single durable source of truth for prompt versions and the active
  version, surviving across runs.
- Assign monotonic version numbers internally so callers never manage numbering.
- Enforce immutability of saved content so a version's text and lineage can never
  be silently rewritten.
- Reconstruct full version lineage by walking `base_version` links.
- Persist atomically so a crash mid-write never corrupts the store.
- Log every operation for auditability.

**Non-Goals:**
- Composition of prompt versions (handled by `stage3-prompt-composer-v2`).
- Version comparison and accept/rollback decisions (handled by
  `stage2-version-comparator`).
- Metrics computation.
- Rollback of activation — activation is append-only; reactivating an earlier
  version is a new activation, not an undo.
- Distributed or concurrent access to the store — single-process, single-thread.
- Auto-recovery from a corrupted store — fail hard with a clear error; recovery
  is a manual operator action.

## Decisions

### D1: Single-module component (`prompt_store.py`)
Mirrors `version_comparator.py` / `prompt_composer.py`: dataclass
`PromptStoreConfig.from_config`, `PromptStore` class wrapping the file,
`PromptStoreError(ValueError)` exception, `PromptVersionRecord` dataclass, and an
argparse CLI. The `PromptStore` class holds the in-memory records loaded from
disk and flushes on every mutation.

**Alternative**: split storage backend from the API into two modules. Rejected —
there is exactly one backend (a JSON file); a backend abstraction adds
indirection without a second implementation to justify it.

### D2: Store file format — single JSON object
The store file is one JSON object:

```json
{
  "schema_version": 1,
  "versions": {
    "0": { "version": 0, "text": "...", "hash": "...", "rules": [...],
           "source_candidates": [...], "base_version": null,
           "created_at": "...", "active": true, "archived": false,
           "reason": "baseline" },
    "1": { ... }
  }
}
```

`versions` is a dict keyed by the stringified version number (JSON object keys
are strings). A top-level `schema_version` allows future format migrations
without guessing the shape. All records live in one file so a single atomic
write covers every mutation.

**Alternative**: one file per version (the composer's current approach). Rejected
— scattering records across files makes atomic multi-record updates (activate +
archive previous) impossible without a coordinator, and there is no index of
which version is active.

### D3: Version number assignment — store-owned, monotonic
`save(record)` ignores any `version` supplied by the caller and assigns
`max(existing versions) + 1`. The first save when the store is empty assigns 0
(the seeded baseline). A collision (the assigned number already exists) raises
`PromptStoreError`. This guarantees monotonic numbering regardless of caller
behavior.

**Alternative**: let the caller supply the version. Rejected — the composer's
version string (`classify-v0`, `classify-v1`) is not a monotonic integer, and
two callers could race to pick the same number.

### D4: Atomic writes — write-to-temp then `os.replace`
On every mutation, the store serializes the full in-memory state to a temporary
file in the same directory (`{store_path}.tmp.{pid}`) and calls
`os.replace(tmp, store_path)`. `os.replace` is an atomic rename on POSIX
(same-directory rename is atomic). The temp file is cleaned up on failure. When
`atomic_writes=false`, the store writes directly to the path (for debugging
only).

**Alternative**: `flock` + in-place rewrite. Rejected — a partial write is
visible to readers if the process crashes mid-write; rename guarantees readers
see either the old or the new file, never a truncation.

### D5: Activation semantics — deactivate and archive previous atomically
`activate(version_number)` loads the target record, finds the current active
record (if any), and in a single in-memory mutation sets the target's
`active=true, archived=false` and the previous active's `active=false,
archived=true`, then flushes atomically (D4). If the target is already active,
the call is a no-op. This guarantees the "exactly one active" invariant is never
violated, even mid-flush.

**Alternative**: two separate writes (deactivate, then activate). Rejected — a
crash between them leaves zero or two active versions.

### D6: Immutability enforcement — field allowlist for updates
The only mutation paths are `activate` (sets `active`/`archived`) and `prune`
(deletes a record). There is no general `update` method. A hypothetical
`update(version, **fields)` would reject any key outside `{active, archived}`
with `PromptStoreError`. Since the public API exposes no such method,
immutability is enforced by construction. The spec's immutability scenarios are
tested by attempting to mutate via the internal `_set_field` helper, which
validates the field is in the mutable allowlist.

**Alternative**: copy-on-write records with frozen dataclasses. Rejected — the
records are JSON-loaded dicts; a frozen-dataclass layer adds conversion overhead
for no extra safety beyond the allowlist.

### D7: Lineage reconstruction — walk `base_version` to the root
`lineage(version_number)` starts at the requested version and prepends each
`base_version` ancestor until reaching a record whose `base_version` is `null`
(the baseline v0). If an ancestor's `base_version` points to a non-existent
record, it raises `PromptStoreError` naming the missing ancestor. The result is
ordered oldest-to-newest. The walk is bounded by the number of versions, so it
cannot loop (version numbers are monotonic and `base_version < version` by
construction).

**Alternative**: store a materialized lineage list on each record. Rejected —
duplicates the walkable links and must be updated on every save, risking
divergence from the `base_version` links.

### D8: Pruning — delete oldest archived, protect active and its ancestors
`prune()` runs after a save when `max_versions` is set and
`len(versions) > max_versions`. It computes the protected set (the active version
plus all its ancestors via D7), then sorts the remaining archived versions by
`version` ascending and deletes them until `len(versions) <= max_versions`.
Non-archived versions are never pruned (they may be candidates for future
activation). The active version and its ancestors are always protected even if
archived. Pruning is automatic on save; there is no separate prune CLI command.

**Alternative**: prune only on explicit request. Rejected — the spec says pruning
MAY be automatic; making it automatic keeps the store within the configured limit
without operator intervention, and the log records every deletion for audit.

### D9: Operation log — append-only JSONL
`log_path` (default `prompt_store.jsonl`) receives one JSON object per line:
`{timestamp, operation, version, details}`. Operations: `save`, `activate`,
`archive`, `get`, `get_active`, `list`, `lineage`, `reload`, `prune`. The log is
opened in append mode (`"a"`) for each write, so concurrent processes do not
overwrite each other (though the store itself is single-process). The log is
best-effort: a log write failure is caught and logged to stderr, never blocking
the store operation.

**Alternative**: SQLite log. Rejected — overkill for a single-process component
and adds a dependency for no querying benefit at this scale.

### D10: v0 baseline seeding — store owns v0
On first init (store file does not exist or is empty), `init_store(config)`
seeds version 0 from `CLASSIFICATION_PROMPT_V0` with `base_version=null`,
`active=true`, `archived=false`, `reason="baseline"`. This makes the store
complete: every version including the baseline is a record, and lineage always
terminates at v0. The seeded v0's `text` is `CLASSIFICATION_PROMPT_V0.text` and
`hash` is its sha256.

**Alternative**: supply v0 as an external file outside the store. Rejected —
lineage reconstruction would have to special-case the missing v0 record, and the
"exactly one active" invariant has no home before the first composed version is
saved.

### D11: Corrupted store — fail hard, no auto-recovery
`load()` parses the JSON and validates the top-level shape (`schema_version`,
`versions` dict). On `json.JSONDecodeError` or a shape violation, it raises
`PromptStoreError` naming the file and the parse error. There is no automatic
fallback to a backup or the last-known-good state — silent recovery from a
corrupted store risks activating a stale or wrong version. The operator inspects
the file, restores from an external backup, and retries. The atomic-write
strategy (D4) makes corruption unlikely in the first place.

**Alternative**: keep a `.bak` copy of the last successful write and fall back to
it on corruption. Rejected — the backup may itself be stale or inconsistent with
the log; auto-recovery masks the problem and violates the "single source of
truth" contract.

### D12: Config dataclass — `[prompt_store]` section
`PromptStoreConfig.from_config(config_path)` reads `[prompt_store]` from
`config.toml` via `tomllib`, mirroring `VersionComparatorConfig`. Keys:
`store_path` (required — raises `PromptStoreError` if absent), `max_versions`
(default `None` = unlimited), `atomic_writes` (default `True`), `log_path`
(default `prompt_store.jsonl`). Defaults are module-level constants matching the
existing convention (`DEFAULT_*`).

**Alternative**: derive `store_path` from `run_id`. Rejected — the store must
persist across runs to be the source of truth; a per-run path would reset the
version history every cycle.

## Risks / Trade-offs

- **[Single-process access]** → The store has no locking; two processes writing
  the same file will lose data. Mitigation: the cycle orchestrator is the single
  driver; document that the store is not safe for concurrent writers.
- **[Full rewrite on every mutation]** → Every save/activate rewrites the entire
  store file. Mitigation: the number of versions is small (≤ `max_versions` or
  ≤ a few dozen), so the rewrite is cheap; atomic rename keeps it safe.
- **[Pruning deletes permanently]** → Pruned versions are gone, not just hidden.
  Mitigation: only archived versions are pruned, the active lineage is always
  protected, and every pruning is logged with the deleted version number; the
  composer's per-version artifact files on disk are not deleted by the store.
- **[No activation history in the store]** → The store records only the current
  `active` flag, not who activated what when. Mitigation: the append-only log
  captures every activation with a timestamp; querying the log reconstructs the
  full activation history.
- **[POSIX-only atomicity]** → `os.replace` is atomic on POSIX; on Windows a
  same-directory rename is atomic only when the target does not exist or with
  `ReplaceFile`. Mitigation: the project targets macOS/Linux (the LLM server
  launcher rejects vLLM on darwin); Windows support is out of scope.

## Open Questions

- **Should the store expose a `deactivate` operation without archiving?** The
  spec only requires activate (which archives the previous). A standalone
  deactivate would leave zero active versions, violating the invariant. Deferred
  — add only if a use case needs a "pause" state.
- **Should `max_versions` count archived versions only or all versions?**
  Currently it counts all records. Deferred — if the store grows large with
  non-archived versions, switch the limit to count archived only; the current
  behavior is simpler and the scale is small.
- **Should the store verify `hash == sha256(text)` on load, or trust saved
  records?** Currently it validates on save and trusts on load. Deferred — add a
  load-time integrity check only if tampering becomes a concern.
