# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[prompt_store]` section to `config.toml` with keys `store_path`,
  `max_versions`, `atomic_writes`, and `log_path`, with inline comments
  documenting each default (`store_path` required, `max_versions` unlimited,
  `atomic_writes=true`, `log_path="prompt_store.jsonl"`). Verify the file parses
  with `tomllib.load` and the section round-trips.
- [x] 1.2 Create `prompt_store.py` with module docstring,
  `from __future__ import annotations`, a `PromptStoreConfig` dataclass with
  `from_config()` reading `[prompt_store]` from `config.toml` (mirroring
  `VersionComparatorConfig`), and default constants (`DEFAULT_STORE_PATH`,
  `DEFAULT_MAX_VERSIONS=None`, `DEFAULT_ATOMIC_WRITES=True`,
  `DEFAULT_LOG_PATH="prompt_store.jsonl"`). Verify `PromptStoreConfig.from_config()`
  returns configured values, falls back to defaults for missing optional keys, and
  raises `PromptStoreError` naming `store_path` when it is absent.
- [x] 1.3 Add a `PromptStoreError(ValueError)` exception class and a
  `PromptVersionRecord` dataclass holding `version`, `text`, `hash`, `rules`,
  `source_candidates`, `base_version`, `created_at`, `active`, `archived`, and
  `reason`. Verify `PromptVersionRecord` constructs with defaults and serializes
  to a plain dict via `dataclasses.asdict`.

## 2. Store record schema and validation

- [x] 2.1 Implement `_validate_record(record)` that checks all required fields are
  present (`version`, `text`, `hash`, `rules`, `source_candidates`,
  `base_version`, `created_at`, `active`, `archived`, `reason`) and that `hash`
  equals `sha256(text)` (full hexdigest). Raise `PromptStoreError` naming the
  missing field or the hash mismatch. Verify: a valid record passes; a record
  missing `version` fails naming `version`; a record with a wrong `hash` fails
  naming the mismatch.
- [x] 2.2 Implement `_normalize_input(prompt_version_dict, reason)` that converts a
  composer-produced prompt-version dict (with `version`, `text`, `hash`, `rules`,
  `source_candidates`, `base_version`, `created_at`) into a
  `PromptVersionRecord`-shaped dict, defaulting `active=False`, `archived=False`,
  and using the supplied `reason`. Verify a composer-shaped dict normalizes to a
  record with all fields populated.

## 3. Store file load and atomic persistence

- [x] 3.1 Implement `load()` that reads the store JSON file, validates the
  top-level shape (`schema_version` int, `versions` dict), and returns the
  in-memory versions dict. When the file does not exist, return an empty dict.
  Raise `PromptStoreError` naming the file and the parse error on
  `json.JSONDecodeError` or a shape violation (design D11). Verify: a missing
  file returns empty; a valid file loads all records; a corrupted file raises
  `PromptStoreError` naming the file.
- [x] 3.2 Implement `_flush(versions)` that serializes `{schema_version: 1,
  versions: versions}` to the store path atomically (design D4): write to a temp
  file `{store_path}.tmp.{pid}` in the same directory, then `os.replace(temp,
  store_path)`. When `atomic_writes=False`, write directly. Clean up the temp
  file on failure. Verify: after a flush, `load()` returns the same records; a
  crash mid-write (simulated by a failing replace) leaves the old file intact.
- [x] 3.3 Implement `init_store(config)` that creates a `PromptStore` instance,
  loads the file, and when empty seeds version 0 from
  `CLASSIFICATION_PROMPT_V0` with `base_version=None`, `active=True`,
  `archived=False`, `reason="baseline"`, then flushes (design D10). Verify: on a
  fresh path, `init_store` creates a store with v0 active; on an existing path,
  it loads the existing records without re-seeding.

## 4. Persist prompt version

- [x] 4.1 Implement `save(prompt_version_dict, reason)` that normalizes the input
  (task 2.2), assigns `version = max(existing) + 1` (or 0 when empty, design D3),
  validates the record (task 2.1), adds it to the in-memory dict, and flushes
  (task 3.2). Raise `PromptStoreError` on version collision. Verify: the first
  save after init assigns version 1 (v0 already exists); saving when v0–v2 exist
  assigns 3; the saved record's `hash` matches `sha256(text)`.
- [x] 4.2 Verify a collision is impossible in normal operation (the store always
  assigns `max+1`) but that manually inserting a duplicate key via the internal
  dict raises `PromptStoreError` naming the collision. Verify the caller-supplied
  `version` is always overwritten by the store-assigned number.

## 5. Retrieve by version id and active version

- [x] 5.1 Implement `get(version_number)` that returns the full record for the
  given version number. Raise `PromptStoreError` naming the missing version when
  it does not exist. Verify: an existing version returns the full record with all
  fields; a non-existent version raises `PromptStoreError` naming the version.
- [x] 5.2 Implement `get_active()` that returns the record with `active=True`.
  Raise `PromptStoreError` when no active version exists. Verify: after init, v0
  is active; after activating v1, `get_active()` returns v1; a store with no
  active flag raises `PromptStoreError`.

## 6. Mark version as active

- [x] 6.1 Implement `activate(version_number)` that finds the target record,
  finds the current active record (if any), and in a single in-memory mutation
  sets the target's `active=True, archived=False` and the previous active's
  `active=False, archived=True`, then flushes atomically (design D5). If the
  target is already active, it is a no-op. Raise `PromptStoreError` when the
  target version does not exist. Verify: activating v1 sets v1 active and v0
  archived+inactive; activating v1 again is a no-op; activating a missing version
  raises `PromptStoreError`.
- [x] 6.2 Verify the "exactly one active" invariant holds after every activation:
  across a sequence of activations (v1, v2, v3), exactly one record has
  `active=True` at each step, and each previously-active version has
  `archived=True, active=False`.

## 7. Immutability enforcement

- [x] 7.1 Implement `_set_field(version_number, field, value)` that rejects any
  `field` outside the mutable allowlist `{active, archived}` with
  `PromptStoreError` naming the immutable field (design D6). Verify: setting
  `active` or `archived` succeeds; setting `text`, `hash`, `rules`,
  `source_candidates`, `base_version`, or `created_at` raises
  `PromptStoreError`.
- [x] 7.2 Verify that after a save, reloading the store returns identical values
  for all immutable fields, and that the only fields that differ across an
  activate cycle are `active` and `archived`.

## 8. List all versions

- [x] 8.1 Implement `list_versions()` that returns a list of metadata dicts
  (`version`, `hash`, `active`, `archived`, `created_at`, `base_version`) sorted
  by version number ascending, excluding the `text` field. Verify: a store with
  v0–v2 returns three metadata dicts in order 0, 1, 2, none containing `text`; an
  empty store returns an empty list.

## 9. Version lineage reconstruction

- [x] 9.1 Implement `lineage(version_number)` that walks `base_version` links from
  the requested version to the root (design D7), returning an ordered list of
  records from the oldest ancestor to the requested version. Raise
  `PromptStoreError` naming the missing ancestor when a `base_version` points to
  a non-existent record. Verify: for v3 with `base_version=2`, v2→1, v1→0,
  lineage returns `[v0, v1, v2, v3]`; for v0 (baseline), lineage returns `[v0]`;
  a broken link raises `PromptStoreError` naming the missing ancestor.

## 10. Version count limit and pruning

- [x] 10.1 Implement `_protected_versions()` that returns the set containing the
  active version plus all its ancestors via `lineage` (design D8). Verify: with v3
  active and lineage v0→v1→v2→v3, the protected set is `{0, 1, 2, 3}`.
- [x] 10.2 Implement `prune()` that, when `max_versions` is set and
  `len(versions) > max_versions`, deletes the oldest archived versions not in the
  protected set until `len(versions) <= max_versions`, flushing after each
  deletion. No-op when `max_versions` is `None`. Verify: with 15 versions,
  `max_versions=10`, and v14 active with ancestors v0–v13, no versions are pruned
  (all are protected or non-archived); with 15 archived versions and only v14
  active+protected, the 5 oldest unprotected archived versions are deleted.
- [x] 10.3 Verify pruning never deletes the active version or any ancestor of the
  active version, even when they are archived and old.

## 11. Operation logging

- [x] 11.1 Implement `_log(operation, version, details)` that appends one JSON
  object (`{timestamp, operation, version, details}`) to `log_path` in append
  mode (design D9). A log write failure is caught and printed to stderr, never
  blocking the store operation. Verify: two operations produce two lines, each
  valid JSON with `timestamp` and `version`; a log path in a non-existent
  directory does not crash the store.
- [x] 11.2 Wire logging into `save`, `activate`, `get`, `get_active`,
  `list_versions`, `lineage`, `load`, and `prune`. Verify each operation appends a
  log line with the correct `operation` name and `version` number.

## 12. CLI and end-to-end integration

- [x] 12.1 Implement the argparse CLI (`python prompt_store.py [--config
  config.toml] <command> [args]`) with subcommands: `init`, `save <artifact.json>
  [--reason <text>]`, `get <version>`, `active`, `activate <version>`, `list`,
  `lineage <version>`, `prune`. Verify `--help` lists all subcommands and a `init`
  on a fresh path creates the store with v0 active.
- [x] 12.2 Run an end-to-end sequence via the CLI: `init` a fresh store, `save`
  two composer-produced prompt-version artifacts, `activate 1`, `active` returns
  v1, `list` shows v0 archived and v1 active, `lineage 1` returns `[v0, v1]`,
  `activate 2`, `active` returns v2, `lineage 2` returns `[v0, v1, v2]`. Verify
  the store file is valid JSON after every step and the log contains one line per
  operation.
- [x] 12.3 Verify the store survives a process restart: run `init` + `save` +
  `activate` in one process, then in a second process load the same `store_path`
  and confirm `active` returns the activated version and `list` shows all
  versions with correct `active`/`archived` flags.
