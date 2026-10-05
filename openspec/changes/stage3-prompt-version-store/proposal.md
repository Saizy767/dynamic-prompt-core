# Proposal

## Why

The pipeline composes prompt versions via `stage3-prompt-composer-v2` and decides
accept/rollback via `stage2-version-comparator`, but there is no persistent store
that owns the lifecycle of those versions. Today the active version is tracked
only in the cycle orchestrator's in-memory state and the built-in
`CLASSIFICATION_PROMPT_V0`; composed versions are scattered as individual
`prompt_v*_*_*.json` files with no index, no active flag, and no way to recover
a specific version or reconstruct lineage without scanning the results
directory. A single, durable prompt-version store is needed as the source of
truth for which version is active, which are archived, and how each version
derives from its predecessors.

## What Changes

- Add a new component (`prompt_store.py`) that persists every prompt version
  produced by `stage3-prompt-composer-v2` as a record in a single JSON store
  file. Each record carries `version` (monotonic integer), `text`, `hash`
  (sha256 of `text`), `rules`, `source_candidates`, `base_version`,
  `created_at`, `active` (boolean), `archived` (boolean), and `reason`.
- The store assigns version numbers itself (one greater than the highest
  existing); the caller never supplies a number. Saving a colliding version
  fails with an error.
- Support retrieving a version by number and retrieving the single active
  version. Exactly one version is active at any time; requesting the active
  version when none is set fails with an error.
- Support marking a version active: the previous active version is deactivated
  and archived atomically. Activating a missing version fails with an error.
- Enforce immutability of `text`, `hash`, `rules`, `source_candidates`,
  `base_version`, and `created_at` after save; only `active` and `archived` may
  change.
- Support listing all versions sorted by number with metadata only (no `text`),
  and reconstructing the lineage of a version by walking `base_version` links
  from the oldest ancestor to the requested version.
- Persist the store as one JSON file at a configurable path with atomic writes
  (write-to-temp then rename) to prevent corruption on crash. A corrupted store
  fails to load with a clear error.
- Optionally prune the oldest archived versions when a configurable
  `max_versions` limit is exceeded, never deleting the active version or any of
  its ancestors.
- Log all store operations (save, activate, archive, get, reload) to an
  append-only JSONL log with timestamp and version number.
- Add a `[prompt_store]` section to `config.toml` for `store_path` (required),
  `max_versions` (default unlimited), `atomic_writes` (default true), and
  `log_path` (default `prompt_store.jsonl`).

## Capabilities

### New Capabilities
- `stage3-prompt-version-store`: Owns the lifecycle of prompt versions —
  persisting each version produced by the composer, assigning monotonic version
  numbers, marking exactly one version active at a time, archiving displaced
  versions, enforcing immutability of saved content, listing version metadata,
  reconstructing version lineage, atomically persisting a single JSON store
  file, optionally pruning old archived versions, and logging all operations to
  an append-only log.

### Modified Capabilities
<!-- None — this change introduces a new standalone store component. It does not
     alter the requirements of the composer, comparator, or orchestrator; those
     components will adopt the store as a dependency in follow-up changes. -->

## Impact

- **New code**: `prompt_store.py` (component + CLI), mirroring the structure of
  `prompt_composer.py` and `version_comparator.py` (dataclass config from TOML,
  artifact load/validate, argparse CLI). Reuses `PromptArtifact` from
  `prompts.py` and the prompt-version dict shape produced by
  `prompt_composer.write_prompt_version`.
- **Config**: new `[prompt_store]` section in `config.toml`.
- **Dependencies**: no new third-party dependencies. Uses only the standard
  library (`json`, `hashlib`, `os`, `dataclasses`, `tomllib`).
- **Artifacts**: new `prompt_store.json` (the store file, path configurable) and
  `prompt_store.jsonl` (append-only operation log, path configurable).
- **Upstream**: consumes prompt-version dicts from
  `stage3-prompt-composer-v2` (`prompt_composer.write_prompt_version` output).
- **Downstream**: `stage2-version-comparator` and
  `stage3-cycle-orchestrator` will read the active version from the store and
  delegate activation to it, replacing the in-memory active-version tracking.
  Adoption is a follow-up change; this change delivers the store in isolation.
