# Proposal

## Why

`update_in_prompt_flags()` in `state.py` writes the thesis bank JSON via a bare
`open(path, "w")` + `json.dump`. If the process crashes mid-write, the existing
file is already truncated and left with a partial JSON document. This corrupts
the thesis bank — a file consumed by multiple downstream stages — and prevents
experiment recovery. The project's own `FileRunRepository` already uses an
atomic write pattern (temp file → fsync → read-back validation → `os.replace`),
but `update_in_prompt_flags` does not.

## What Changes

- Replace the direct overwrite in `update_in_prompt_flags()` with an atomic
  write: serialize to a temp file in the same directory, fsync, read back and
  validate JSON, then `os.replace` for atomic replacement. Clean up the temp
  file on any failure.
- Add a unit test verifying that the original file remains intact when the
  write fails (e.g. non-serializable content).

## Capabilities

### New Capabilities
<!-- None — this change fixes an existing function's write safety. -->

### Modified Capabilities
- `stage3-cycle-orchestrator`: The thesis bank update SHALL persist atomically,
  leaving the existing file untouched until the new content is fully validated
  and written.

## Impact

- **Code**: `src/dynamic_prompt_core/application/use_cases/run_cycle/state.py`
  — `update_in_prompt_flags()` write section (lines 129-130).
- **Tests**: `tests/unit/run_cycle/test_state.py` — add crash-safety test.
- **No behavior change** on the happy path; the function's observable output is
  identical. The change only affects failure semantics: the thesis bank is no
  longer left truncated on crash.
