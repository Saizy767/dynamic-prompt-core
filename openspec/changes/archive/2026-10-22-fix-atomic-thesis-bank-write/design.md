# Design

## Context

`update_in_prompt_flags` in `state.py:129-130` writes the thesis bank with
`open(path, "w")` + `json.dump`, truncating the file before writing. The
project's `FileRunRepository._atomic_write` (`run_repository.py:61`) already
implements the safe pattern: temp file → `f.flush()` → `os.fsync()` →
read-back JSON validation → `os.replace()` → directory fsync. However,
`state.py` is in the application layer and must not import from
`infrastructure/`.

## Goals / Non-Goals

**Goals:**
- Make `update_in_prompt_flags` crash-safe: original file preserved on
  partial write.
- Keep the fix self-contained in `state.py` (no new cross-layer imports).

**Non-Goals:**
- Extracting a shared atomic-write utility (would require a new shared module;
  out of scope for a single call site).
- Adding locking or conflict detection (the orchestrator is single-threaded
  per run).
- Changing other non-atomic writes elsewhere in the codebase.

## Decisions

### Inline atomic write in `update_in_prompt_flags`

Write the JSON to a temp file named `.{basename}.tmp.{pid}` in the same
directory, fsync, read back and `json.load` to validate, then `os.replace`.
Clean up the temp file in a `finally` block. This mirrors the pattern in
`_atomic_write` without importing from infrastructure.

**Alternative**: extract a shared helper in `application/` or `domain/`.
Rejected — only one call site needs it; a shared utility can be extracted
later if more call sites appear.

### Same directory for temp file

The temp file must be in the same directory as the target so `os.replace` is
atomic on the same filesystem.

## Risks / Trade-offs

- [Extra I/O] The write now does an additional read-back pass. → Negligible
  for thesis bank sizes; correctness outweighs the minor cost.
- [Temp file leak on kill -9] `os.replace` hasn't run, so the original is
  intact; the temp file remains but is harmless (hidden, `.tmp.` prefix).
  → The `finally` block handles normal exceptions; `kill -9` is an extreme
  edge case.
