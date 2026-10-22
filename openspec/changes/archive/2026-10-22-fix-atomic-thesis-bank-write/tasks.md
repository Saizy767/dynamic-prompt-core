# Tasks

## 1. Atomic write in `update_in_prompt_flags`

- [x] 1.1 In `src/dynamic_prompt_core/application/use_cases/run_cycle/state.py`,
      replace the direct `open(thesis_bank_path, "w")` + `json.dump` block
      (lines 129-130) with an atomic write: serialize to a temp file
      `.{basename}.tmp.{pid}` in the same directory, `f.flush()` +
      `os.fsync(f.fileno())`, read back and `json.load` to validate, then
      `os.replace(tmp, thesis_bank_path)`. Use a `try/finally` to remove the
      temp file on failure. Verify by reading the updated code.
- [x] 1.2 Verify the existing test
      `test_update_in_prompt_flags_sets_correct_values` still passes with
      `python3 -m pytest tests/unit/run_cycle/test_state.py::test_update_in_prompt_flags_sets_correct_values -x -v`.

## 2. Crash-safety test

- [x] 2.1 In `tests/unit/run_cycle/test_state.py`, add a test that creates a
      thesis bank file, then calls `update_in_prompt_flags` with content that
      would cause a write failure (e.g. monkeypatch `json.dump` to raise),
      and verifies the original file is unchanged and no temp file remains.
      Verify with
      `python3 -m pytest tests/unit/run_cycle/test_state.py::test_update_in_prompt_flags_preserves_original_on_failure -x -v`.

## 3. Full test suite and lint

- [x] 3.1 Run `python3 -m pytest tests/unit/run_cycle/test_state.py -v` and
      confirm all tests pass.
- [x] 3.2 Run `ruff check` and `mypy` on
      `src/dynamic_prompt_core/application/use_cases/run_cycle/state.py` and
      `tests/unit/run_cycle/test_state.py` and confirm no new errors.
