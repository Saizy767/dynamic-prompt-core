# Tasks

## 1. Import-linter contract for domain stdlib-only allowlist

- [x] 1.1 Add a domain stdlib-only allowlist contract to `pyproject.toml` under
  `[tool.importlinter]` that enforces `dynamic_prompt_core.domain` may import
  only the Python standard library plus an explicit allowlist (currently empty)
  of permitted third-party modules. Any third-party import in `domain` not on
  the allowlist is a violation. Use a `forbidden` contract listing
  non-allowlisted third-party top-level packages (`transformers`, `aiohttp`,
  `torch`, `sentence_transformers`, `pydantic`, `numpy`, `sklearn`, `pymorphy3`,
  `spacy`, `pyarrow`, `tqdm`) as the enforcement mechanism, with the allowlist
  documented as the policy. Verify the contract is syntactically valid by
  running `lint-imports` and confirming it appears without configuration errors.
- [x] 1.2 Run `lint-imports` and verify all contracts pass (including the new
  domain allowlist contract) against the current codebase. Confirm that no
  existing `domain` module imports any non-allowlisted third-party module — the
  contract must be green immediately since the layered migration already kept
  `domain` stdlib-only.

## 2. Architectural validation and behavior preservation

- [x] 2.1 Run `openspec validate --changes candidate-scoring-architecture-prep`
  and verify the spec deltas (`candidate-scoring-architecture` and
  `architecture`) are valid: every requirement has at least one scenario, the
  new capability has a `## Purpose` section, and the modified `architecture`
  delta uses `## ADDED Requirements` with no `## Purpose`.
- [x] 2.2 Run the existing test suite (`pytest`) and verify all tests pass,
  confirming that no production code or runtime behavior changed in this stage.
  The only file modified is `pyproject.toml` (import-linter contract); no
  `src/` files are touched.
- [x] 2.3 Run `mypy` and `ruff check` and verify they pass, confirming the
  `pyproject.toml` changes do not introduce type or lint regressions. Inspect
  `git diff --stat` and confirm the only modified file is `pyproject.toml`.
