# Tasks

## 1. Project scaffolding and pyproject.toml

- [x] 1.1 Create `pyproject.toml` with `[build-system]` (setuptools), `[project]`
  metadata (name `dynamic-prompt-core`, version, python `>=3.11`), and
  `[tool.setuptools.packages.find]` pointing at `src/`. Verify `pip install -e .`
  succeeds and `python -c "import dynamic_prompt_core"` works.
- [x] 1.2 Create the `src/dynamic_prompt_core/` package tree with `__init__.py`
  for each layer: `domain/`, `application/`, `application/ports/`,
  `application/ports/inbound/`, `application/ports/outbound/`,
  `application/use_cases/`, `application/use_cases/run_cycle/`,
  `infrastructure/`, `interfaces/`, `interfaces/cli/`, `interfaces/server/`.
  Verify the directory tree matches design D1 and all packages are importable.
- [x] 1.3 Add `import-linter` to the project dependencies (in `pyproject.toml`
  `[project.optional-dependencies]` under a `dev` extra, or in
  `requirements.txt`). Verify `pip install -e ".[dev]"` installs
  `import-linter` and `lint-imports --version` runs.

## 2. import-linter contracts

- [x] 2.1 Configure `[tool.importlinter]` in `pyproject.toml` with the
  `layer-direction` contract: `LayeredArchitecture` with layers `domain`,
  `application`, `infrastructure`, `interfaces` (each as
  `dynamic_prompt_core.<layer>`), and `interfaces` as the top layer. Verify
  `lint-imports` runs and reports the contract as present (pass or fail).
- [x] 2.2 Add the `domain-isolated` contract: `domain` forbids imports of
  `application`, `infrastructure`, `interfaces`, and third-party packages
  (use `ForbiddenImports` or a layered contract with `domain` as the
  innermost layer). Verify `lint-imports` enforces it by checking a
  deliberate violation is caught.
- [x] 2.3 Add the `application-isolated` contract: `application` forbids
  `infrastructure` and `interfaces`. Verify `lint-imports` catches a
  deliberate `application` → `infrastructure` import.
- [x] 2.4 Add the `infrastructure-isolated` contract: `infrastructure` forbids
  `application.use_cases` and `interfaces`, but allows `application.ports`
  and `domain`. Verify `lint-imports` allows `infrastructure` →
  `application.ports` and catches `infrastructure` → `application.use_cases`.
- [x] 2.5 Add the `no-cyclic-imports` contract using import-linter's built-in
  cycle detection. Verify `lint-imports` catches a deliberate two-module
  cycle.
- [x] 2.6 Run `lint-imports` against the current codebase. Verify all contracts
  pass (the new package tree has no imports yet, so all boundary checks
  should be green).

## 3. Static analysis tooling (mypy and ruff)

- [x] 3.1 Configure `[tool.mypy]` in `pyproject.toml`: `strict = true`,
  `packages = ["dynamic_prompt_core"]`, `python_version = "3.11"`. Add
  per-module overrides for third-party libraries missing stubs if needed.
  Verify `mypy src/dynamic_prompt_core` runs without configuration errors.
- [x] 3.2 Configure `[tool.ruff]` in `pyproject.toml`: `target-version =
  "py311"`, `line-length = 100`, `select = ["E", "F", "I", "UP", "B"]`.
  Verify `ruff check src/dynamic_prompt_core` runs and `ruff format
  --check src/dynamic_prompt_core` runs.
- [x] 3.3 Add `mypy` and `ruff` to the `dev` optional dependencies in
  `pyproject.toml`. Verify `pip install -e ".[dev]"` installs both and
  their `--version` commands succeed.

## 4. Port definitions

- [x] 4.1 Create `application/ports/outbound/llm_client.py` with a
  `typing.Protocol` defining the LLM client contract (methods matching the
  existing `AsyncTaskClient` interface: `classify`, batch calls). Include
  typed method signatures using `async` and `Coroutine` return types. Verify
  the module imports without error and mypy accepts the Protocol.
- [x] 4.2 Create `application/ports/outbound/task_store.py` with a
  `typing.Protocol` for async task management (submit, status, result).
  Verify the module imports and mypy accepts it.
- [x] 4.3 Create `application/ports/outbound/server_launcher.py` with a
  `typing.Protocol` for server lifecycle (start, stop, health). Verify the
  module imports and mypy accepts it.
- [x] 4.4 Create `application/ports/inbound/run_cycle_input.py` with a typed
  dataclass or Protocol for the `run_cycle` use case input DTO. Verify the
  module imports and mypy accepts it.
- [x] 4.5 Create `application/ports/outbound/__init__.py` with `__all__`
  re-exporting the outbound port Protocols. Create
  `application/ports/inbound/__init__.py` with `__all__` re-exporting the
  inbound port. Verify `from dynamic_prompt_core.application.ports.outbound
  import LLMClient` (and siblings) works.
- [x] 4.6 Verify no port module imports `infrastructure`. Run `lint-imports`
  and confirm the `infrastructure-isolated` and `layer-direction` contracts
  still pass.

## 5. Use case contract (run_cycle)

- [x] 5.1 Create `application/use_cases/run_cycle/run_cycle_deps.py` with a
  `@dataclass(frozen=True)` `RunCycleDeps` holding typed fields for the
  outbound ports (`llm_client: LLMClient`, `task_store: TaskStore`, etc.).
  Verify the dataclass is frozen, typed, and mypy accepts it.
- [x] 5.2 Create `application/use_cases/run_cycle/run_cycle.py` with the
  `async def run_cycle(deps: RunCycleDeps, input: RunCycleInput) ->
  RunCycleResult` signature and a `RunCycleResult` typed dataclass. The body
  may be a stub (`raise NotImplementedError`) — this change defines the
  contract, not the implementation. Verify the module imports, mypy accepts
  the signature, and `run_cycle` does not import `infrastructure`.
- [x] 5.3 Create `application/use_cases/run_cycle/__init__.py` with `__all__`
  re-exporting `run_cycle`, `RunCycleDeps`, `RunCycleInput`, and
  `RunCycleResult`. Verify `from dynamic_prompt_core.application.use_cases.run_cycle
  import run_cycle` works.
- [x] 5.4 Run `lint-imports` and verify the `application-isolated` contract
  passes (the use case does not import `infrastructure` or `interfaces`).

## 6. Composition roots and entrypoints

- [x] 6.1 Create `interfaces/cli/main.py` with `parse_args()` (argparse with
  `--config`), `load_config(path)` (TOML via `tomllib`), `build_cli_deps(config)`
  (returns a `RunCycleDeps`), `configure_logging(config)`, and `async def
  main()` orchestrating parse → load → build → configure logging → invoke
  `run_cycle` → shutdown. The body of `build_cli_deps` may stub adapter
  instantiation. Verify the module imports, `python -m
  dynamic_prompt_core.interfaces.cli.main --help` prints the `--config` flag,
  and `main.py` contains no business logic (only orchestration calls).
- [x] 6.2 Create `interfaces/server/main.py` with `parse_args()`,
  `load_config(path)`, `build_server_deps(config)`, `configure_logging(config)`,
  and `async def main()` orchestrating parse → load → build → configure
  logging → start `ServerLauncher` → wait for signals → shutdown. The body of
  `build_server_deps` may stub adapter instantiation. Verify the module
  imports, `python -m dynamic_prompt_core.interfaces.server.main --help`
  prints the `--config` flag, and `main.py` contains no business logic.
- [x] 6.3 Create `interfaces/cli/__init__.py` and `interfaces/server/__init__.py`
  (internal packages — no `__all__` required per design D10). Verify they
  import without error.
- [x] 6.4 Run `lint-imports` and verify the `layer-direction` contract passes
  (`interfaces` imports `application` and `infrastructure` but not vice
  versa).

## 7. Graceful shutdown and logging

- [x] 7.1 Implement SIGINT and SIGTERM handling in `interfaces/cli/main.py`
  via `loop.add_signal_handler` setting an `asyncio.Event` or cancelling the
  main task. Mandatory cleanup runs in a `finally` block. Exit codes: 0
  clean, 1 error, 130 SIGINT. Verify a `SIGINT` during a stub run triggers
  the cleanup path and exits with code 130.
- [x] 7.2 Implement SIGINT and SIGTERM handling in `interfaces/server/main.py`
  via `loop.add_signal_handler`, stopping the `ServerLauncher` and running
  cleanup in a `finally` block. Verify a `SIGTERM` during a stub run triggers
  the cleanup path and exits with code 0.
- [x] 7.3 Implement `configure_logging(config)` in both entrypoints using
  `logging.basicConfig` or `logging.config.dictConfig` from the app config
  section. Verify calling it sets up a handler on the root logger and that
  `domain` / `application` modules log via `logging.getLogger(__name__)`
  without configuring handlers themselves.

## 8. Public API declarations and final verification

- [x] 8.1 Add `__all__` to `domain/__init__.py` (if domain exposes public
  types) and `application/ports/outbound/__init__.py` (already done in 4.5,
  verify). Add `__all__` to `application/use_cases/run_cycle/__init__.py`
  (already done in 5.3, verify). Confirm internal packages
  (`infrastructure/`, `interfaces/cli/`, `interfaces/server/`) do not
  require `__all__`. Verify all `__all__` entries resolve to importable
  names.
- [x] 8.2 Run `lint-imports` and verify all contracts pass: `layer-direction`,
  `domain-isolated`, `application-isolated`, `infrastructure-isolated`,
  `no-cyclic-imports`. Verify the output shows zero violations.
- [x] 8.3 Run `mypy src/dynamic_prompt_core` and verify it passes with no
  errors on the new package (strict mode). Fix any type errors in the
  scaffolded code.
- [x] 8.4 Run `ruff check src/dynamic_prompt_core` and `ruff format --check
  src/dynamic_prompt_core` and verify both pass. Fix any lint or formatting
  issues in the scaffolded code.
- [x] 8.5 Verify the acceptance criteria from the spec: four layers exist
  under `src/dynamic_prompt_core/`; ports are in
  `application/ports/{inbound,outbound}/` using `typing.Protocol`;
  `run_cycle` is in `application/use_cases/run_cycle/` with a typed
  dependency object; both entrypoints are thin composition roots; logging
  configuration is in the entrypoints; `lint-imports` passes.
