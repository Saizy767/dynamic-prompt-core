# Proposal

## Why

The project is a flat collection of 19 Python modules at the repository root with
no enforced dependency boundaries. As the codebase has grown through four
completed stages, the absence of architectural contracts makes it unclear what is
a public API versus an implementation detail, which dependencies are allowed, and
where new code should live. This change defines the target layered architecture,
the dependency rules between layers, the port and composition-root policies, and
the tooling that enforces them — establishing what is an architectural contract
(stable) versus what is free to refactor (implementations, files, internal
modules).

## What Changes

- Introduce a four-layer package structure under `src/dynamic_prompt_core/`:
  `domain`, `application`, `infrastructure`, `interfaces`.
- Define and enforce dependency direction: domain ← application ←
  infrastructure ← interfaces, with `domain` isolated to the standard library.
- Restrict `infrastructure` to depend on `application.ports` only, not
  `application.use_cases` or `interfaces`.
- Place ports in `application/ports/inbound/` and
  `application/ports/outbound/`, using `typing.Protocol` by default.
- Establish `interfaces/cli/main.py` and `interfaces/server/main.py` as
  composition roots — thin entrypoints containing only configuration, dependency
  wiring, lifecycle orchestration, and use-case invocation.
- Place the `run_cycle` use case in `application/use_cases/run_cycle/` with a
  typed dependency object.
- Configure import-linter in `pyproject.toml` to check layer direction, domain
  isolation, infrastructure restrictions, and cyclic dependencies.
- Add mypy (type checking) and ruff (formatting/linting) as the static analysis
  tools for non-boundary architectural invariants.
- Define public API policy (`__init__.py` + `__all__`), refactoring policy (no
  immutable files; contracts evolve explicitly), and testing policy (integration
  tests use public APIs; unit tests may import internals).
- Define logging configuration policy (entrypoint or composition root) and
  configuration separation (tool config in `pyproject.toml`; app config in a
  separate file).
- Handle SIGINT and SIGTERM via controlled cancellation in both entrypoints.

## Capabilities

### New Capabilities
- `architecture`: Defines the project's architectural contracts — layered
  structure, dependency boundaries, port policy, composition roots, public API
  policy, contract evolution, and static-check enforcement — for
  `dynamic-prompt-core`.

### Modified Capabilities
<!-- None — this change introduces a new cross-cutting architecture capability.
     It does not alter the requirements of existing component specs. -->

## Impact

- **New code**: `src/dynamic_prompt_core/` package with four layer subpackages;
  `pyproject.toml` with import-linter, mypy, and ruff configuration.
- **Existing code**: All 19 root-level Python modules will eventually migrate
  into the layered structure, but migration is a separate task (non-goal). This
  change defines the target architecture and enforcement tooling, not the
  migration steps.
- **Dependencies**: import-linter (new), mypy (new), ruff (new). Existing
  dependencies (pydantic, aiohttp, transformers, etc.) remain unchanged.
- **Config**: `pyproject.toml` for tool configuration; application configuration
  remains in a separate file (currently `config.toml`).
- **CI**: CI should run architectural checks (`lint-imports`, mypy, ruff) before
  tests — wiring this into an existing CI pipeline is a separate task.
