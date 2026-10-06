# Design

## Context

The project is a flat collection of 19 Python modules at the repository root
(`runner.py`, `cycle_orchestrator.py`, `server_launcher.py`, `asyncTask.py`,
`prompts.py`, `metrics.py`, etc.) with a shared `config.toml` and
`requirements.txt`. There is no `pyproject.toml`, no `src/` layout, and no
dependency-boundary enforcement. Existing components are standalone modules with
dataclass configs loaded from TOML sections and argparse CLIs. Two infrastructure
adapters already exist: `server_launcher.py` (`ServerLauncher`) and
`asyncTask.py` (`AsyncTaskClient`). The orchestrator (`cycle_orchestrator.py`)
is the top-level use case that coordinates the pipeline stages.

## Goals / Non-Goals

**Goals:**
- Define the target package layout and the import-linter contracts that enforce
  it.
- Specify the port, composition-root, and use-case structures.
- Configure mypy and ruff in `pyproject.toml`.
- Resolve all fundamental architectural decisions (no open architectural
  questions remain in the spec).

**Non-Goals:**
- Migrating existing code into the new structure (separate task).
- Implementing concrete use cases beyond the `run_cycle` contract.
- Implementing concrete adapters beyond existing `ServerLauncher` and
  `AsyncTaskClient`.
- Full test implementation.
- CI pipeline setup (separate task if CI already exists).

## Decisions

### D1: `src/` layout with root package `dynamic_prompt_core`

Source lives under `src/dynamic_prompt_core/` with four layer subpackages:

```
src/dynamic_prompt_core/
    domain/
    application/
        ports/
            inbound/
            outbound/
        use_cases/
            run_cycle/
    infrastructure/
    interfaces/
        cli/
            main.py
        server/
            main.py
```

The `src/` layout prevents accidental imports of the package from the repo root
during tests and is the Python packaging convention. The root package name
`dynamic_prompt_core` matches the project directory.

**Alternative**: keep modules at the repo root. Rejected — no boundary
enforcement is possible without a package hierarchy, and `src/` is the standard
layout for importable packages.

### D2: import-linter contracts in `pyproject.toml`

import-linter is configured under `[tool.importlinter]` with named contracts:

- **`domain-isolated`**: `domain` forbids `application`, `infrastructure`,
  `interfaces`, and all third-party packages except an explicit allowlist
  (initially empty; `dataclasses`, `typing`, `enum`, `collections` are stdlib
  and need no allowlist).
- **`application-isolated`**: `application` forbids `infrastructure` and
  `interfaces`.
- **`infrastructure-isolated`**: `infrastructure` forbids
  `application.use_cases` and `interfaces`. `infrastructure` MAY import
  `application.ports` and `domain`.
- **`layer-direction`**: `interfaces` → `application` / `infrastructure` →
  `domain`. Enforced via `LayeredArchitecture` contract with layers in order
  `domain`, `application`, `infrastructure`, `interfaces`.
- **`no-cyclic-imports`**: `Forbidden` contract on known cyclic pairs, or the
  built-in `no-cycles` check.

**Alternative**: ArchUnit or custom AST scripts. Rejected — import-linter is
purpose-built for Python import boundaries and integrates with `lint-imports`.

### D3: Ports as `typing.Protocol` in `application/ports/{inbound,outbound}/`

- **Outbound ports** (driven by application, implemented by infrastructure):
  `llm_client` (LLM calls), `task_store` (async task management),
  `server_launcher` (server lifecycle).
- **Inbound ports** (driving the application): `run_cycle_input` (use case
  input DTO).

Each port is a `typing.Protocol` with typed methods. No infrastructure imports.
`abc.ABC` is used only if runtime `isinstance` checks or mixin hierarchies
require it, with a comment justifying the choice.

**Alternative**: a single `ports/` package with name suffixes. Rejected —
subpackages group by direction, making the dependency-inversion direction
visible from the import path.

### D4: Composition roots with separate `build_*_deps()` functions

`interfaces/cli/main.py` and `interfaces/server/main.py` each have:

1. `parse_args()` — argparse.
2. `load_config(path)` — read TOML into typed config objects.
3. `build_cli_deps(config)` / `build_server_deps(config)` — instantiate
   infrastructure adapters, wire them to ports, return a typed dependency
   object.
4. `configure_logging(config)` — set up handlers/formatters.
5. `run()` — orchestrate: parse → load → build → configure logging → invoke use
   case / start launcher → handle signals → shutdown.

The `build_*_deps()` functions are extracted rather than inline because the
server and CLI share some wiring (LLM client, task store) but differ in
lifecycle, and separate functions make the shared parts reusable without a
shared base class.

**Alternative**: inline wiring in `main()`. Rejected — the server and CLI would
duplicate the shared wiring; extracted functions are testable in isolation.

### D5: `run_cycle` use case with typed dependency object

`application/use_cases/run_cycle/` contains:

- `run_cycle_deps.py` — a `@dataclass(frozen=True)` `RunCycleDeps` holding the
  outbound ports the use case needs (`llm_client`, `task_store`, etc.).
- `run_cycle.py` — `async def run_cycle(deps: RunCycleDeps, input: ...) ->
  RunCycleResult`.

The use case receives `RunCycleDeps` from the composition root; it never
instantiates infrastructure classes. `RunCycleDeps` is the contract — adding a
dependency is a breaking change to the use case's interface.

**Alternative**: a service locator / `**kwargs`. Rejected — untyped, hides the
dependency contract, and makes breaking changes invisible to the type checker.

### D6: mypy and ruff configuration in `pyproject.toml`

- `[tool.mypy]`: `strict = true`, `packages = ["dynamic_prompt_core"]`,
  explicit per-module overrides for third-party stubs if needed.
- `[tool.ruff]`: `target-version = "py311"`, `line-length = 100`, select rules
  for `E`, `F`, `I` (isort), `UP` (pyupgrade), `B` (bugbear). Ruff replaces
  flake8 + black + isort with a single fast tool.

**Alternative**: flake8 + black + isort. Rejected — three tools with separate
configs versus one; ruff is faster and covers the same rules.

### D7: Configuration separation

- `pyproject.toml` — tool configuration (import-linter, mypy, ruff), package
  metadata, build system.
- `config.toml` — application configuration (model endpoints, dataset paths,
  cycle parameters, logging levels). Loaded via CLI `--config` argument.
  Architecture dependency rules are never defined here.

**Alternative**: YAML or JSON for app config. Rejected — the project already
uses `config.toml`; TOML is native to Python 3.11+ (`tomllib`) and avoids a
new dependency.

### D8: Logging configuration in composition root

`configure_logging(config)` in each `main.py` sets up `logging.basicConfig` or
a `dictConfig` from the app config. `domain` and `application` call
`logging.getLogger(__name__)` and log through the standard API — no handlers,
formatters, or levels configured there. External logging integrations (e.g.,
structured JSON logging) live in `infrastructure/` as handler implementations.

### D9: Graceful shutdown via `asyncio` signal handling

Both entrypoints register SIGINT and SIGTERM handlers via
`loop.add_signal_handler(signal.SIGINT, handler)`. The handler sets an
`asyncio.Event` or cancels the main task, triggering controlled shutdown.
Mandatory cleanup (flushing logs, closing the LLM client, stopping the server)
runs in a `finally` block before exit. Exit codes: 0 on clean stop, 1 on
unrecoverable error, 130 on SIGINT.

**Alternative**: `signal.signal()` callbacks. Rejected — they run in the main
thread and cannot safely interact with the asyncio loop; `add_signal_handler`
is the asyncio-native approach.

### D10: Public API via `__init__.py` + `__all__`

Packages that expose a public API (`domain`, `application.ports.outbound`,
`application.use_cases.run_cycle`) declare `__all__` in their `__init__.py`.
Internal packages (`infrastructure.llm`, `infrastructure.storage`) do not
require `__all__`. `__init__.py` re-exports public symbols; it does not contain
implementations.

### D11: Pydantic at boundaries, not in domain

Pydantic models are used for configuration loading (`application` boundary) and
external API serialization (`interfaces` boundary). `domain` uses plain
dataclasses and `typing` constructs. If a domain entity needs validation, it is
done via `__post_init__` on a dataclass, not via Pydantic — keeping `domain`
framework-free.

## Risks / Trade-offs

- **[Migration not included]** → This change defines the target but does not
  move code. Mitigation: migration is a separate task; the spec and design
  provide the target structure and rules for that task to follow.
- **[import-linter false positives]** → Dynamic imports or `TYPE_CHECKING`
  blocks may trip the linter. Mitigation: import-linter supports
  `ignore_imports` for known safe cases; configure narrowly.
- **[Strict mypy on existing code]** → Existing modules lack type annotations.
  Mitigation: mypy strict mode applies to the new package; migrated modules
  add annotations as they move in (migration task).
- **[ruff vs. existing style]** → Existing code may not match ruff's formatting.
  Mitigation: `ruff format` is run as part of migration; this change only
  configures the tool.
- **[Two entrypoints share wiring]** → `build_cli_deps()` and
  `build_server_deps()` may diverge. Mitigation: shared wiring is extracted
  into helper functions in `interfaces/` (not a shared base class); divergence
  is visible in review.

## Open Questions

- **Should `domain` allow `pydantic` as an explicit exception for any entity?**
  Currently no — all domain entities use dataclasses. Deferred — revisit only if
  a domain entity genuinely needs Pydantic's validation or serialization.
- **Should the `no-cyclic-imports` contract use import-linter's built-in cycle
  detection or explicit `Forbidden` pairs?** The built-in is broader but less
  precise. Deferred — start with the built-in and switch to explicit pairs only
  if it produces false negatives.
