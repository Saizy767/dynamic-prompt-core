# Proposal

## Why

The project ships 19 monolithic `.py` files in the repository root, bypassing
the layered architecture already specified in `architecture`. The
`src/dynamic_prompt_core/` package has a partial scaffold (`domain`,
`application`, `infrastructure`, `interfaces`) but none of the existing logic
lives there. Until the code is moved into the layers and the flat files are
removed, the architectural contracts (dependency direction, domain isolation,
port policy, composition roots) cannot be enforced and `lint-imports` /
`mypy` / `ruff` cannot pass against the real codebase.

## What Changes

- **BREAKING**: All 19 root-level `.py` files are removed after their logic is
  split into `src/dynamic_prompt_core/{domain,application,infrastructure,interfaces}/`.
- Split `asyncTask.py` into `infrastructure/llm/` (enums, adapters, client,
  response, errors) with a public `__init__.py` + `__all__`.
- Split `server_launcher.py` into `infrastructure/config/` and
  `infrastructure/server/` (loader, argv, readiness, launcher) plus
  `interfaces/server/main.py` composition root.
- Split `dataset.py` into `domain/models/`, `domain/services/`,
  `infrastructure/data/`, and `application/ports/dataset_repository/`.
- Split `prompts.py` into `domain/prompts/` (base + fixed layers) and
  `application/services/prompt_render.py`.
- Place `schemas.py` into `application/schemas/` grouped by purpose
  (`classification.py`, `thesis.py`, `refinement.py`).
- Place `metrics.py` into `application/services/metrics/` (classification,
  grouped, comparison) — kept out of `domain` due to numpy/sklearn.
- Split `normalize.py` into `application/ports/normalizer/` (port) and
  `infrastructure/nlp/normalizer.py` (pymorphy3/spaCy implementation).
- Place `runner.py` → `application/use_cases/run_baseline/`.
- Place `thesis_analyzer.py` → `application/use_cases/analyze_theses/`;
  clustering → `application/services/clustering/`.
- Split `rule_candidate_selector.py` into `domain/services/rule_selection.py`
  and `application/use_cases/select_candidates/`.
- Place `prompt_composer.py` → `application/use_cases/compose_prompt/`.
- Split `version_comparator.py` into `domain/services/decision.py` and
  `application/use_cases/compare_versions/`.
- Split `prompt_store.py` into `application/ports/prompt_repository/` and
  `infrastructure/storage/prompt_store.py`.
- Place `thesis_refiner.py` → `application/use_cases/refine_theses/` with a
  dedicated `teacher_llm_client` port.
- Place `cross_task_transfer.py` → `application/use_cases/cross_task_transfer/`.
- Place `cycle_orchestrator.py` → `application/use_cases/run_cycle/` with a
  typed dependency object.
- Split `cycle_observer.py` into `application/use_cases/observe_cycle/` and
  `interfaces/cli/observer.py`.
- Split `convert_csv_to_jsonl.py` into `infrastructure/data/csv_to_jsonl.py`
  and `interfaces/cli/convert.py`.
- Move `smoke_test.py` → `tests/integration/test_smoke.py` using public APIs only.
- Create all `application/ports/` protocols (`llm_client`,
  `teacher_llm_client`, `embedding_client`, `normalizer`,
  `prompt_repository`, `run_repository`, `dataset_repository`) via
  `typing.Protocol`.
- Establish `interfaces/cli/main.py` and `interfaces/server/main.py` as
  composition roots; restrict `logging.basicConfig` to entrypoints.
- Add `__init__.py` + `__all__` to every package exposing a public API;
  preserve existing public signatures (`AsyncTask`, `ServerLauncher`,
  `load_config`, `run_cycle`, etc.).
- Configure `pyproject.toml` dependencies + import-linter contracts
  (layered, domain isolation, infrastructure restriction, no cycles) +
  ruff + mypy + pytest.
- Migrate all tests into `tests/{unit,integration,contract}/` mirroring `src/`.
- Update `README.md` to use new entrypoint paths.
- Perform the migration atomically in a dedicated git branch
  (`migration/architecture`); remove old files only after new modules pass
  all checks.

## Capabilities

### New Capabilities
- `migration-to-architecture`: Defines the one-time structural migration of
  all root-level modules into the layered `src/dynamic_prompt_core/` packages,
  the concrete placement of each split module, the port inventory, the
  composition roots, and the atomic removal of the old flat files.

### Modified Capabilities
<!-- None. The architecture contracts are unchanged; this change realizes them. -->

## Impact

- **Code**: Every root-level `.py` file is relocated; `src/dynamic_prompt_core/`
  gains the full four-layer tree with subpackages.
- **Public API**: Preserved by re-export through package `__init__.py` + `__all__`;
  entrypoints move to `python -m dynamic_prompt_core.interfaces.cli.main` and
  `python -m dynamic_prompt_core.interfaces.server.main`.
- **Dependencies**: `pyproject.toml` gains `pytest`, `pytest-asyncio` dev deps;
  import-linter contracts enforced.
- **Tests**: Relocated under `tests/` mirroring `src/`; integration/contract
  tests use public APIs only.
- **Docs**: `README.md` examples updated to new entrypoint paths.
- **Git**: Migration executed in branch `migration/architecture`, merged after
  acceptance criteria pass.
