# Design

## Context

The `stage4-teacher-refinement` capability is already implemented in
`application/use_cases/refine_theses/refiner.py` (592 lines) and its behavioral
spec is synced to `openspec/specs/stage4-teacher-refinement/spec.md`. The
implementation works but violates the architectural contracts from
`architecture`:

- `refiner.py` imports `AsyncTask` directly from `infrastructure.llm`
  (`refiner.py:32`), so the use case depends on infrastructure — forbidden by
  the "Application isolated from infrastructure" contract.
- The `TeacherLLMClient` port (`application/ports/outbound/teacher_llm_client.py`)
  exists but defines `classify` / `extract_theses` methods that do not match the
  refinement contract. The use case does not use the port at all.
- `_build_teacher_task` (`refiner.py:216`) constructs the `AsyncTask` inside the
  use case, so the use case creates a concrete infrastructure dependency —
  forbidden by the "Use case contract" and "Dependency inversion" contracts.
- Pure refinement logic (`_apply_refinement`, `refiner.py:277`) — keep, drop,
  reformulate with length enforcement, add — is framework-free but lives in the
  use case module instead of `domain/services/`.
- The composition root (`interfaces/cli/main.py`) does not wire a `TeacherClient`
  or expose a refinement CLI subcommand. The refiner has its own `main()` with
  `pass  # logging configured by composition root` (`refiner.py:572`).
- `refiner.py` uses `logging.getLogger` (correct) but the module is not wired
  through the composition root's `configure_logging`.

The existing `RunCycleDeps` (`run_cycle_deps.py`) shows the dependency-object
pattern: a frozen dataclass of ports. The `LLMClient` and `RunRepository` ports
show the `typing.Protocol` + `@runtime_checkable` pattern.

## Goals / Non-Goals

**Goals:**
- Make the refinement use case depend on `application.ports.outbound.teacher_llm_client`,
  not `infrastructure.llm`.
- Move pure refinement logic to `domain/services/thesis_refinement.py`.
- Receive all dependencies via a typed `RefineThesesDeps` object.
- Wire `TeacherClient` in `interfaces/cli/main.py` only.
- Preserve all existing behavior: artifact format, filtering, failure isolation,
  cost accounting, logging.

**Non-Goals:**
- Changing the artifact format, filtering semantics, or failure isolation.
- Optimizing the teacher prompt or the refinement algorithm.
- Adopting `theses_refined` in `stage1-thesis-analyzer` (follow-up change).
- Adding a `served_model_name` constructor parameter to `AsyncTask` (deferred —
  the override remains, now in `TeacherClient`).
- Scheduling refinement within the cycle orchestrator (follow-up change).

## Decisions

### D1: Port contract — `review_theses` replacing `classify` / `extract_theses`
The `TeacherLLMClient` port is redefined with a single `review_theses` method:

```python
async def review_theses(
    self, text: str, theses: list[str], *, system_prompt: str,
) -> RefinementReview:
    ...
```

`RefinementReview` is a typed result (plan + latency_ms + usage) defined in the
port module or `application/schemas/refinement.py`. The old `classify` /
`extract_theses` methods are removed — nothing in the codebase calls them (the
use case bypassed the port entirely).

**Alternative**: keep `classify` / `extract_theses` and add `review_theses`.
Rejected — the old methods are unused and do not match the refinement contract;
keeping them misleads future consumers.

### D2: `TeacherClient` in `infrastructure/llm/teacher_client.py`
The implementation wraps `AsyncTask` pointed at `teacher.endpoint`, overriding
`_served_model_name` to `teacher.model_name` (moving the existing
`_build_teacher_task` logic from `refiner.py:216` into the infrastructure layer).
It builds the system prompt, sends the call via `AsyncTask.analyze_raw`, validates
the response against `RefinementPlan`, and returns a `RefinementReview`. No
filtering, reformulation, or keep/drop decisions are made here.

**Alternative**: implement the port inside `refiner.py`. Rejected — the use case
must not import `infrastructure`, and the implementation belongs with the other
LLM adapters in `infrastructure/llm/`.

### D3: Pure logic in `domain/services/thesis_refinement.py`
`_apply_refinement` (`refiner.py:277`) is extracted to
`domain/services/thesis_refinement.py` as `apply_refinement(theses_raw, plan,
config) -> (theses_refined, filtered_out, added)`. The function uses only stdlib
types (`list`, `str`, `dict`) and the `RefinementPlan` dataclass. To keep `domain`
framework-free, `RefinementPlan` is redefined as a plain dataclass in
`domain/services/thesis_refinement.py` (or `domain/models/`) instead of a Pydantic
model. The Pydantic `RefinementPlan` stays in `application/schemas/refinement.py`
for response parsing; the use case converts the Pydantic model to the domain
dataclass before calling `apply_refinement`.

**Alternative**: keep `RefinementPlan` as Pydantic and let `domain` import Pydantic.
Rejected — the architecture spec says `domain` SHOULD remain independent of
framework-specific libraries; the conversion is a one-liner.

**Alternative**: leave the pure logic in the use case. Rejected — the spec
explicitly requires `domain/services/thesis_refinement.py`.

### D4: Typed dependency object — `RefineThesesDeps`
A frozen dataclass `RefineThesesDeps(teacher_llm_client: TeacherLLMClient,
run_repository: RunRepository)` in
`application/use_cases/refine_theses/refine_theses_deps.py`, mirroring
`RunCycleDeps`. The use case function `refine_theses(deps, input) -> RefineThesesResult`
receives all dependencies from the caller. The existing `ThesisRefiner` class is
refactored into a function (or keeps the class but takes `deps` in the
constructor instead of constructing `AsyncTask`).

**Alternative**: pass dependencies as individual function arguments. Rejected —
the architecture spec requires a typed dependency object when there are multiple
dependencies.

### D5: Composition root wiring in `interfaces/cli/main.py`
`build_cli_deps` is extended (or a parallel `build_refine_deps`) to construct
`TeacherClient(endpoint, model_name, ...)` and `RunRepository` and pass them
through `RefineThesesDeps`. A `--refine` subcommand (or a separate entry in the
argparse dispatch) invokes `refine_theses(deps, input)`. `configure_logging` is
called once in the composition root; no refinement module calls
`logging.basicConfig`.

**Alternative**: keep the refiner's standalone `main()`. Rejected — the
architecture spec requires the composition root to assemble dependencies; a
standalone `main()` that constructs `AsyncTask` violates the contract.

### D6: `RefineThesesResult` — typed return
The use case returns a `RefineThesesResult` dataclass with the `RefinerTotals`
fields (processed, teacher_errors, filtered, reformulated, added, skipped,
total_prompt_tokens, total_completion_tokens, total_latency_ms) plus the
artifact path. This makes the use case contract explicit: the caller gets a
typed result, not a tuple.

**Alternative**: return a tuple `(records, totals)`. Rejected — the architecture
spec says the use case SHALL return a defined typed result.

### D7: `RefinementPlan` schema stays in `application/schemas/refinement.py`
The Pydantic `RefinementPlan`, `ReformulateItem`, and `DropItem` models remain in
`application/schemas/refinement.py` (currently the file only has `RuleFormulation`
— the refinement models are added alongside it). The `TeacherClient` uses them
for response parsing. The domain service uses a framework-free mirror.

**Alternative**: put the Pydantic models in the port module. Rejected —
`application/schemas/` is the established location for Pydantic schemas, and the
port should define only the protocol.

## Risks / Trade-offs

- **[Port method signature change]** → Redefining `TeacherLLMClient` from
  `classify` / `extract_theses` to `review_theses` is a breaking change to the
  port. Mitigation: nothing currently implements or consumes the old methods; the
  change is safe.
- **[Pydantic-to-domain conversion overhead]** → Converting `RefinementPlan`
  (Pydantic) to a domain dataclass adds a mapping step. Mitigation: it is a
  one-liner per field and keeps `domain` framework-free, which is the
  architectural contract.
- **[`AsyncTask._served_model_name` override]** → The override moves from
  `refiner.py` to `TeacherClient` but remains a private-attribute access.
  Mitigation: documented in `TeacherClient`; a future change can add a
  constructor parameter to `AsyncTask`.
- **[Composition root grows]** → `main.py` gains refinement wiring. Mitigation:
  a separate `build_refine_deps` function keeps the composition root organized;
  the architecture spec explicitly allows separate composition functions.

## Open Questions

- **Should the refinement CLI be a subcommand of `main.py` or a separate
  entrypoint?** Deferred — a `--refine` flag is simplest; a separate entrypoint
  can be added later if the CLI grows.
- **Should `RunRepository` be used for artifact I/O or does the use case write
  directly?** The existing `write_refined_artifact` / `load_refined` are file
  I/O helpers. Using `RunRepository` is architecturally cleaner but the port
  currently only has `save_results` / `load_results`. Deferred — the use case can
  use `RunRepository` if extended, or keep file helpers until the port is
  broadened in a follow-up.
