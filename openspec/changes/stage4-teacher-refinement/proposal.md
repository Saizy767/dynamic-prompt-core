# Proposal

## Why

The `stage4-teacher-refinement` capability is already implemented in
`application/use_cases/refine_theses/refiner.py` and its behavioral spec is
synced to `openspec/specs/stage4-teacher-refinement/spec.md`. However, the
current implementation does not satisfy the architectural contracts from
`architecture`: it imports `AsyncTask` directly from `infrastructure.llm`
instead of going through a port, mixes pure filtering/reformulation logic into
the use case module, constructs the `AsyncTask` itself instead of receiving it
via a typed dependency object, is not wired in the composition root
(`interfaces/cli/main.py`), and the existing `TeacherLLMClient` port defines
`classify` / `extract_theses` methods that do not match the refinement contract
(review + reformulate). This change brings the component into compliance with
the layered architecture without altering its externally observable behavior.

## What Changes

- **Redefine the `TeacherLLMClient` port** (`application/ports/outbound/teacher_llm_client.py`)
  to define the refinement contract — a `review_theses` method that takes a text
  and candidate theses and returns a structured `RefinementPlan` (keep,
  reformulate, drop, add) — replacing the current `classify` / `extract_theses`
  methods. The port SHALL use `typing.Protocol` and SHALL NOT depend on
  `infrastructure`.
- **Add a `TeacherClient` implementation** in
  `infrastructure/llm/teacher_client.py` that implements the port via `AsyncTask`
  pointed at the teacher endpoint. The implementation SHALL contain no
  refinement business logic — only transport, prompt assembly, and response
  parsing. The teacher endpoint SHALL be distinct from the working model's
  endpoint.
- **Extract pure refinement logic** into `domain/services/thesis_refinement.py`
  — applying a `RefinementPlan` to candidate theses (keep, reformulate with
  length enforcement, drop, add) is framework-free and belongs in `domain`. The
  use case delegates to this service for the pure transform.
- **Refactor the use case** (`application/use_cases/refine_theses/`) to receive
  dependencies through a typed `RefineThesesDeps` object (`teacher_llm_client`,
  `run_repository`) instead of constructing `AsyncTask` itself. The use case
  SHALL return a typed `RefineThesesResult` (statistics) or raise. No concrete
  implementations are created inside the use case.
- **Wire `TeacherClient` in the composition root** (`interfaces/cli/main.py`).
  No other module SHALL instantiate `TeacherClient`. The CLI entrypoint
  assembles the `TeacherClient`, passes it through `RefineThesesDeps`, and
  invokes the use case.
- **Restrict logging configuration**: the refinement modules SHALL NOT call
  `logging.basicConfig`. Logging configuration resides in the composition root
  only. The append-only refinement log continues to use `logging.getLogger`.
- **Add `teacher.use_refined` explicit-override scenario**: when
  `use_refined=false`, downstream stages use `theses_raw_tiny` even when the
  refined artifact is available. This was missing from the synced spec.
- **Add `teacher.allow_additions` disabled scenario**: when `allow_additions=false`,
  the teacher model does not add new theses. This refines the existing
  "Add missed theses" requirement.
- No behavioral change to the artifact format, filtering semantics, failure
  isolation, cost accounting, or candidate loading — those are already
  implemented and synced.

## Capabilities

### New Capabilities
<!-- None — this change refactors an existing capability into architectural
     compliance. No new capability is introduced. -->

### Modified Capabilities
- `stage4-teacher-refinement`: Adds architectural requirements — port contract
  (`TeacherLLMClient` with `review_theses`), infrastructure implementation
  (`infrastructure/llm/teacher_client.py` via `AsyncTask`), domain purity
  (`domain/services/thesis_refinement.py`), use case contract (typed dependency
  object, no concrete wiring), composition root wiring, logging restriction
  (no `logging.basicConfig`), and the `use_refined=false` explicit override. The
  behavioral requirements (filter, reformulate, artifact, failure isolation,
  cost accounting) are already synced and unchanged.

## Impact

- **New code**: `infrastructure/llm/teacher_client.py` (port implementation),
  `domain/services/thesis_refinement.py` (pure refinement logic),
  `application/use_cases/refine_theses/refine_theses_deps.py` (typed dependency
  object).
- **Modified code**: `application/ports/outbound/teacher_llm_client.py` (port
  contract redefined), `application/use_cases/refine_theses/refiner.py` (use
  case refactored to use port + dependency object + domain service),
  `interfaces/cli/main.py` (composition root wires `TeacherClient`).
- **Config**: no new keys. The existing `[teacher]` section in `config.toml` is
  unchanged; `teacher.endpoint` and `teacher.model_name` remain required.
- **Dependencies**: no new third-party dependencies. Reuses `AsyncTask`
  (already present), `pydantic` (already present for `RefinementPlan`).
- **Artifacts**: unchanged — `theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl`
  format and fields are identical.
- **Upstream**: unchanged — consumes `results_*.jsonl` from
  `stage1-baseline-runner`.
- **Downstream**: unchanged — `stage1-thesis-analyzer` consumes
  `theses_refined` / `theses_raw_tiny` as before.
- **Static checks**: `lint-imports` passes (use case depends on port, not
  infrastructure; domain has no framework imports), `mypy` passes.
