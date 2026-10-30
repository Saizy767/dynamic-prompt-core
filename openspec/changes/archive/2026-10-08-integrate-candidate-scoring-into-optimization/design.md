# Design

## Context

The candidate-scoring classification architecture is fully implemented:

- `CandidateScorer` port (`application/ports/outbound/candidate_scorer.py`)
  with the `LLMLogitCandidateScorer` infrastructure implementation
  (`infrastructure/llm/scoring/candidate_scorer.py`).
- `ClassificationPolicy` protocol + `ArgmaxClassificationPolicy`
  (`application/services/classification_policy.py`).
- `Classification` domain model (`domain/models/classification.py`).
- `ClassifyInput` use case (`application/use_cases/classify_input/`).
- `BaselineRunner` already classifies through
  `scorer.score(text, candidates)` → `policy.classify(judgments)`
  (`run_baseline/runner.py:343-352`).

The generative classification path (`ClassificationResult`,
`classify_detailed`, `classify_many`) has been removed. Contract tests
(`tests/contract/test_no_generative_classification.py`) verify no production
code references those symbols.

The remaining gap is in the optimization cycle. `run_cycle/steps.py`
constructs `BaselineRunner` in three places (`run_active_on_dev`,
`run_new_on_dev`, `run_holdout`) without passing `scorer`, `policy`, or
`candidates` — all three are required `BaselineRunner.__init__` parameters
with no defaults. `RunCycleDeps` (`run_cycle_deps.py`) does not carry these
dependencies. The composition root (`build_cli_deps` in
`interfaces/cli/main.py`) does not wire them. The baseline runner CLI
(`runner.py:_main_async`) already contains the correct wiring pattern.

This stage is a **dependency-wiring stage**, not a scoring-integration
stage. The scorer is already implemented; this stage wires the existing
classification dependency graph into the optimization composition root.

## Goals / Non-Goals

**Goals:**

- Wire `CandidateScorer`, `ClassificationPolicy`, and `candidates` from
  the composition root through `RunCycleDeps` into every `BaselineRunner`
  construction inside the optimization round sequence.
- Ensure the optimization cycle uses the same scorer implementation, policy
  implementation, and candidate configuration as production classification.
  These are two distinct invariants:
  - **Between separate CLI processes**: same construction recipe and
    configuration, not the same Python objects.
  - **Within one process**: all `BaselineRunner` instances from the same
    `RunCycleDeps` reuse the same scorer and policy object instances.
- Extract a shared scorer construction helper so the baseline runner CLI
  and the cycle composition root share one wiring path.
- Make optimization unit tests runnable with a fake scorer and the real
  `ArgmaxClassificationPolicy` (no language model).
- Preserve all existing optimization-cycle behavior: iteration, candidate
  queue, rollback, reporting, state dump, resumability, stopping conditions.

**Non-Goals:**

- Change the `CandidateScorer` port, `ClassificationPolicy` protocol, or
  `Classification` domain model.
- Introduce an optimization-specific scorer or policy.
- Change the optimization algorithm, metric semantics, or stopping
  conditions.
- Introduce cross-example batching, score caching, or model-lifecycle
  optimization.
- Change `BaselineRunner` — it already uses candidate scoring correctly.
- Persist infrastructure-specific tensors, token IDs, or logits as
  optimization-domain state.

## Decisions

### Decision: Add three fields to `RunCycleDeps`

`RunCycleDeps` gains `candidate_scorer: CandidateScorer`,
`classification_policy: ClassificationPolicy`, and
`candidates: tuple[Candidate, ...]` as required fields (no defaults). This
is breaking for direct constructors, but `RunCycleDeps` is a frozen
dataclass assembled only in the composition root and in test fixtures.
Making them required (not optional) forces every call site to supply them
— a missing scorer is a wiring bug, not a runtime fallback case.

`candidate_scorer` and `classification_policy` are application
dependencies (outbound port and application-internal strategy). `candidates`
is cycle-wide classification configuration, not per-example or per-task
data — it has the same lifetime as `RunCycleDeps`. Placing it in
`RunCycleDeps` is correct because the entire optimization cycle uses one
fixed candidate set.

- **Alternative considered**: default to `None` and fall back to generative
  classification. Rejected: generative classification no longer exists, and
  a silent fallback would violate the no-fallback invariant.
- **Alternative considered**: construct the scorer inside `run_cycle`.
  Rejected: the use case must not import infrastructure; the scorer is an
  outbound port supplied via dependency injection.
- **Alternative considered**: pass `candidates` per-example through the
  evaluation context. Rejected: the candidate set is cycle-wide
  configuration, not per-example data; per-example passing would imply
  that different examples could have different candidate sets, which is not
  the case.

### Decision: Use `tuple[Candidate, ...]` for candidates

`candidates` is typed as `tuple[Candidate, ...]`, not `list[Candidate]`.
`ArgmaxClassificationPolicy` resolves ties by original order, so candidates
are an ordered immutable input. A `tuple` makes this explicit at the type
level and prevents accidental mutation of the candidate sequence after
construction. The composition root creates the tuple via
`tuple(Candidate(v) for v in ...)`.

- **Alternative considered**: use `list[Candidate]` with a documented
  non-mutation constraint. Rejected: the type system should enforce
  immutability where the domain semantics require it; a doc comment is
  weaker than the type.

### Decision: One candidate-set source of truth

The optimization cycle has exactly one candidate-set source:
`[classification].candidates` in `config.toml`. The flow is:

```
[classification].candidates
        ↓
build_cli_deps
        ↓
tuple[Candidate, ...]
        ↓
RunCycleDeps
        ↓
every BaselineRunner
```

No step, runner, or evaluation context constructs or overrides the candidate
set. This prevents a future implementation from accidentally hardcoding
`BaselineRunner(..., candidates=["0", "1"])` inside `steps.py`.

### Decision: Extract shared `build_candidate_scorer` factory

The scorer construction logic currently lives inline in
`runner.py:_main_async`. Instead of copying it into `build_cli_deps`, both
composition roots SHALL call a shared `build_candidate_scorer(config)`
helper that constructs the `LLMLogitCandidateScorer` with
`ScoringPromptBuilder`, `HuggingFaceBatchTokenizerAdapter`,
`TorchBatchedCausalLMAdapter`, and `LogitScorer`. This prevents the two
wiring paths from diverging (e.g., different tokenizer or model versions).

`build_candidate_scorer` is a composition-root factory implemented outside
the application layer. It may depend on infrastructure implementations and
third-party model libraries, while callers receive the result through the
`CandidateScorer` port. The exact module location is chosen during
implementation — the important invariant is the dependency direction, not
the directory. A placement in `interfaces/` (alongside the composition
root) or `infrastructure/` are both acceptable; `application/` is not.

- **Alternative considered**: duplicate the wiring temporarily and document
  it as intentional. Rejected: duplicated composition logic is a
  maintenance hazard — the two paths can silently diverge, causing
  optimization to evaluate with different scoring infrastructure than
  production. A shared factory is a small extraction that eliminates this
  risk.

### Decision: Thread through `steps.py` to `BaselineRunner`

Each of the three `BaselineRunner` constructions in `steps.py`
(`run_active_on_dev`, `run_new_on_dev`, `run_holdout`) gains
`scorer=deps.candidate_scorer`, `policy=deps.classification_policy`, and
`candidates=list(deps.candidates)` (the runner accepts a `list`; the deps
store a `tuple`). No other step logic changes. The runner already knows how
to use them.

All three constructions receive the same scorer, policy, and candidate
instances from the single `RunCycleDeps` object — the optimization loop
does not create separate scorers per step.

### Decision: Composition root constructs scorer only for classification commands

`build_cli_deps` is only reached by CLI commands whose dependency graph
requires the classification scorer. Currently this is the `cycle` command;
the `refine` command has its own `build_refine_deps` and exits before
`build_cli_deps` is called. The scorer construction is inside
`build_cli_deps`, so the model is loaded only when a command that requires
classification is invoked. This invariant should survive future CLI
additions: any new command that does not need classification should have
its own dependency builder or exit before `build_cli_deps`.

The scorer is constructed once per process and reused across all rounds —
the optimization loop does not recreate model/tokenizer infrastructure per
example or iteration.

### Decision: Test fixtures use fake scorer + real `ArgmaxClassificationPolicy`

`tests/unit/run_cycle/conftest.py` gains a `FakeCandidateScorer` that
returns deterministic `Judgment` objects. The policy is the real
`ArgmaxClassificationPolicy` — this exercises actual selection behavior
(tie-breaking, max-score selection) without a language model. A mock policy
is used only if a specific test needs a predetermined `Classification`
result. The `build_mock_deps` fixture passes the fake scorer, real policy,
and a fixed candidate tuple `(Candidate("0"), Candidate("1"))` into
`RunCycleDeps`.

- **Alternative considered**: mock both scorer and policy. Rejected: mocking
  the policy adds a test abstraction that hides real selection behavior;
  the policy is a pure function with no external dependencies, so there is
  no cost to using the real implementation.

### Decision: `Judgment.score` is not an optimization metric

The optimization metric is computed from `ResultRow.predicted_decision` vs
`ResultRow.true_label` via the existing `compute_metrics` service, exactly
as today. `predicted_decision` is populated by the baseline runner from
`classification.selected.value` (the candidate-scoring result).
`Judgment.score` remains an opaque ranking signal. The evaluation layer
(runner → result row → metrics) determines correctness; the optimization
layer (version comparator) decides accept/rollback. This change does not
alter that boundary.

The chain is:

```
CandidateScorer.score(text, candidates)
    → Judgment[]
    → ClassificationPolicy.classify(judgments)
    → Classification
    → ResultRow.predicted_decision
    → compute_metrics(predicted_decision, true_label)
    → optimization comparator
```

## Risks / Trade-offs

- **[Breaking `RunCycleDeps` construction]** Any code that constructs
  `RunCycleDeps` without the three new fields will fail at construction
  time.
  → Mitigation: the only production call site is `build_cli_deps`; the only
  test call sites are in `conftest.py`. Both are updated in the same change.
  A `TypeError` at construction is the desired failure mode for a missing
  scorer.

- **[Model loading in composition root]** Wiring `LLMLogitCandidateScorer`
  in `build_cli_deps` means the model is loaded when the CLI starts the
  `cycle` command, not lazily on first classification.
  → Mitigation: `build_cli_deps` is called only for the `cycle` command;
  the `refine` command has its own dependency builder. The model is loaded
  once and reused. Future lazy-loading is a separate optimization.

- **[Shared factory placement]** The `build_candidate_scorer` factory
  imports `transformers` and infrastructure scoring modules, so it cannot
  live in `application`.
  → Mitigation: it is a composition-root factory outside the application
  layer — placement in `interfaces/` or `infrastructure/` are both
  acceptable. The application layer never imports it. The exact location is
  chosen during implementation.

## Migration Plan

1. Extract `build_candidate_scorer(config)` from the inline wiring in
   `runner.py:_main_async`.
2. Refactor `runner.py:_main_async` to call the shared factory.
3. Add the three fields to `RunCycleDeps`.
4. Thread them through `steps.py` to the three `BaselineRunner`
   constructions.
5. Wire them in `build_cli_deps` (`interfaces/cli/main.py`) using the
   shared factory.
6. Update `conftest.py` with fake scorer, real policy, and candidate tuple.
7. Run `pytest`, `mypy`, `ruff check`, and `lint-imports` to verify no
   regressions.

No runtime fallback is required. If the change must be reverted, use a
repository-level revert of the migration commit. No alternative
classification path is introduced.

## Acceptance Criteria

- [ ] `RunCycleDeps` requires `CandidateScorer`.
- [ ] `RunCycleDeps` requires `ClassificationPolicy`.
- [ ] `RunCycleDeps` has one explicitly defined classification candidate
      set as `tuple[Candidate, ...]`.
- [ ] `build_cli_deps` derives the candidate set from the configured
      classification candidates and constructs the scorer and policy from
      the same configuration used by the production classification path.
- [ ] The scorer is constructed via a shared `build_candidate_scorer`
      factory, not duplicated wiring.
- [ ] The baseline runner CLI also uses `build_candidate_scorer`.
- [ ] The scorer is constructed only in the composition path for commands
      that require classification/optimization.
- [ ] `run_active_on_dev` passes scorer, policy, and candidates to
      `BaselineRunner`.
- [ ] `run_new_on_dev` passes scorer, policy, and candidates to
      `BaselineRunner`.
- [ ] `run_holdout` passes scorer, policy, and candidates to
      `BaselineRunner`.
- [ ] All `BaselineRunner` instances created during one optimization cycle
      receive the same scorer, policy, and candidate instances from
      `RunCycleDeps`.
- [ ] No `BaselineRunner` construction inside `run_cycle` omits scoring
      dependencies.
- [ ] The optimization cycle has exactly one candidate-set source
      (`[classification].candidates`); no step or runner constructs or
      overrides it.
- [ ] No infrastructure scoring class is imported by `run_cycle`
      application code.
- [ ] Optimization metrics remain based on `predicted_decision` vs
      `true_label` via the existing `compute_metrics` service.
- [ ] `Judgment.score` is not used as the optimization metric.
- [ ] Existing optimization behavior (iteration, rollback, reporting, state
      dump, resumability, stopping conditions) is unchanged.
- [ ] Unit tests run without loading a language model.
- [ ] Unit tests use the real `ArgmaxClassificationPolicy`.
- [ ] No runtime fallback to generative classification exists.
- [ ] Full test suite passes.

## Final Architecture

```
                         Composition Root
                               │
                    build_candidate_scorer()
                               │
                               ▼
                    LLMLogitCandidateScorer
                               │
                               │ CandidateScorer
                               ▼
                         RunCycleDeps
                    ┌──────────┼──────────┐
                    │          │          │
                 scorer      policy   candidates
                    │          │          │
                    └──────────┼──────────┘
                               ▼
                         run_cycle/steps
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          BaselineRunner  BaselineRunner  BaselineRunner
             dev-active      dev-new        holdout
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    CandidateScorer.score()
                               │
                               ▼
                           Judgment[]
                               │
                               ▼
                     ClassificationPolicy
                               │
                               ▼
                        Classification
                               │
                               ▼
                       ResultRow / metrics
                               │
                               ▼
                    Optimization comparator
```

## Open Questions

(None — the wiring pattern is already established in the baseline runner
CLI; this change extracts it into a shared factory and calls it from the
composition root.)
