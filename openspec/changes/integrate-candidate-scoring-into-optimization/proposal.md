# Proposal

## Why

The optimization cycle (`run_cycle`) constructs `BaselineRunner` without
supplying the `CandidateScorer`, `ClassificationPolicy`, and candidate set
that the runner now requires. `RunCycleDeps` does not carry these
dependencies, and the composition root (`build_cli_deps`) does not wire
them. The optimization cycle therefore cannot execute the same
candidate-scoring classification path that production classification uses.
This change wires the existing candidate-scoring dependency graph into the
optimization composition root so every evaluation round classifies through
`CandidateScorer.score` → `ClassificationPolicy.classify`, removing the
last unwired gap in the classification migration.

## What Changes

- Add `candidate_scorer: CandidateScorer`, `classification_policy:
  ClassificationPolicy`, and `candidates: tuple[Candidate, ...]` to
  `RunCycleDeps` (**BREAKING** for callers that construct `RunCycleDeps`
  without these fields). The candidates are cycle-wide classification
  candidates with the same lifetime as `RunCycleDeps`; they are not
  per-example or per-task data. A `tuple` is used instead of a `list` to
  make ordering and immutability explicit at the type level, since
  `ArgmaxClassificationPolicy` resolves ties by original order.
- Thread `deps.candidate_scorer`, `deps.classification_policy`, and
  `deps.candidates` through every `BaselineRunner` construction in
  `run_cycle/steps.py` (`run_active_on_dev`, `run_new_on_dev`,
  `run_holdout`). All `BaselineRunner` instances created during one
  optimization cycle MUST receive the same scorer, policy, and candidate
  instances from `RunCycleDeps`.
- Extract a shared scorer construction helper (`build_candidate_scorer`)
  implemented outside the application layer so the baseline runner CLI
  (`runner.py:_main_async`) and the cycle composition root (`build_cli_deps`
  in `interfaces/cli/main.py`) share one wiring path instead of duplicating
  it. Wire `ArgmaxClassificationPolicy` and the candidate tuple in the
  composition root.
- Update `run_cycle` unit-test fixtures (`conftest.py`) to include a fake
  `CandidateScorer` and the real `ArgmaxClassificationPolicy` so
  optimization tests run without a language model while still exercising
  the actual selection behavior.
- Remove any remaining optimization-cycle code paths that bypass the
  candidate-scoring classification flow.

## Capabilities

### New Capabilities

(None — no new capability is introduced. The candidate-scoring contracts and
flow already exist as `candidate-scoring-contracts` and
`candidate-scoring-flow`.)

### Modified Capabilities

- `stage3-cycle-orchestrator`: The dependency object (`RunCycleDeps`) gains
  `candidate_scorer`, `classification_policy`, and `candidates`. The
  composition root wires them via a shared factory. Every `BaselineRunner`
  construction inside the round sequence receives them. The optimization
  cycle uses the same scorer implementation, policy implementation, and
  candidate configuration as production classification.

## Impact

- **Code**: `RunCycleDeps` gains three required fields (breaking for direct
  constructors). `run_cycle/steps.py` passes them to `BaselineRunner`.
  A shared `build_candidate_scorer` helper is extracted and used by both
  the baseline runner CLI and `build_cli_deps`. `run_cycle` test fixtures
  gain a fake scorer and real policy.
- **Specs**: `stage3-cycle-orchestrator` delta updates the dependency-object,
  composition-root, and round-sequence requirements.
- **Dependencies**: No new runtime dependencies. The scorer infrastructure
  (`LLMLogitCandidateScorer`, tokenizer, model adapter) is already
  implemented; this change reuses it from the composition root via the
  shared factory.
- **Public API**: `RunCycleDeps` construction changes — all call sites must
  supply the three new fields. The CLI `cycle` command is unchanged from the
  user's perspective.
- **Migration**: The scorer wiring already exists in the baseline runner CLI
  (`runner.py:_main_async`); this change extracts it into a shared helper
  and calls it from the cycle composition root. No fallback to generative
  classification is introduced.
