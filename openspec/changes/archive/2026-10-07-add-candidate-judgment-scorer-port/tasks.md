# Tasks

## 1. Domain model: Candidate

- [x] 1.1 Create `src/dynamic_prompt_core/domain/models/candidate.py` defining
  `Candidate` as a `@dataclass(frozen=True)` with a single `value: str` field.
  Implement `__post_init__` rejecting empty/blank values (raise `ValueError`).
  Store the value verbatim — do not apply `strip`, `lower`, `casefold`, or any
  normalization. Use `from __future__ import annotations` and import only
  `dataclasses` from the stdlib. Verify by running
  `python -c "from dynamic_prompt_core.domain.models.candidate import Candidate; print(Candidate('  x  '))"` and confirming the value is `'  x  '` unchanged, and that `Candidate('')` raises `ValueError`.
- [x] 1.2 Re-export `Candidate` from `src/dynamic_prompt_core/domain/models/__init__.py`
  by adding it to the import block and `__all__`. Verify by running
  `python -c "from dynamic_prompt_core.domain.models import Candidate; print(Candidate('a') == Candidate('a'))"` and confirming it prints `True`.
- [x] 1.3 Create `tests/unit/candidate_scoring/__init__.py` and
  `tests/unit/candidate_scoring/test_candidate.py` covering: valid creation,
  equality of candidates with the same value, inequality with different values,
  hashing equality for equal candidates (`hash(Candidate('a')) == hash(Candidate('a'))`),
  use as a dict key, immutability (attempting to set `value` raises
  `FrozenInstanceError`), rejection of empty/blank values, and that the value is
  stored verbatim with no auto-normalization (`Candidate('  X  ').value == '  X  '`,
  `Candidate('A') != Candidate('a')`). Verify by running
  `pytest tests/unit/candidate_scoring/test_candidate.py -v` and confirming all
  tests pass.

## 2. Domain model: Judgment

- [x] 2.1 Create `src/dynamic_prompt_core/domain/models/judgment.py` defining
  `Judgment` as a `@dataclass(frozen=True)` with `candidate: Candidate` and
  `score: float` fields. Implement `__post_init__` that rejects `NaN`,
  positive infinity, and negative infinity using `math.isnan` and `math.isinf`
  (raise `ValueError`). Do not impose a `[0, 1]` constraint and do not clamp.
  Import only `dataclasses`, `math`, and `Candidate` from the domain. Verify by
  running `python -c "from dynamic_prompt_core.domain.models.judgment import Judgment; from dynamic_prompt_core.domain.models.candidate import Candidate; print(Judgment(Candidate('a'), 1.5))"` and confirming it prints, and that
  `Judgment(Candidate('a'), float('nan'))` raises `ValueError`.
- [x] 2.2 Re-export `Judgment` from `src/dynamic_prompt_core/domain/models/__init__.py`
  by adding it to the import block and `__all__`. Verify by running
  `python -c "from dynamic_prompt_core.domain.models import Judgment, Candidate; print(Judgment(Candidate('a'), 0.0).candidate == Candidate('a'))"` and confirming it prints `True`.
- [x] 2.3 Create `tests/unit/candidate_scoring/test_judgment.py` covering:
  association of candidate and score, equality of judgments with equal candidate
  and score, inequality when candidate or score differs, immutability
  (attempting to set `candidate` or `score` raises `FrozenInstanceError`),
  acceptance of finite scores outside `[0, 1]` (e.g., `-3.2`, `42.0`),
  rejection of `NaN`, rejection of `+inf`, rejection of `-inf`, and that
  out-of-range finite scores are not clamped (the stored score equals the input).
  Verify by running `pytest tests/unit/candidate_scoring/test_judgment.py -v` and
  confirming all tests pass.

## 3. CandidateScorer application outbound port

- [x] 3.1 Create `src/dynamic_prompt_core/application/ports/outbound/candidate_scorer.py`
  defining `CandidateScorer` as a `@runtime_checkable typing.Protocol` with a
  single async method
  `async def score(self, text: str, candidates: list[Candidate]) -> list[Judgment]: ...`.
  The docstring documents the all-or-nothing contract: a successful call returns
  exactly one `Judgment` per supplied candidate; if any candidate cannot be
  evaluated the call raises an exception; partial results are not supported. It
  also documents that the scorer evaluates candidates and does not decide, rank,
  select, or aggregate. Import `Candidate` and `Judgment` from the domain. Use
  `from __future__ import annotations`. Verify by running
  `python -c "from dynamic_prompt_core.application.ports.outbound.candidate_scorer import CandidateScorer; print(CandidateScorer)"` and confirming it prints the protocol.
- [x] 3.2 In the same module, define
  `validate_unique_candidates(candidates: list[Candidate]) -> None` that raises
  `ValueError` when the list contains duplicate candidates (detected via
  `Candidate` equality/hashing) and returns `None` when all candidates are
  unique. This is the application-boundary validation called before the scorer.
  Verify by running
  `python -c "from dynamic_prompt_core.application.ports.outbound.candidate_scorer import validate_unique_candidates; from dynamic_prompt_core.domain.models.candidate import Candidate; validate_unique_candidates([Candidate('a'), Candidate('a')])"` and confirming it raises `ValueError`, and that a unique list does not raise.
- [x] 3.3 Re-export `CandidateScorer` and `validate_unique_candidates` from
  `src/dynamic_prompt_core/application/ports/outbound/__init__.py` by adding them
  to the import block and `__all__`. Verify by running
  `python -c "from dynamic_prompt_core.application.ports.outbound import CandidateScorer, validate_unique_candidates; print(CandidateScorer, validate_unique_candidates)"` and confirming both print.
- [x] 3.4 Create `tests/unit/candidate_scoring/test_candidate_scorer_port.py`
  verifying: the port is a `typing.Protocol`, `CandidateScorer` is
  `runtime_checkable`, a minimal async class with a matching `score` signature
  satisfies `isinstance(obj, CandidateScorer)`, `validate_unique_candidates`
  accepts a unique list, `validate_unique_candidates` rejects a list with
  duplicates, and `validate_unique_candidates` rejects a list with repeated
  identical candidates. Do not instantiate an actual LLM. Verify by running
  `pytest tests/unit/candidate_scoring/test_candidate_scorer_port.py -v` and
  confirming all tests pass.

## 4. Import-linter port boundary contract

- [x] 4.1 Add an `application-ports-isolated` contract to `pyproject.toml` under
  `[tool.importlinter]` (type `forbidden`,
  `source_modules = ["dynamic_prompt_core.application.ports"]`,
  `forbidden_modules = ["dynamic_prompt_core.infrastructure", "dynamic_prompt_core.interfaces"]`).
  This enforces that `application.ports` does not import `infrastructure` or
  `interfaces`, directly or indirectly. Verify by running `lint-imports` and
  confirming all contracts pass, including the new `application-ports-isolated`
  contract, with no configuration errors.

## 5. Architecture validation and behavior preservation

- [x] 5.1 Run `openspec validate --changes add-candidate-judgment-scorer-port`
  and verify the `candidate-scoring-contracts` spec delta is valid: every
  requirement has at least one scenario and the new capability has a
  `## Purpose` section.
- [x] 5.2 Run `mypy` and `ruff check` and verify they pass, confirming the new
  domain and port modules introduce no type or lint regressions.
- [x] 5.3 Run `pytest` and verify the full test suite passes, confirming the new
  domain/port tests pass and no existing classification tests regressed.
- [x] 5.4 Inspect `git diff --stat` and confirm the only modified existing files
  are `pyproject.toml` (import-linter contract), `domain/models/__init__.py`,
  and `application/ports/outbound/__init__.py` (public API re-exports), and that
  no existing classification file (`runner.py`, `client.py`, `classification.py`,
  `run_cycle`) is modified.
