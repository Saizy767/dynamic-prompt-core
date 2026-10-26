# Design

## Context

The current `BaselineRunner` (724 lines in `application/use_cases/run_baseline/runner.py`) interleaves two concerns in a single `_process_example` method:

1. **Extraction** (lines 434-451): calls `AsyncTask.extract_theses_detailed()` with the hardcoded `EXTRACTION_PROMPT` from `domain/prompts/fixed.py`, then normalizes theses via `normalize_theses()`.
2. **Classification** (lines 466-500): calls `CandidateScorer.score(text, candidates, rules)` → `ClassificationPolicy.classify(judgments)`, where the scorer is `LLMLogitCandidateScorer` using logit-based scoring with `ScoringPromptBuilder`, `BatchTokenizerAdapter`, `BatchedCausalLanguageModel`, and `LogitScorer`.

The current order is extraction-then-classification. Rules are extracted once at construction via `_extract_rules()`, which raises `ValueError` when `layers is None` for non-default prompts. `ResultRow` is a local dataclass with 13 fields. All prompts are hardcoded Python strings — no packaged resource files exist. `RunCycleDeps` bundles `scorer`, `policy`, `candidates`, and other ports. Three composition roots construct `BaselineRunner`: CLI main, runner CLI, and comparator CLI.

## Goals / Non-Goals

**Goals:**
- Separate classification and extraction into independent components with their own system prompts and model calls.
- Reverse the execution order: classification first, then extraction.
- Handle absent/empty rules gracefully (classification = 0 without calling Classifier) instead of raising.
- Introduce `ClassificationRules` as an immutable domain model.
- Remove `CandidateScorer` and `ClassificationPolicy` as separate architectural abstractions; replace with a single `Classifier` port.
- Preserve the logit-based scoring algorithm, `ResultRow` schema, JSONL format, and cycle behavior.

**Non-Goals:**
- Redesigning the extraction algorithm (normalization, thesis parsing, status semantics).
- Redesigning the logit scoring algorithm (causal alignment, mean log-probability, softmax).
- Changing the cycle policy, acceptance criteria, rollback logic, or rule discovery mechanism.
- Adding new fields to `ResultRow` or changing existing field names/types/semantics.
- Changing the CLI interface, config format, or external API.

## Decisions

### D1: `Classifier` port returns `Classification` directly

**Choice**: The `Classifier` port accepts `(text, ClassificationRules)` and returns `Classification`, absorbing both `CandidateScorer.score()` and `ClassificationPolicy.classify()`.

**Rationale**: The spec requires the Classifier to be "самостоятельно отвечать за полный процесс классификации" — validation, scoring, and decision. A single port that returns the final decision simplifies the runner and eliminates the two-step orchestration. `Candidate` and `Judgment` remain as internal domain models used by the implementation.

**Alternative**: Keep `CandidateScorer` returning `list[Judgment]` and have the `Classifier` wrap it. Rejected — the spec explicitly requires removing `CandidateScorer` as an architectural abstraction.

### D2: `ClassificationRules` as a separate domain model with `tuple[str, ...]`

**Choice**: New frozen dataclass in `domain/models/classification_rules.py` with a single `rules: tuple[str, ...]` field. Conversion from `PromptArtifact` handles `None` layers and empty lists by producing an empty `ClassificationRules`.

**Rationale**: The spec requires an immutable model that distinguishes absent rules (valid, empty) from invalid rules (error). Using `tuple` ensures deep immutability. The conversion function encapsulates the `PromptArtifact` → `ClassificationRules` mapping at the domain boundary.

**Alternative**: Reuse `Sequence[str]` as today. Rejected — the spec requires a dedicated domain model with explicit absence/invalidity semantics.

### D3: Execution order reversed to classification-then-extraction

**Choice**: `BaselineRunner._process_example` runs classification first, then extraction. When rules are absent/empty, classification is set to `0` (with `predicted_decision=0`, `selected_candidate="0"`, `classify_status="ok"`) without calling the `Classifier`, then extraction proceeds normally.

**Rationale**: The spec requires "Классификация должна завершаться до запуска extraction" and "Если `layers` отсутствует или `rules` — пустой список, установить классификацию в `0`, не вызывая `Classifier`". The `0` classification uses the first candidate (which is `Candidate("0")` in the default set), consistent with the existing `Classification` type.

**Alternative**: Keep extraction-then-classification order. Rejected — explicitly required to reverse by the spec.

### D4: System prompts as packaged text files under `infrastructure/llm/prompts/`

**Choice**: Two `.txt` files (`classifier.txt`, `extractor.txt`) under `infrastructure/llm/prompts/`. Each component loads its prompt at initialization via `importlib.resources` or `Path(__file__).parent / "prompts" / ...`. `pyproject.toml` gains `[tool.setuptools.package-data]` to include the files.

**Rationale**: The spec requires "два отдельных файла системных промптов" and "проверить, что файлы включаются в дистрибутив Python-пакета". Simple text format matches the existing prompt style. `importlib.resources` is the standard Python mechanism for accessing package data and works after `pip install`.

**Alternative**: Keep prompts as Python string constants. Rejected — the spec requires file-based resources with independent loading.

### D5: `Extractor` wraps existing `AsyncTask.extract_theses_detailed()` + normalization

**Choice**: The `Extractor` component encapsulates the existing extraction call (`AsyncTask.extract_theses_detailed()` with `ThesisExtraction` schema), `_try_repair()`, `_map_call_status()`, and `normalize_theses()`. It receives the same inputs as today (text, label, language) and returns `(theses_raw, theses_norm, extract_status, extract_latency_ms, raw_extract)`.

**Rationale**: The spec requires "сохранить существующую логику extraction и выделить её в самостоятельный компонент". Wrapping the existing logic minimizes risk and preserves all statuses, error handling, and normalization. The `Extractor` does not receive classification results.

**Alternative**: Redesign extraction. Rejected — "Полная переработка алгоритма extraction не входит в область работ".

### D6: `ScoringPromptBuilder` adapted, not replaced

**Choice**: The existing `ScoringPromptBuilder` is renamed/adapted to use the `Classifier`'s system prompt from the resource file as the initial prefix, followed by rules, input text, and candidate as today. The `_FIXED_INSTRUCTIONS` constant is replaced by the loaded prompt content. No separate `system` segment is introduced.

**Rationale**: The spec requires "Существующую логику `ScoringPromptBuilder` адаптировать к новой архитектуре" and "Отдельный `system`-сегмент не вводить". The scoring prefix structure is preserved; only the source of the fixed instructions changes from a hardcoded constant to a loaded resource file.

### D7: `RunCycleDeps` swaps `scorer`/`policy` for `classifier`/`extractor`

**Choice**: `RunCycleDeps` replaces `candidate_scorer` and `classification_policy` fields with `classifier` and `extractor`. The `candidates` field remains (passed to `Classifier` at construction). All three composition roots (CLI main, runner CLI, comparator CLI) are updated to construct `Classifier` and `Extractor` via a shared `build_classifier` factory.

**Rationale**: The spec requires "Получать зависимости `Classifier` и `Extractor` через существующий механизм внедрения зависимостей". The existing frozen-dataclass DI pattern is preserved. The single-factory invariant is maintained for the `Classifier`.

### D8: `domain/prompts/fixed.py` — remove only unused entities

**Choice**: `EXTRACTION_PROMPT` is removed from `fixed.py` only if the `Extractor` no longer references it (it loads from a resource file instead). `CLASSIFICATION_PROMPT_V0` is preserved if still used as the default/initial prompt version. Other code and prompts in `fixed.py` are preserved.

**Rationale**: The spec requires "Существующие сущности в `domain/prompts/fixed.py` удалять только тогда, когда они стали неиспользуемыми."

## Risks / Trade-offs

- **Reversed execution order changes timing**: Classification latency and extraction latency are now measured in a different order. Mitigation: latencies are measured independently per stage, so `classify_latency_ms` and `extract_latency_ms` remain accurate regardless of order. No downstream code depends on ordering of these measurements.

- **Absent-rules behavior change is observable**: The runner previously raised `ValueError` on `None` layers for non-default prompts; now it sets classification to `0`. Mitigation: this is an explicit requirement. Tests that assert the old `ValueError` must be updated. The holdout path (which uses the active version with valid layers) is unaffected.

- **Port removal breaks downstream tests**: Removing `CandidateScorer` and `ClassificationPolicy` breaks all tests that import or mock them. Mitigation: the task plan includes a dedicated step to update all affected tests. `FakeCandidateScorer` in `conftest.py` is replaced by a `FakeClassifier`.

- **Resource file packaging**: If `package-data` is misconfigured, prompts won't load after `pip install`. Mitigation: add a test that verifies prompt files are accessible via `importlib.resources` (not just from the repo root).

- **`ClassificationPolicy` removal affects `ClassifyInput` use case**: The `ClassifyInput` use case in `application/use_cases/classify_input/` depends on both `CandidateScorer` and `ClassificationPolicy`. Mitigation: `ClassifyInput` is adapted to use the `Classifier` port directly, or removed if `BaselineRunner` is the only caller.

## Open Questions

- **`ClassifyInput` fate**: Is `ClassifyInput` used outside `BaselineRunner`? If not, it can be removed. If yes, it should be adapted to call `Classifier.classify()`. This can be resolved during implementation by grepping for callers.
- **Classifier error type name**: The spec says to preserve existing error handling. `CandidateScoringError` is the current error type. Should it be renamed to `ClassificationError` or kept as-is? Minor; can be decided during implementation based on what reads cleaner after the refactor.
