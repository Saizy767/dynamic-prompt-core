# Tasks

## 1. Pydantic schemas

- [x] 1.1 Create `schemas.py` with `ClassificationResult(BaseModel)` having `decision: Literal[0, 1]` and `confidence` constrained to 0–100. Verify `ClassificationResult.model_validate_json('{"decision": 1, "confidence": 85}')` passes and `decision=2` raises `ValidationError`.
- [x] 1.2 Add `ThesisExtraction(BaseModel)` with `theses: list[str]` bounded to 3–5 items, and a `@field_validator` enforcing 2–6 words per thesis (whitespace split). Verify a 4-item, 4-word-each list validates and an 8-item list raises `ValidationError`.
- [x] 1.3 Verify both schemas produce a JSON schema via `model_json_schema()` compatible with `asyncTask._schema_for` (name, schema, strict) so `AsyncTask.classify` / `AsyncTask.extract_theses` accept them.

## 2. Prompt store and renderer

- [x] 2.1 Create `prompts.py` with a dependency-free `render(layers) -> str` that joins the five layers (role, task, rules, output contract, fallback) with section separators. Verify rendering a sample five-layer prompt returns the expected concatenated text.
- [x] 2.2 Add a `PromptArtifact` dataclass with `version`, `layers`, `text` (rendered), and `sha256` (`hashlib.sha256(text)[:16]`). Verify two artifacts built from identical layers have equal `sha256`.
- [x] 2.3 Define the classification prompt v0 as five layers with 3–5 rules. Add an assembly check that raises when the rules count is outside 3–5. Verify v0 assembles successfully and that a 2-rule or 6-rule variant raises.
- [x] 2.4 Define the fixed extraction prompt as a constant `PromptArtifact` (version `"extract-v0"`, frozen text body, not assembled from layers). Verify it carries a stable version and hash across repeated imports.

## 3. Dataset loader

- [x] 3.1 Create `dataset.py` with a loader that detects format by extension and reads jsonl (standard library `json` per line). Verify a valid jsonl file with `id`, `text`, `label` loads with record count equal to line count.
- [x] 3.2 Add parquet support via a lazy `pyarrow`/`pandas` import that raises a clear `ImportError` naming the missing package when a `.parquet` path is given without the dependency. Verify the error message names the install command.
- [x] 3.3 Add per-record validation for required fields (`id`, `text`, `label`) and `label` in `{0, 1}`. Verify a missing `text` raises an error naming the record `id` and the missing field, and an invalid `label` raises naming the `id` and the invalid value.

## 4. Fixed dev/holdout/ambiguous split

- [x] 4.1 Implement the split: sort non-ambiguous records by `id`, `random.Random(seed).shuffle`, slice at `int(len * holdout_ratio)`. Assert `0 < cut < len` and raise a clear error otherwise. Verify two loads with the same seed produce identical `dev` and `holdout` sets.
- [x] 4.2 Separate `ambiguous=true` records into a distinct group excluded from `dev` and `holdout`. Verify records with `ambiguous=true` appear in neither `dev` nor `holdout`.
- [x] 4.2.1 Document that holdout metrics are logged but MUST NOT drive Stage 1 accept/reject decisions (enforced by convention in the runner, not by this module). Verify the split output exposes `dev`, `holdout`, and `ambiguous` as disjoint groups.

## 5. Dataset artifact persistence

- [x] 5.1 Implement artifact writing to jsonl (or parquet) with fields `id`, `text`, `label`, `split` (`dev`/`holdout`/`ambiguous`), `notes`. Use filename `prepared_<seed>_<YYYYMMDDTHHMMSSZ>.jsonl`. Verify the file is written and its name contains the seed and a timestamp.
- [x] 5.2 Implement artifact reload that re-derives `dev`, `holdout`, `ambiguous` from the `split` field. Verify reloading an artifact reproduces the original three-way split exactly.

## 6. Config integration

- [x] 6.1 Read `[dataset].path`, `[dataset].seed` (default `42`), and `[dataset].holdout_ratio` (default `0.2`) from `config.toml` using `tomllib`. Verify defaults apply when the `[dataset]` section is absent and explicit values are used when present.
- [x] 6.2 Add the `[dataset]` section to `config.toml` pointing at the prepared jsonl path with the chosen seed and ratio. Verify `asyncTask.py` still loads its `[llm]` section unchanged.

## 7. Source CSV → jsonl conversion

- [x] 7.1 Add a one-time conversion step (script or documented snippet) that maps `data/train.csv` columns `id`→`id`, `text`→`text`, `target`→`label` to jsonl. Verify the output jsonl loads through the dataset loader with no validation errors.

## 8. Integration verification

- [x] 8.1 Verify `ClassificationResult` and `ThesisExtraction` are accepted by `AsyncTask.classify` / `AsyncTask.extract_theses` signature-wise (model param typed `Type[BaseModel]`) and that `_schema_for` produces a valid structured-output schema for each.
- [x] 8.2 Verify the prepared dataset artifact + prompt v0 + extraction prompt + schemas together form a usable starting point: loading the artifact, rendering v0, and fetching the extraction prompt all succeed without a running server.
