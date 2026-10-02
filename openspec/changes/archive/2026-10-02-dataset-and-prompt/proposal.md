# Proposal

## Why

Stage 1 (baseline run) needs a prepared dataset, a fixed train/holdout split, a
starting classification prompt, and the Pydantic schemas that
`AsyncTask.classify` / `AsyncTask.extract_theses` consume for structured output.
None of these exist yet: `data/train.csv` is raw, there is no split artifact, no
prompt store, and `smoke_test.py` uses throwaway placeholder models. This change
provides the starting point on which the runner, metrics, and optimization cycle
will build.

## What Changes

- **Dataset loader** that reads jsonl or parquet files with required fields
  `id`, `text`, `label` (`label` in `{0, 1}`), validates every record, and names
  the first offending record on failure.
- **Fixed dev/holdout split** with a configurable seed. The split is
  reproducible: same dataset + same seed yields identical dev and holdout.
  Ambiguous examples (`ambiguous=true`) are separated into their own group and
  excluded from dev and holdout. Holdout metrics are logged but MUST NOT drive
  accept/reject decisions on Stage 1.
- **Prompt v0 for classification**: a five-layer system prompt (role, task,
  rules 3–5 items, output contract, fallback) stored as a versioned artifact
  with a content hash. Rendered to text by a dependency-free templating helper.
  Build fails if the rules count is outside 3–5.
- **Fixed extraction prompt**: a system prompt for thesis extraction, versioned
  and hashed, that MUST NOT be optimized on Stage 1 and is reused unchanged
  across all optimization rounds.
- **Pydantic schemas**: `ThesisExtraction` (list of 3–5 theses, each 2–6 words)
  and `ClassificationResult` (`decision` in `{0, 1}`, `confidence` 0–100) for
  use with `AsyncTask.classify` / `AsyncTask.extract_theses` structured output.
- **Dataset artifact persistence**: the prepared dataset (dev, holdout,
  ambiguous) is written as jsonl or parquet with fields `id`, `text`, `label`,
  `split`, `notes`. The artifact is reproducible from the source file and seed,
  and reloads to the same split.

## Capabilities

### New Capabilities

- `dataset-and-prompt`: Dataset loading, fixed dev/holdout/ambiguous split,
  baseline classification prompt v0, fixed extraction prompt, and Pydantic
  schemas for structured output. The starting point for the Stage 1 baseline
  run.

### Modified Capabilities

None. The existing `llm-transport-client` spec covers transport, parsing, and
logging; this change consumes its `classify` / `extract_theses` wrappers but
does not alter their behavior.

## Impact

- **New module(s)**: a dataset loader/preparer, a prompt store with the v0
  classification prompt and the fixed extraction prompt, and the two Pydantic
  schemas. Exact file layout is decided in design.md.
- **`data/`**: the prepared split artifact is written here (or a configured
  output dir); the source `train.csv` is read-only input.
- **`config.toml`**: optional new keys for the dataset path, split seed, and
  artifact output directory, read alongside the existing `[llm]` section. No
  change to existing keys.
- **Dependencies**: no new runtime dependencies. Pydantic v2 (already in use
  via `asyncTask.py`) provides the schemas; jsonl/parquet I/O uses the standard
  library plus any parquet reader already available (design.md decides the
  parquet path).
- **API compatibility**: purely additive. `AsyncTask` is consumed, not modified.
