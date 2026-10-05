# Proposal

## Why

The pipeline's extraction signal depends entirely on the Tiny model's
`theses_raw` / `theses_norm` output (`runner.py:98`), which is noisy: it mixes
informative extractive theses with generic phrases and interpretive conclusions,
and its phrasing is unstable across minor input variations. These weaknesses
propagate downstream into the thesis bank (`stage1-thesis-analyzer`) and into
rule formulation (`stage3-prompt-composer-v2`). A larger teacher model can
review the Tiny model's candidates, drop noise, reformulate unstable phrasings
into stable extractive forms, and add missed theses — strengthening the signal
that reaches the collection without replacing the Tiny model at inference time.

## What Changes

- Add a new component (`thesis_refiner.py`) that loads thesis candidates from a
  `stage1-baseline-runner` results artifact (`theses_raw` / `theses_norm` per
  example) and asks a teacher model to refine them.
- The teacher model is reached via a separate, configurable endpoint
  (`teacher.endpoint`, `teacher.model_name`), distinct from the working model's
  `[llm]` endpoint. Initialization fails with an error naming the missing key
  when `teacher.endpoint` is absent.
- For each example, the teacher model filters noisy theses (generic phrases
  applicable to any text), filters or reformulates interpretive theses into
  extractive form, reformulates unstable phrasings while preserving meaning and
  not exceeding the original length, and — when allowed — adds theses the Tiny
  model missed, tagging them `source=teacher`.
- Original Tiny-model theses are preserved verbatim in `theses_raw_tiny`; the
  refined result is stored separately in `theses_refined`. The source artifact
  is never mutated.
- Persist the refined result as a jsonl artifact named
  `theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl`, with one record
  per example carrying `id`, `theses_raw_tiny`, `theses_refined`, `filtered_out`
  (thesis + reason), and `added` (teacher-added theses). The artifact is
  reloadable without re-calling the teacher model.
- A teacher-model failure on one example does not abort the run: the example
  keeps its original Tiny theses unchanged and processing continues. A summary
  at the end reports counts of processed, teacher-errors, filtered,
  reformulated, and added theses plus aggregate token/latency cost.
- Per-call accounting (latency, prompt/completion tokens, status) is logged for
  every teacher call; totals are included in the summary.
- All refinement operations (load candidates, teacher call, filter,
  reformulate, add, write artifact) are appended to an append-only JSONL log
  with timestamp and example id.
- Add a `[teacher]` section to `config.toml` with `endpoint` (required),
  `model_name` (required), `temperature` (default 0), `max_tokens` (default
  512), `timeout` (default 60), `max_retries` (default 3), `use_refined`
  (default true), `filter_noisy` (default true), `filter_interpretive` (default
  true), and `allow_additions` (default true).
- Downstream consumers (`stage1-thesis-analyzer`) use `theses_refined` by default
  when the artifact is available, falling back to `theses_raw_tiny` otherwise.
  The `use_refined` flag configures this. Adoption by the analyzer is a
  follow-up change; this change delivers the refiner in isolation.

## Capabilities

### New Capabilities
- `stage4-teacher-refinement`: Refines Tiny-model thesis candidates using a
  larger teacher model over a separate endpoint — filtering noisy and
  interpretive theses, reformulating unstable phrasings, adding missed theses
  with source tagging, preserving original Tiny theses, persisting a reloadable
  jsonl artifact, isolating per-example teacher failures, accounting cost, and
  logging all operations.

### Modified Capabilities
<!-- None — this change introduces a new standalone refinement component. It
     does not alter the requirements of the baseline runner or thesis analyzer;
     those components will consume the refined artifact in follow-up changes. -->

## Impact

- **New code**: `thesis_refiner.py` (component + CLI), mirroring the structure of
  `thesis_analyzer.py` and `prompt_composer.py` (dataclass config from TOML,
  artifact load/validate, argparse CLI). Reuses `AsyncTask` from `asyncTask.py`
  as the transport layer for teacher-model calls and `CallResult` / `RawResponse`
  for latency/usage accounting.
- **Config**: new `[teacher]` section in `config.toml`.
- **Dependencies**: no new third-party dependencies. Reuses `asyncTask.py`
  (aiohttp, pydantic) already required by the runner.
- **Artifacts**: new `theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl`
  (the refined artifact) and an append-only refinement log (path configurable).
- **Upstream**: consumes `results_*.jsonl` artifacts from
  `stage1-baseline-runner` (`runner.ResultRow` with `theses_raw`, `theses_norm`,
  `extract_status`).
- **Downstream**: `stage1-thesis-analyzer` will read `theses_refined` in
  preference to `theses_raw` when the artifact is available and `use_refined` is
  true. Adoption is a follow-up change; this change delivers the refiner in
  isolation.
