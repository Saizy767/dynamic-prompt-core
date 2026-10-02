# Proposal

## Why

Stage 1 (baseline run) needs to run the prepared dataset (dev and holdout) through
the model with a fixed extraction prompt and the current classification prompt
version, collect every call's parsed output, raw response, latency, and parse
status into a table, and persist it as a jsonl artifact. This artifact is the
input for the metrics stage and the thesis-analysis stage. No runner exists yet:
`smoke_test.py` makes a few ad-hoc calls, and `AsyncTask.classify` /
`extract_theses` return only the parsed model, discarding the raw content,
latency, and parse status the result schema requires.

## What Changes

- **Rich-result transport API (additive)**: extend `AsyncTask` with methods that
  return a `CallResult` containing the parsed model, raw content, `latency_ms`,
  `parse_status`, transport `status`, `finish_reason`, and `error`, while still
  writing the existing jsonl request log. The current `classify` /
  `extract_theses` / `classify_many` / `extract_theses_many` wrappers stay
  unchanged (backwards-compatible); the new methods are additive siblings.
- **Baseline runner**: a new component that, for each dataset example, performs
  an `extract_theses` call (fixed extraction prompt) and a `classify` call
  (current classification prompt version) via the rich-result API, and assembles
  a result row with `id`, `text`, `true_label`, `predicted_decision`,
  `confidence`, `theses_raw`, `theses_norm`, `extract_status`, `classify_status`,
  `extract_latency_ms`, `classify_latency_ms`, `raw_extract`, `raw_classify`.
- **Thesis normalization**: lemmatize (pymorphy3 for Russian, spaCy for
  English), lower-case, collapse whitespace, preserving negation particles
  (`не`, `нет`, `без`, `никогда`). Raw theses are kept unchanged in
  `theses_raw`.
- **Checkpoints**: append-only jsonl checkpoint written after each example so an
  interrupted run resumes processing only the remaining examples.
- **Run artifact**: full results saved as
  `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl`, loadable by the
  metrics and analyzer stages without post-processing.
- **Batch execution**: concurrency-limited batch via the rich-result batch
  methods, with a sane default for local llama.cpp (configurable).
- **Failure handling**: per-example errors (parse failure, timeout, HTTP error)
  are recorded in the result row; the run continues. A summary (ok, failed,
  network_error, timeout) is printed at the end.

## Capabilities

### New Capabilities

- `stage1-baseline-runner`: Runs the prepared dataset (dev or holdout) through
  `extract_theses` (fixed prompt) and `classify` (current prompt version),
  normalizes theses, checkpoints progress, and persists a jsonl results artifact
  for the metrics and thesis-analysis stages.

### Modified Capabilities

- `llm-transport-client`: Additive rich-result return path. New methods return a
  `CallResult` (parsed model + raw content + latency + parse_status + transport
  status) and write the existing jsonl request log. No change to existing
  `classify` / `extract_theses` / `classify_many` / `extract_theses_many`
  signatures or behavior.

## Impact

- **New module(s)**: a runner module (plus a thesis-normalization helper). Exact
  layout decided in design.md.
- **`asyncTask.py`**: additive methods and a `CallResult` dataclass. Existing
  wrappers and `RawResponse` are unchanged.
- **`config.toml`**: optional new `[runner]` section (concurrency, checkpoint
  interval, output dir, prompt version). No change to existing keys.
- **Dependencies**: new runtime deps `pymorphy3` (Russian lemmatization) and
  `spacy` (English lemmatization) + a small spaCy language model. Both are
  loaded lazily so the runner imports without them until normalization runs.
- **API compatibility**: purely additive. Existing transport wrappers are
  untouched; the runner only consumes the new methods and the existing
  `dataset-and-prompt` artifact, prompts, and schemas.
