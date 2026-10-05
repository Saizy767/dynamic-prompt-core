# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[teacher]` section to `config.toml` with keys `endpoint`,
  `model_name`, `temperature`, `max_tokens`, `timeout`, `max_retries`,
  `use_refined`, `filter_noisy`, `filter_interpretive`, `allow_additions`,
  `output_dir`, and `log_path`, with inline comments documenting each default
  (`endpoint` and `model_name` required; `temperature=0`, `max_tokens=512`,
  `timeout=60`, `max_retries=3`, `use_refined=true`, `filter_noisy=true`,
  `filter_interpretive=true`, `allow_additions=true`,
  `output_dir="data/results"`, `log_path="data/thesis_refiner.jsonl"`). Verify
  the file parses with `tomllib.load` and the section round-trips.
- [x] 1.2 Create `thesis_refiner.py` with module docstring,
  `from __future__ import annotations`, a `ThesisRefinerError(ValueError)`
  exception, and a `TeacherRefinerConfig` dataclass with `from_config()` reading
  `[teacher]` from `config.toml` (mirroring `ThesisAnalyzerConfig`), plus
  default constants (`DEFAULT_TEMPERATURE=0.0`, `DEFAULT_MAX_TOKENS=512`,
  `DEFAULT_TIMEOUT=60`, `DEFAULT_MAX_RETRIES=3`, `DEFAULT_USE_REFINED=True`,
  `DEFAULT_FILTER_NOISY=True`, `DEFAULT_FILTER_INTERPRETIVE=True`,
  `DEFAULT_ALLOW_ADDITIONS=True`, `DEFAULT_OUTPUT_DIR="data/results"`,
  `DEFAULT_LOG_PATH="data/thesis_refiner.jsonl"`). Verify
  `TeacherRefinerConfig.from_config()` returns configured values, falls back to
  defaults for missing optional keys, and raises `ThesisRefinerError` naming
  `endpoint` when it is absent and naming `model_name` when it is absent.

## 2. Teacher response schema

- [x] 2.1 Define a pydantic `RefinementPlan` model (in `thesis_refiner.py` or
  `schemas.py`) with four fields: `keep: list[str]`, `reformulate: list[dict]`
  (each with `from: str` and `to: str`), `drop: list[dict]` (each with `thesis:
  str` and `reason: str`), and `add: list[str]`. Verify the model validates a
  well-formed response and rejects one missing a required field or with a wrong
  type.
- [x] 2.2 Build the teacher system prompt string that instructs the model to
  return a `RefinementPlan` JSON object, conditionally including the noisy
  filter, interpretive filter, and addition instructions based on
  `filter_noisy`, `filter_interpretive`, and `allow_additions`. Verify the
  prompt omits the addition instruction when `allow_additions=false` and
  includes it when true.

## 3. Load thesis candidates

- [x] 3.1 Implement `load_candidates(path)` that reads a
  `results_*.jsonl` artifact line by line (mirroring
  `thesis_analyzer.load_results`), validates each row has `id`, `theses_raw`,
  `theses_norm`, and `extract_status`, and returns a list of candidate dicts.
  Raise `ThesisRefinerError` naming the first offending record's id and the
  missing field. Verify: a valid artifact loads all rows; a row missing
  `theses_raw` raises `ThesisRefinerError` naming the id and field.
- [x] 3.2 Implement candidate skipping: when `extract_status == "failed"`, the
  candidate is skipped (not sent to the teacher) and the skip is logged with the
  example id. Verify: a candidate with `extract_status="failed"` is excluded
  from the teacher-call list and a skip log entry is written.

## 4. Teacher-model transport

- [x] 4.1 Implement `_build_teacher_task(config)` that constructs an `AsyncTask`
  with `endpoint=config.endpoint`, `completion_timeout=config.timeout`,
  `max_retries=config.max_retries`, `temperature=config.temperature`,
  `max_tokens=config.max_tokens`, `truncate_tokens=2000`, then sets
  `task._served_model_name = config.model_name` (design D2). Verify: the
  returned task's `_endpoint` matches the teacher endpoint and
  `_served_model_name` matches `config.model_name`.
- [x] 4.2 Implement `async _call_teacher(task, session, text, theses, config)`
  that sends one call per example (design D3) with the teacher system prompt and
  a user message containing the example text and theses, expecting a
  `RefinementPlan` via `AsyncTask`'s structured-output path. Return the parsed
  `RefinementPlan` plus `latency_ms` and `usage` from the underlying
  `RawResponse`. Verify: a mocked teacher response returns the parsed plan with
  latency and usage; a malformed response raises after retries.

## 5. Refinement logic

- [x] 5.1 Implement `_apply_refinement(theses_raw, plan, config)` that builds
  `theses_refined`, `filtered_out`, and `added` from a `RefinementPlan`:
  `keep` theses go to `theses_refined` in order; `reformulate` pairs contribute
  the `to` string (after length check, task 5.3); `drop` theses go to
  `filtered_out` with their `reason`; `add` theses go to `theses_refined` and
  `added` with `source="teacher"` (only when `config.allow_additions`). Verify:
  a plan with keep=[a], reformulate=[{from:b,to:b'}], drop=[{thesis:c,reason:noise}],
  add=[d] yields `theses_refined=[a, b', d]`, `filtered_out=[{thesis:c,reason:noise}]`,
  `added=[{text:d, source:"teacher"}]`.
- [x] 5.2 Implement noisy filtering: when `config.filter_noisy=false`, `drop`
  entries are ignored and their theses are kept in `theses_refined`. Verify: with
  `filter_noisy=false`, a `drop` entry's thesis appears in `theses_refined` and
  not in `filtered_out`.
- [x] 5.3 Implement length enforcement (design D5): when a reformulation `to` is
  longer than its `from`, keep the original `from` in `theses_refined` instead
  and log the length violation. Verify: a reformulation with `len(to) > len(from)`
  keeps `from` in `theses_refined`; one with `len(to) <= len(from)` uses `to`.

## 6. Refined artifact write and reload

- [x] 6.1 Implement `write_refined_artifact(records, run_id, prompt_version,
  output_dir)` that writes a jsonl file named
  `theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl` (design D6), one
  record per line, each with `id`, `theses_raw_tiny`, `theses_refined`,
  `filtered_out`, `added`. Return the written path. Verify: the file exists at
  the expected name, each line is valid JSON with all five fields, and the
  filename contains the run_id, prompt_version, and a timestamp.
- [x] 6.2 Implement `load_refined(path)` that reads a refined artifact back into
  a list of records without any teacher calls. Verify: a freshly written
  artifact reloads to identical records; a corrupted line raises
  `ThesisRefinerError` naming the file.

## 7. Per-example failure isolation

- [x] 7.1 Implement the refinement loop's per-example try/except (design D7):
  on any exception during `_call_teacher` or `_apply_refinement`, the example
  record is written with `theses_refined = theses_raw_tiny`, `filtered_out = []`,
  `added = []`, the `teacher_errors` counter is incremented, and a
  `teacher_error` log entry is written with the id and error text. Verify: a
  teacher call that raises for id=42 produces a record with
  `theses_refined == theses_raw_tiny` for id=42 and processing continues for
  other examples.

## 8. Cost and latency accounting

- [x] 8.1 Implement per-call accounting: after each teacher call, append a log
  entry with `latency_ms`, `prompt_tokens`, `completion_tokens`, and `status`
  from the `RawResponse.usage` and response status. Maintain running totals
  (`total_prompt_tokens`, `total_completion_tokens`, `total_latency_ms`) on the
  refiner instance. Verify: two teacher calls produce two log entries with
  latency and token counts, and the instance totals equal their sum.
- [x] 8.2 Implement the end-of-run summary that prints all five refinement
  counters (`processed`, `teacher_errors`, `filtered`, `reformulated`, `added`)
  and the three aggregate cost figures (`total_prompt_tokens`,
  `total_completion_tokens`, `total_latency_ms`). Verify: after a run with 10
  processed, 1 error, 3 filtered, 2 reformulated, 1 added, the summary contains
  all five counts and the aggregate token/latency totals.

## 9. Refinement logging

- [x] 9.1 Implement `_log(operation, id, details)` that appends one JSON object
  (`{timestamp, operation, id, details}`) to `log_path` in append mode (design
  D10). A log write failure is caught and printed to stderr, never blocking
  refinement. Verify: two operations produce two lines, each valid JSON with
  `timestamp` and `operation`; a log path in a non-existent directory does not
  crash the refiner.
- [x] 9.2 Wire logging into `load_candidates` (skip), `_call_teacher`
  (`teacher_call`), `_apply_refinement` (`filter`, `reformulate`, `add`),
  `write_refined_artifact` (`write_artifact`), and the failure path
  (`teacher_error`). Verify each operation appends a log line with the correct
  `operation` name and example id.

## 10. CLI and end-to-end integration

- [x] 10.1 Implement the argparse CLI (`python thesis_refiner.py [--config
  config.toml] --results <path> [--run-id <id>] [--prompt-version <v>]`) that
  loads config, builds the teacher task, loads candidates, runs the refinement
  loop, writes the artifact, and prints the summary. Verify `--help` lists all
  flags and a run against a real results artifact produces a
  `theses_refined_*.jsonl` file.
- [x] 10.2 Run an end-to-end sequence: given a `results_*.jsonl` artifact with 5
  examples (4 ok, 1 `extract_status=failed`), run the refiner against a stubbed
  teacher endpoint. Verify: 4 examples are refined, 1 is skipped, the artifact
  contains 5 records (4 refined + 1 skipped with originals), the summary
  counters are correct, and `load_refined` on the artifact returns the same
  records without teacher calls.
- [x] 10.3 Verify failure isolation end-to-end: with a teacher endpoint that
  errors on one example id, the run completes, that example's record has
  `theses_refined == theses_raw_tiny`, the `teacher_errors` counter is 1, and
  the other examples are refined normally.
