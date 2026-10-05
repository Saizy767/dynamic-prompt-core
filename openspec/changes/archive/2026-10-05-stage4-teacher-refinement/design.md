# Design

## Context

The baseline runner (`runner.py`) writes one `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl`
artifact whose rows are `ResultRow` dicts (`runner.py:91`) carrying, per
example, `id`, `text`, `true_label`, `theses_raw` (Tiny model output),
`theses_norm` (normalized), and `extract_status`. The thesis analyzer
(`thesis_analyzer.py:87` `load_results`) consumes these rows and requires
`theses_norm` / `theses_raw` as lists of strings (`thesis_analyzer.py:39`).

Every stage component follows the same shape: a `*Config` dataclass with
`from_config()` reading a TOML section via `tomllib`, a `*Error(ValueError)`
exception, artifact load/validate helpers, and an argparse CLI. Artifacts are
jsonl, one record per example, with filenames carrying `run_id`,
`prompt_version`, and `timestamp` (see `thesis_analyzer.py:395`,
`prompt_composer.py`).

The transport layer `AsyncTask` (`asyncTask.py:286`) is an OpenAI-compatible
async client. Its constructor accepts `endpoint`, `completion_timeout`,
`max_retries`, `temperature`, `max_tokens`, and `truncate_tokens`, but reads
`[llm].model_path` (tokenizer) and `[llm].served_model_name` (the `model` field
sent in chat payloads) from the `[llm]` config section. A `CallResult`
(`asyncTask.py:98`) bundles `latency_ms`, `status`, and `parse_status`; the
underlying `RawResponse.usage` (`asyncTask.py:91`) carries prompt/completion
token counts from the server response.

## Goals / Non-Goals

**Goals:**
- Strengthen the thesis signal reaching the collection by filtering noise and
  reformulating unstable/interpretive theses via a larger teacher model.
- Preserve the original Tiny-model theses verbatim so the change is reversible
  and comparable.
- Isolate per-example teacher failures so one bad call never aborts the run.
- Produce a reloadable artifact so refinement is a one-time cost per run, not
  paid on every downstream read.
- Account teacher cost (tokens, latency) so the operator can budget.

**Non-Goals:**
- Replacing the Tiny model at inference time (the teacher only refines offline).
- Training or fine-tuning a refinement model.
- Automatically optimizing the teacher's prompt.
- Changing the baseline runner's extraction or the thesis analyzer's clustering
  algorithm (adoption of `theses_refined` by the analyzer is a follow-up change).
- Concurrent multi-process refinement (single-process, like every other stage).

## Decisions

### D1: Single-module component (`thesis_refiner.py`)
Mirrors `thesis_analyzer.py` / `prompt_composer.py`: dataclass
`TeacherRefinerConfig.from_config`, `ThesisRefiner` class driving the loop,
`ThesisRefinerError(ValueError)` exception, a pydantic model for the teacher
response, and an argparse CLI. The `ThesisRefiner` holds the config, the
`AsyncTask` instance for the teacher endpoint, and aggregate counters.

**Alternative**: split the teacher client, the refinement logic, and the
artifact writer into three modules. Rejected — every other stage is one module;
the total surface is small enough to keep together.

### D2: Reuse `AsyncTask` as the transport, override the served model name
The refiner constructs `AsyncTask(endpoint=teacher.endpoint,
completion_timeout=teacher.timeout, max_retries=teacher.max_retries,
temperature=teacher.temperature, max_tokens=teacher.max_tokens,
truncate_tokens=2000)`. Because `AsyncTask` reads `served_model_name` from
`[llm]` (the working model), the refiner sets `task._served_model_name =
teacher.model_name` after construction so the teacher endpoint receives the
correct model field. The tokenizer loaded from `[llm].model_path` is used only
for input truncation (a safety bound); `truncate_tokens=2000` keeps it from
firing on the short example texts we process. Token accounting comes from the
server's `usage` response, not the tokenizer, so per-call and summary token
counts are accurate for the teacher.

**Alternative**: make direct `aiohttp` calls to the teacher endpoint, bypassing
`AsyncTask`. Rejected — it would duplicate retry, timeout, backoff, and
structured-output handling already in `AsyncTask`, and the only coupling is the
single `_served_model_name` override.

**Alternative**: extend `AsyncTask` with a `served_model_name` constructor
parameter. Deferred — that is a cleaner long-term fix but modifies a shared
module outside this change's scope; the override is minimal and documented.

### D3: One teacher call per example, structured response
The refiner sends one call per example containing the example `text` and its
`theses_raw` list, and asks the teacher to return a structured JSON object with
four arrays: `keep` (theses to preserve as-is), `reformulate` (pairs of
`from` → `to`), `drop` (theses to exclude with a `reason`), and `add` (new
theses). This gives the teacher full example context to judge noise and
relevance, and minimizes call count (one call per example, not one per thesis).

The response is validated against a pydantic model (`RefinementPlan`) via
`AsyncTask`'s structured-output path. If validation fails after retries, the
example falls through the failure path (D7).

**Alternative**: one call per thesis. Rejected — N× more calls for no context
gain, and the teacher cannot judge a thesis's relevance without the full text
anyway.

### D4: `theses_refined` is a list of strings for downstream compatibility
`theses_refined` is a flat list of thesis strings (kept + reformulated `to` +
added), in a stable order: kept first (in original order), then reformulated
(in original order), then added. This makes the refined artifact a drop-in
replacement for `theses_norm` / `theses_raw` downstream, where
`thesis_analyzer.load_results` expects lists of strings. Provenance is tracked
separately: `filtered_out` lists dropped theses with `reason`, and `added`
lists teacher-added theses with `source="teacher"`. Reformulated theses carry
their original in `theses_raw_tiny`; the mapping is recoverable by comparing
`theses_raw_tiny` against `theses_refined`.

**Alternative**: make `theses_refined` a list of `{text, source}` objects.
Rejected — breaks downstream compatibility with `thesis_analyzer`, which
expects strings; provenance is already captured in `added` and `filtered_out`.

### D5: Length enforcement on reformulation
After the teacher returns a reformulation `to`, the refiner checks
`len(to) <= len(from)`. If the reformulation is longer, the refiner keeps the
original thesis instead (and logs the length violation). This guarantees the
spec's length invariant without trusting the teacher to self-limit.

**Alternative**: truncate the reformulation to the original length. Rejected —
truncation can produce a malformed or meaningless fragment; keeping the original
is safer and still preserves meaning.

### D6: Artifact format and naming — jsonl, one record per example
The artifact is jsonl, one record per example:
`{id, theses_raw_tiny, theses_refined, filtered_out, added}`. Filename:
`theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl` in the configured
`output_dir`, matching the naming convention of `thesis_analyzer` and
`prompt_composer` artifacts. The artifact is written once at the end of the run
(streamed line-by-line as each example completes, then the file is closed). It
is reloadable: a `load_refined(path)` helper reads it back into a list of
records without any teacher calls.

**Alternative**: one JSON file with a top-level list. Rejected — every other
stage uses jsonl for streaming and line-by-line validation; consistency matters.

### D7: Per-example failure isolation — keep originals, continue
The refinement loop wraps each example's teacher call in a try/except. On any
exception (network, timeout, parse failure, schema mismatch after retries), the
refiner records the example with `theses_refined = theses_raw_tiny`,
`filtered_out = []`, `added = []`, increments the `teacher_errors` counter, logs
the failure with the example id and error text, and continues. The run never
aborts on a single example. Only a missing config key (D9) or an unreadable
source artifact aborts at startup.

**Alternative**: fail fast on the first teacher error. Rejected — the spec
explicitly requires continuation; one bad example should not waste an entire
run's teacher budget.

### D8: Cost and latency accounting — per-call and aggregate
Each teacher call's `RawResponse.usage` (`prompt_tokens`, `completion_tokens`)
and `latency_ms` are appended to the refinement log with the example id and
status. The refiner maintains running totals (`total_prompt_tokens`,
`total_completion_tokens`, `total_latency_ms`). The end-of-run summary prints
all five refinement counters (processed, teacher_errors, filtered,
reformulated, added) plus the three aggregate cost figures.

**Alternative**: compute totals from the log at the end. Rejected — keeping
running totals is O(1) memory and lets the summary print even if the log write
fails.

### D9: Config dataclass — `[teacher]` section
`TeacherRefinerConfig.from_config(config_path)` reads `[teacher]` from
`config.toml` via `tomllib`, mirroring `ThesisAnalyzerConfig`. Keys:
`endpoint` (required — raises `ThesisRefinerError` naming the missing key),
`model_name` (required), `temperature` (default 0), `max_tokens` (default 512),
`timeout` (default 60), `max_retries` (default 3), `use_refined` (default
true), `filter_noisy` (default true), `filter_interpretive` (default true),
`allow_additions` (default true), `output_dir` (default `data/results`),
`log_path` (default `data/thesis_refiner.jsonl`). Defaults are module-level
`DEFAULT_*` constants. When `filter_noisy` / `filter_interpretive` /
`allow_additions` are false, the corresponding teacher instruction is omitted
and the relevant result arrays are ignored.

**Alternative**: derive `endpoint` from `[llm]`. Rejected — the teacher is a
distinct model on a distinct endpoint by design; sharing would defeat the
purpose.

### D10: Append-only refinement log
`log_path` (default `data/thesis_refiner.jsonl`) receives one JSON object per
line: `{timestamp, operation, id, details}`. Operations: `load_candidates`,
`teacher_call`, `filter`, `reformulate`, `add`, `write_artifact`, `skip`,
`teacher_error`. The log is opened in append mode (`"a"`) per write. Log write
failures are caught and printed to stderr, never blocking refinement.

**Alternative**: reuse the runner's request log. Rejected — different scope and
shape; the refinement log records semantic operations, not raw HTTP requests.

### D11: Downstream selection — `use_refined` flag + artifact presence
The artifact is the source of truth for refined theses. Downstream consumers
check: if `use_refined` is true and a `theses_refined_*.jsonl` artifact exists
for the run, use `theses_refined`; otherwise use `theses_raw_tiny`. This logic
lives in the consumer (follow-up change); this change only produces the artifact
and the flag. Rollback is therefore operator-driven: delete or rename the
refined artifact, or set `use_refined=false`.

**Alternative**: write a pointer file recording which theses to use. Rejected —
file presence + a config flag is simpler and already supports both directions.

## Risks / Trade-offs

- **[Teacher cost]** → One call per example with the full text can be expensive
  on a large dev split. Mitigation: the artifact is written once per run and
  reused on every downstream read; the summary surfaces total tokens so the
  operator can budget; the dev split is small (<200 examples).
- **[AsyncTask coupling to `[llm]`]** → The refiner overrides
  `_served_model_name` after construction (D2). Mitigation: the override is a
  single documented line; a future change can add a constructor parameter to
  `AsyncTask` to remove it.
- **[Tokenizer mismatch]** → The tokenizer loaded from `[llm].model_path` is the
  working model's, not the teacher's; truncation token counts are approximate.
  Mitigation: `truncate_tokens=2000` avoids truncation on short texts; token
  accounting uses the server's `usage`, not the tokenizer, so reported costs are
  exact.
- **[Teacher adds bad theses]** → The teacher may add theses that are not
  actually present in the text. Mitigation: added theses are tagged
  `source=teacher` in the artifact, so they can be audited and filtered
  downstream; `allow_additions=false` disables the feature entirely.
- **[Reformulation changes meaning]** → The teacher may reformulate a thesis into
  something with different meaning. Mitigation: the original is always preserved
  in `theses_raw_tiny`; setting `use_refined=false` restores the originals
  without re-running the teacher.
- **[Single-process]** → No concurrent refinement across processes. Mitigation:
  the cycle orchestrator is the single driver; the refiner is called once per
  run.

## Open Questions

- **Which model serves as teacher?** A 7B/13B/70B local model or an external API
  endpoint. Deferred — the endpoint is config-driven, so this is an operational
  choice, not a code change.
- **Separate teacher server or the same launcher with a different model?**
  Deferred — `server_launcher.py` currently launches one model; running the
  teacher is an operational setup concern outside this component's scope.
- **Refinement once per cycle or after every round?** Deferred — the cycle
  orchestrator decides when to call the refiner; this change delivers the
  component, not the scheduling.
- **Is there a total token budget per cycle?** Deferred — the summary reports
  total tokens; enforcing a hard budget is a follow-up if cost becomes a
  problem.
