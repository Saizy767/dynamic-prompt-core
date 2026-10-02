# Design

## Context

See proposal.md for motivation. The project has `asyncTask.py` (transport +
parsing + jsonl logging, spec `llm-transport-client`), `dataset.py` /
`prompts.py` / `schemas.py` (spec `dataset-and-prompt`), and a prepared dataset
artifact with a `split` field (`dev` / `holdout` / `ambiguous`).

The existing `AsyncTask.analyze` computes `parse_status` and writes a log entry
but returns only the parsed model (`Optional[T]`), discarding `RawResponse`
(content, latency, status) and `parse_status`. `analyze_raw` returns the
`RawResponse` but does not parse or log. The runner needs all three — parsed,
raw + latency, and parse_status — plus logging, so an additive rich-result path
is added to the transport layer (decision confirmed with the user).

`classify_many` / `extract_theses_many` create their own `aiohttp` session
internally and return `List[Optional[T]]`. They catch per-example exceptions and
fill the slot with `None`, so a batch never raises on a single failure.

## Goals / Non-Goals

**Goals:**

- Add a backwards-compatible `CallResult` rich-result API to `AsyncTask` that
  returns parsed + raw + latency + parse_status and logs.
- Run dev/holdout through `extract_theses` (fixed prompt) + `classify` (current
  prompt version) with configurable concurrency.
- Normalize theses (lemmatize, lower-case, collapse whitespace, preserve
  negation) without mutating `theses_raw`.
- Checkpoint per example so an interrupted run resumes without reprocessing
  successful examples.
- Persist a jsonl results artifact named with `run_id`, `prompt_version`,
  `split`, and `timestamp`.
- Continue past per-example failures and print an end-of-run summary.

**Non-Goals:**

- Metrics computation, thesis clustering/rule selection, prompt optimization
  (separate stages).
- Running multiple prompt versions in parallel.
- Persisting results to a database.
- Auto-detecting text language (language is config-selected).
- Dry-run mode (useful but deferred; tracked as an open question).

## Decisions

### D1: Transport enhancement — `CallResult` + `*_detailed` methods, refactor `analyze`

```python
@dataclass
class CallResult:
    parsed: Optional[BaseModel]   # validated model or None
    raw_content: Optional[str]    # raw string from the model
    latency_ms: float
    parse_status: Optional[ParseStatus]   # ok/invalid_json/schema_mismatch/truncated
    status: ResponseStatus                 # ok/empty/error/timeout
    finish_reason: Optional[str]
    error: Optional[str]
    truncated: bool
```

Add an internal `_analyze_detailed(session, text, model, call_type, ...)` that
does what `analyze` does today (call `analyze_raw`, parse, classify `parse_status`,
write log) but returns a `CallResult`. Refactor the existing `analyze` to delegate
to `_analyze_detailed` and return `result.parsed` — behavior is identical, no
duplicate parse/log logic. Add `classify_detailed` / `extract_theses_detailed`
as thin wrappers (per-call defaults) over `_analyze_detailed`, and
`classify_many_detailed` / `extract_theses_many_detailed` over a new
`_analyze_many_detailed` that mirrors `analyze_many` but collects `CallResult`.

**Rationale**: One shared parse+log path, zero behavior change for existing
wrappers, and the runner gets every field it needs in one call.

**Alternative**: Have the runner call `analyze_raw` and parse/log itself.
Rejected — duplicates the parse-status taxonomy and log-entry construction, and
`analyze_raw` does not log.

### D2: Runner module layout — `runner.py` + `normalize.py`

```
runner.py     # BaselineRunner: load split, run examples, checkpoint, artifact, summary
normalize.py  # ThesisNormalizer: lemmatize + lower-case + collapse, negation-safe
```

**Rationale**: Two focused modules. `normalize.py` is separate because it carries
the pymorphy3/spaCy dependency and is independently testable on the 5 manual
examples in the acceptance criteria. Flat layout matches the project
(`asyncTask.py`, `dataset.py`, `prompts.py`, `schemas.py`).

### D3: Per-example concurrency with semaphore, checkpoint per example

The runner submits all split examples as coroutines under
`asyncio.Semaphore(concurrency)`. Each coroutine calls
`extract_theses_detailed` + `classify_detailed` for one example, assembles the
result row, normalizes theses, and **appends the row to the checkpoint
immediately** (flush + fsync). Results are stored in a position-indexed list to
preserve input order in the final artifact.

**Rationale**: The spec requires that successful examples MUST NOT be reprocessed
after an interruption. Batch-then-checkpoint (calling `*_many_detailed` on a
chunk, then checkpointing the chunk) would reprocess every succeeded example in
an unfinished chunk on crash. Per-example checkpointing under a semaphore gives
both configurable concurrency and true per-example durability. The transport
layer's `*_many_detailed` batch methods (D1) remain available for consumers that
do not need per-example checkpointing.

**Alternative**: Chunk-based batch with `*_many_detailed`, checkpoint per chunk.
Rejected — violates "MUST NOT reprocess successful" for any chunk > 1 on
mid-chunk crash.

### D4: Checkpoint = in-progress artifact, renamed on completion

`run_id` and `timestamp` are generated at start. During the run the checkpoint
file is:

```
<output_dir>/results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl.tmp
```

Each completed example is appended as one jsonl line (flush + fsync). On
completion the file is atomically renamed to `.jsonl`. On resume, the runner
looks for a `.tmp` file matching `run_id` + `split`, reads processed IDs, and
skips them. If no `.tmp` exists, a fresh run starts.

**Rationale**: One file, no duplication between checkpoint and artifact. The
`.tmp` suffix distinguishes incomplete from complete runs. `os.replace` is
atomic on POSIX, so the final artifact is never partially written.

### D5: Thesis normalization — config-selected language, negation-protected set

```python
NEGATION_PARTICLES = {"не", "нет", "без", "никогда"}

def normalize_thesis(text: str, lang: str) -> str:
    words = text.split()
    out = []
    for w in words:
        lw = w.lower()
        if lw in NEGATION_PARTICLES:
            out.append(lw)          # pass through, never lemmatize
        else:
            out.append(lemmatize(lw, lang))
    return re.sub(r"\s+", " ", " ".join(out)).strip()
```

`lemmatize` uses `pymorphy3.MorphAnalyzer` for `lang="ru"` and a spaCy
`blank("en")` pipeline with `token.lemma_` for `lang="en"`. Both analyzers are
loaded lazily on first use and cached as module-level singletons. Language is
read from `[runner].language` (default `"ru"`).

**Rationale**: Negation particles are short, invariable words; lemmatizers may
map them to unexpected forms, so they are protected explicitly. Lower-casing
before the check makes it case-insensitive. Lazy loading keeps `runner.py`
importable without pymorphy3/spaCy installed.

**Alternative**: Auto-detect language per text. Rejected — adds a dependency and
unpredictability; the dataset language is known.

### D6: Status mapping — transport `CallResult` → runner status

| `CallResult.status` / `parse_status`         | runner status |
|-----------------------------------------------|---------------|
| `parse_status=ok`                             | `ok`          |
| `parse_status` in {`invalid_json`,`schema_mismatch`,`truncated`} and content was repaired | `repaired` |
| `parse_status` in {`invalid_json`,`schema_mismatch`,`truncated`} (no repair) | `failed` |
| `status` in {`error`,`timeout`}               | `failed`      |
| call not made (e.g. extract skipped)          | `not_attempted` |

`repaired` is set when a best-effort repair succeeded (e.g. stripping markdown
fences then re-parsing). For this change, repair is limited to stripping
```` ```json ```` fences; if that yields valid JSON, status is `repaired`,
otherwise `failed`.

**Rationale**: The spec requires four states. `repaired` captures the common
case where models wrap JSON in fences despite instructions, without conflating
it with `ok` or `failed`.

### D7: Config — `[runner]` section

```toml
[runner]
concurrency = 4            # sane for local llama.cpp on CPU
checkpoint_interval = 1    # fsync every N examples
output_dir = "data/results"
language = "ru"            # ru -> pymorphy3, en -> spaCy
prompt_version = "classify-v0"
```

Read with `tomllib` alongside the existing `[llm]` and `[dataset]` sections.
`prompt_version` selects the classification prompt artifact from `prompts.py`
(for the baseline, `CLASSIFICATION_PROMPT_V0`). The extraction prompt is always
`EXTRACTION_PROMPT` (fixed).

**Rationale**: Centralizes runner knobs in `config.toml`. `concurrency=4` is
conservative for CPU llama-server where higher parallelism can slow throughput.

### D8: Empty/short text — processed, not skipped

Examples with empty or very short text (< 3 whitespace tokens) are still
processed: both calls are made, and the model's fallback behavior applies
(extraction prompt returns `["text too short", ...]`; classification returns
`decision=0, confidence=0`). The result row is saved normally. This keeps the
artifact complete (one row per dataset example) and lets the analyzer decide
whether to filter short-text examples.

**Rationale**: Skipping would create a gap in the artifact and complicate
resume (the skipped ID would need separate tracking). The model prompts already
handle short text via their fallback layers.

## Risks / Trade-offs

- **[pymorphy3/spaCy install + model]** Normalization requires `pymorphy3` and,
  for English, a spaCy model (`python -m spacy download en_core_web_sm`).
  → Mitigation: lazy import; the runner and transport layer import and run
  without them. The error message names the install command. Russian-only runs
  need only `pymorphy3`.

- **[fsync per example throughput]** `fsync` after every example is durable but
  slow on large datasets. → Mitigation: `checkpoint_interval` is configurable;
  set it > 1 to batch fsyncs at the cost of reprocessing up to N-1 examples on
  crash. Default 1 for maximum safety.

- **[Repair heuristic is narrow]** Only ```` ```json ```` fence stripping is
  attempted. Models that emit other malformations stay `failed`. → Mitigation:
  acceptable for the baseline; the raw response is preserved in `raw_classify`
  for offline analysis. Repair heuristics can grow in a later change.

- **[per-example gather vs. batch methods]** The runner uses single `*_detailed`
  calls under a semaphore rather than `*_many_detailed`, for checkpoint
  durability. This means N session-lifetime coroutines instead of one batch
  call. → Mitigation: a single shared `aiohttp.ClientSession` is created once
  for the run and passed to each coroutine; the semaphore bounds concurrent
  in-flight requests. Overhead is negligible vs. model latency.

- **[extract once per text across rounds]** This change extracts theses once
  per example in a single baseline pass. Future optimization rounds that reuse
  the fixed extraction prompt could cache theses by text hash to avoid
  re-extraction. → Mitigation: out of scope; the artifact records `theses_raw`
  per run, so a future round can load a prior run's theses by `id` if desired.

## Open Questions

- **Dry-run flag**: Should the runner support `--dry-run` (load dataset, render
  prompts, exit without calling the model)? It is cheap and useful for
  validating config before a long run, but is not required by any acceptance
  criterion. Deferred — can be added without changing the specs.

- **Default concurrency value**: 4 is chosen as a conservative CPU default. The
  right value depends on the specific llama.cpp build and hardware. This should
  be tuned empirically on the target machine; the config key makes that a
  runtime change.
