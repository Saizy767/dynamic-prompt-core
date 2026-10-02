# Tasks

## 1. Transport rich-result API

- [x] 1.1 Add a `CallResult` dataclass to `asyncTask.py` with fields `parsed`, `raw_content`, `latency_ms`, `parse_status`, `status`, `finish_reason`, `error`, `truncated`. Verify it can be constructed from a `RawResponse` + parsed model + `ParseStatus`.
- [x] 1.2 Refactor `analyze` into an internal `_analyze_detailed` that performs the existing call→parse→log flow but returns a `CallResult`; have `analyze` delegate to it and return `result.parsed`. Verify `analyze` still returns the parsed model or `None` and that `smoke_test.py` still passes unchanged.
- [x] 1.3 Add `classify_detailed` and `extract_theses_detailed` as thin wrappers over `_analyze_detailed` using the same per-call defaults as `classify` / `extract_theses` (`max_tokens`, `truncate_tokens`). Verify `classify_detailed` returns a `CallResult` with `parse_status=ok` on valid JSON and `parsed=None` + `parse_status=invalid_json` on malformed JSON, and that one jsonl log entry is written when `log_path` is set.
- [x] 1.4 Add `_analyze_many_detailed` (mirrors `analyze_many` but collects `CallResult`), plus `classify_many_detailed` and `extract_theses_many_detailed`. Verify the returned list length and order match the input texts, concurrency is honored, and exactly N log entries are written for N texts.

## 2. Thesis normalization

- [x] 2.1 Create `normalize.py` with `normalize_thesis(text, lang)` that lower-cases, lemmatizes (pymorphy3 for `ru`, spaCy `blank("en")` for `en`), collapses whitespace, and protects negation particles (`не`, `нет`, `без`, `никогда`) from lemmatization. Load analyzers lazily and cache as module singletons. Verify `normalize_thesis("Клиент отменил заявку", "ru")` returns `"клиент отменить заявка"`.
- [x] 2.2 Verify negation is preserved: `normalize_thesis("Клиент не отменил заявку", "ru")` returns `"клиент не отменить заявка"`. Verify the raw input is not mutated. Verify lazy import: `import normalize` succeeds without pymorphy3 installed, and the `ImportError` on first call names the install command.
- [x] 2.3 Manually verify normalization on 5 representative Russian theses (affirmative, negated, mixed punctuation, short, multi-clause) and record expected outputs as inline assertions in a small test snippet.

## 3. Runner core

- [x] 3.1 Create `runner.py` with a `ResultRow` dataclass carrying all result-schema fields (`id`, `text`, `true_label`, `predicted_decision`, `confidence`, `theses_raw`, `theses_norm`, `extract_status`, `classify_status`, `extract_latency_ms`, `classify_latency_ms`, `raw_extract`, `raw_classify`) and a `to_dict` for jsonl serialization. Verify a `ResultRow` round-trips through `json.dumps` / `json.loads`.
- [x] 3.2 Implement the status mapping (design D6): `CallResult` → runner status (`ok` / `repaired` / `failed` / `not_attempted`), including the markdown-fence repair attempt. Verify a fenced valid JSON response maps to `repaired`, an unrecoverable response maps to `failed`, and a transport error maps to `failed`.
- [x] 3.3 Implement the per-example coroutine: call `extract_theses_detailed` (fixed `EXTRACTION_PROMPT`) and `classify_detailed` (selected classification prompt), assemble a `ResultRow`, and normalize theses via `normalize_thesis`. Verify a successful example produces a row with `extract_status=ok`, `classify_status=ok`, populated `theses_raw` / `theses_norm`, and `raw_extract` / `raw_classify` set.
- [x] 3.4 Implement the concurrency-limited gather: one shared `aiohttp.ClientSession`, `asyncio.Semaphore(concurrency)`, position-indexed results list. Verify with `concurrency=4` that no more than 4 requests are in flight simultaneously and that the final results list preserves input order.

## 4. Checkpointing and resume

- [x] 4.1 Implement checkpoint writing: open `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl.tmp` in append mode, write one jsonl line per completed example with `flush` + `os.fsync` (every `checkpoint_interval` examples). Verify the `.tmp` file grows by one line per example and is readable mid-run.
- [x] 4.2 Implement resume: on start, look for a `.tmp` file matching `run_id` + `split`, read processed IDs into a set, and skip those examples. Verify that after an artificial interruption at 50/200 examples, a second run processes only the remaining 150 and the final artifact has 200 rows.
- [x] 4.3 On completion, atomically rename the `.tmp` file to `.jsonl` via `os.replace`. Verify the final artifact has no `.tmp` suffix and contains exactly one row per split example.

## 5. Run artifact and summary

- [x] 5.1 Verify the completed artifact filename matches `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl` and that each record contains every result-schema field (reload with `json.loads` per line and assert keys).
- [x] 5.2 Implement the end-of-run summary printed to stdout: counts of `ok`, `parse_status=failed`, network errors, and timeouts across both call types. Verify a mixed run (some ok, one parse failure, one timeout) prints the correct counts.

## 6. Config integration

- [x] 6.1 Add a `RunnerConfig` dataclass reading `[runner].concurrency` (default 4), `[runner].checkpoint_interval` (default 1), `[runner].output_dir` (default `data/results`), `[runner].language` (default `ru`), and `[runner].prompt_version` (default `classify-v0`) from `config.toml` via `tomllib`. Verify defaults apply when the section is absent and explicit values are used when present.
- [x] 6.2 Add the `[runner]` section to `config.toml`. Verify `asyncTask.py` and `dataset.py` still load their own sections unchanged.

## 7. Integration verification

- [x] 7.1 Run the runner against `split=dev` on the prepared dataset artifact with a running llama-server. Verify the run completes without crashing, every dev example has a result row, and the request log contains two entries per example (one `extract_theses`, one `classify`).
- [x] 7.2 Run the runner against `split=holdout`. Verify the run completes and the artifact contains only holdout examples.
- [x] 7.3 Reload both artifacts and verify every record has all result-schema fields populated or explicitly set to `not_attempted` / empty for the failure cases.
- [x] 7.4 Artificially interrupt a dev run (SIGTERM after ~20 examples) and restart with the same `run_id`. Verify the resumed run skips completed examples, finishes the remaining examples, and the final artifact row count equals the dev split size.
- [x] 7.5 Verify the request log file exists, its name is tied to `run_id`, and its entry count equals twice the number of processed examples.
