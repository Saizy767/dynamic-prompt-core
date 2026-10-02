# Tasks

## 1. Module scaffold and config

- [x] 1.1 Create `metrics.py` with a module docstring describing the metrics stage, and a `MetricsConfig` dataclass reading `[metrics].bias_threshold` (default `0.15`), `[metrics].group_short_max` (default `10`), `[metrics].group_long_min` (default `30`), and `[metrics].output_dir` (default `data/results`) from `config.toml` via `tomllib`. Verify defaults apply when the section is absent and explicit values are used when present.
- [x] 1.2 Add the `[metrics]` section to `config.toml`. Verify `[llm]`, `[dataset]`, and `[runner]` still load unchanged via their respective config dataclasses.
- [x] 1.3 Verify `import metrics` succeeds without `scikit-learn` or `numpy` installed (lazy import inside the compute function). Verify the `ImportError` raised on first compute names the install command (`pip install scikit-learn numpy`).

## 2. Artifact loading and validation

- [x] 2.1 Implement `load_results(path)` that reads a `results_*.jsonl` artifact line by line into a list of dicts. Verify it loads an existing `data/results/results_*.jsonl` artifact and the record count equals the line count.
- [x] 2.2 Implement schema validation: every record SHALL carry `id`, `text`, `true_label`, `predicted_decision`, `confidence`, `classify_status`, `theses_norm`. On the first missing field, raise an error naming the record's `id` and the missing field. Verify a valid artifact loads without error and a synthetic record missing `predicted_decision` raises an error naming the `id` and `predicted_decision`.
- [x] 2.3 Implement `parse_artifact_name(path)` that extracts `run_id`, `prompt_version`, `split`, and `timestamp` from a `results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl` filename. Verify it parses a real artifact filename and raises on a name that does not match the pattern.

## 3. Core classification metrics

- [x] 3.1 Implement record filtering: split rows into `ok_rows` (`classify_status=ok`) and `parse_counts` (`Counter` of all `classify_status` values). Verify that with a synthetic artifact of 180 `ok` + 10 `failed` + 5 `repaired`, `ok_rows` has 180 entries and `parse_counts` reports `{"ok": 180, "failed": 10, "repaired": 5}`.
- [x] 3.2 Implement accuracy computation over `ok_rows` only (`correct / len(ok_rows)`). Verify with 180 `ok` rows of which 150 are correct that accuracy equals `150/180`. Verify `parse_counts["failed"]` is reported separately and does not affect accuracy.
- [x] 3.3 Implement per-class precision, recall, and F1 via `sklearn.metrics.precision_recall_fscore_support` (lazy import, `zero_division=0`, `labels=[0, 1]`), plus macro-F1 and weighted-F1. Verify on a synthetic 2×2 confusion (e.g. TP=80, TN=60, FP=20, FN=20) that per-class precision/recall/F1 and macro/weighted F1 match a hand-computed reference.
- [x] 3.4 Implement the 2×2 confusion matrix via `sklearn.metrics.confusion_matrix` and expose it as labeled `{tp, tn, fp, fn}` plus a `matrix` array (rows=true_label, cols=predicted). Verify with 100 `true_label=1` and 80 `true_label=0` rows that `tp + tn + fp + fn == 180`.

## 4. Minority-class F1

- [x] 4.1 Implement minority-class detection: `minority_class = min(label_counts, key=label_counts.get)` over `true_label` counts in `ok_rows`. Verify with 150 class-1 and 50 class-0 rows that `minority_class == 0`.
- [x] 4.2 Report `minority_class` and `f1.minority` (the F1 of the minority class) separately from `f1.macro`. Verify the minority F1 equals the per-class F1 of the minority class and differs from macro-F1 when classes are imbalanced. Verify accuracy is not the sole reported metric (the artifact exposes `f1.minority` alongside `accuracy`).

## 5. Prediction distribution and bias

- [x] 5.1 Implement prediction distribution: counts and shares of `predicted_decision` 0 and 1 over `ok_rows`. Verify with 30 predictions of 0 and 150 of 1 that shares are `0 — 16.7%, 1 — 83.3%` (rounded to one decimal).
- [x] 5.2 Implement true-label distribution over `ok_rows` and bias detection: `bias_gap = max(|pred_share[c] - true_share[c]|)`, `biased = bias_gap > bias_threshold`, `bias_toward = 1 if pred_share[1] > true_share[1] else 0`. Verify with true 50/50 and predicted 10/90 that `biased == True`, `bias_toward == 1`, and `bias_gap == 0.4`. Verify with true 50/50 and predicted 45/55 that `biased == False` at the default 0.15 threshold.

## 6. Group breakdown by text length

- [x] 6.1 Implement `length_group(word_count, short_max, long_min)` returning `"short"` (< `short_max`), `"medium"` (`short_max` ≤ count ≤ `long_min`), `"long"` (> `long_min`). Verify boundaries: 9 words → short, 10 → medium, 30 → medium, 31 → long.
- [x] 6.2 Implement per-group accuracy and F1 over `ok_rows` grouped by `len(text.split())`. A group with zero rows reports `{"count": 0, "accuracy": null, "f1": null}`. Verify on a synthetic artifact with short, medium, and long texts that each group reports the correct count, accuracy, and F1, and that an empty group reports nulls without raising.
- [x] 6.3 Verify group boundaries are overridden by `MetricsConfig.group_short_max` and `group_long_min`: with `short_max=5, long_min=15`, a 7-word text falls in medium and a 16-word text in long.

## 7. Metrics artifact persistence

- [x] 7.1 Implement `compute_metrics(rows, config, source_artifact)` returning the full metrics dict per design D10 (counts, accuracy, precision, recall, f1, minority_class, confusion_matrix, prediction_distribution, true_distribution, bias, groups, run_id, prompt_version, split, timestamp, source_artifact, `comparison=null`). Verify every key is present and types match the schema.
- [x] 7.2 Implement `write_metrics(metrics_dict, output_dir)` that writes `metrics_{run_id}_{prompt_version}_{split}_{timestamp}.json`. Verify the filename matches the pattern and the file is valid json.
- [x] 7.3 Implement `load_metrics(path)` that reads the json artifact. Verify a round-trip: `load_metrics(write_metrics(compute_metrics(...)))` returns all metrics without recomputation and every value matches the original.

## 8. Version comparison

- [x] 8.1 Implement `compare_versions(results_path_v0, results_path_v1, config)` that loads both results artifacts, aligns records by `id`, computes metrics for each, and returns per-metric deltas (`v1 - v0` for accuracy, macro-F1, minority-class F1). Verify on two synthetic artifacts where v1 has 2 more correct predictions that the accuracy delta is positive and matches the hand-computed difference.
- [x] 8.2 Implement the changed-decision list: for each `id` present in both, if `predicted_decision` differs, record `{id, v0, v1, direction}` where direction is `fixed` (v0 wrong → v1 right), `broke` (v0 right → v1 wrong), or `flip` (both wrong, prediction changed). Verify with 5 fixed and 3 broke that the list has 8 entries with the correct directions.
- [x] 8.3 Verify comparison rejects artifacts with different `split` values with a clear error, and handles non-overlapping `id` sets by comparing only the intersection (recording the non-overlap counts).

## 9. Dev vs holdout and CLI

- [x] 9.1 Implement the CLI `main()` with subcommands: `compute --results <path> [--config <path>]` and `compare --v0 <path> --v1 <path> [--config <path>]`. Verify `python metrics.py compute --results <existing artifact>` writes a metrics json without error.
- [x] 9.2 Verify that computing metrics on a `dev` results artifact and a `holdout` results artifact produces two separate metrics artifacts with `split=dev` and `split=holdout` respectively, and that holdout metrics are persisted but the artifact does not mark them as decision-driving (the `split` field distinguishes them).
- [x] 9.3 Verify the CLI prints a concise summary to stdout (counts, accuracy, macro-F1, minority-class F1, bias flag) and the path of the written metrics artifact.

## 10. Integration verification

- [x] 10.1 Run `python metrics.py compute --results <real dev results artifact>` against an existing `data/results/results_*.jsonl` artifact. Verify the run completes without crashing, the metrics artifact is written, and reloading it yields all schema fields populated.
- [x] 10.2 Verify the metrics artifact contains: confusion matrix with `tp+tn+fp+fn == counts.ok`, per-class F1 for class 0 and class 1, minority-class F1, group breakdown with all three groups, and parse-failure counts reported separately from accuracy.
- [x] 10.3 Run `python metrics.py compare --v0 <results_v0> --v1 <results_v1>` on two dev results artifacts (or two runs of the same artifact as a smoke test). Verify the delta dict and changed-decision list are returned, and the changed list directions sum correctly (fixed + broke + flip == len(changed)).
- [x] 10.4 Verify `config.toml` loads cleanly with the new `[metrics]` section and that `runner.py`, `dataset.py`, and `asyncTask.py` still import and read their own sections unchanged.
