# Design

## Context

See proposal.md for motivation. The project has `runner.py` (spec
`stage1-baseline-runner`) which produces
`results_{run_id}_{prompt_version}_{split}_{timestamp}.jsonl` — one jsonl line
per dataset example, each a `ResultRow` with `id`, `text`, `true_label`,
`predicted_decision`, `confidence`, `theses_raw`, `theses_norm`,
`extract_status`, `classify_status`, `extract_latency_ms`,
`classify_latency_ms`, `raw_extract`, `raw_classify`. The `classify_status`
field distinguishes `ok`, `repaired`, `failed`, `not_attempted`.

`dataset.py` (spec `dataset-and-prompt`) provides `load_artifact` returning a
`Dataset` with `dev` / `holdout` lists of `Record(id, text, label)`. The
prepared dataset artifact is split-specific and reproducible.

`config.toml` already has `[llm]`, `[dataset]`, and `[runner]` sections read via
`tomllib`. The project has no `requirements.txt`; dependencies are documented
inline and installed ad hoc.

No metrics component exists today. The results artifact is self-sufficient for
metric computation: it is split-specific (the filename carries `split`) and
every row carries `true_label`, `predicted_decision`, `classify_status`, and
`text`, so accuracy, the confusion matrix, minority-class F1, prediction
distribution, and length-group breakdown can all be computed from the rows
alone. The `dataset-and-prompt` artifact is a cross-check / source of the split
structure but is not required at computation time.

## Goals / Non-Goals

**Goals:**

- Load and schema-validate a `results_*.jsonl` artifact, failing fast on the
  first malformed record.
- Compute classification metrics only over `classify_status=ok` rows; count and
  report `failed` / `repaired` / `not_attempted` separately.
- Report accuracy, per-class precision/recall/F1, macro-F1, weighted-F1, a
  labeled 2×2 confusion matrix, and minority-class F1.
- Compute prediction distribution, compare to true-label distribution, and flag
  bias above a configurable threshold.
- Break down accuracy and F1 by text-length group with configurable boundaries.
- Persist a single json metrics artifact named with `run_id`, `prompt_version`,
  `split`, `timestamp`, loadable without recomputation.
- Compare two metrics artifacts (same split, different prompt version) and
  return per-metric deltas plus changed-decision `id`s with direction.
- Compute dev and holdout separately; holdout is logged but not used for
  Stage 1 accept/reject decisions.

**Non-Goals:**

- Thesis analysis, clustering, or rule selection (separate stage).
- Prompt optimization (separate stage).
- Visualization (graphs, dashboards).
- Comparing more than two versions at once.
- Statistical significance (bootstrap, confidence intervals) — deferred to
  Stage 2.
- A human-readable markdown/console report beyond the json artifact (tracked as
  an open question).

## Decisions

### D1: Module layout — single `metrics.py`

```
metrics.py   # load + validate, compute, write artifact, compare, CLI
```

**Rationale**: One focused module. Metric computation is a pure transformation
of loaded rows; there is no I/O concurrency, no heavy state, and no separate
dependency surface that would justify splitting (unlike `runner.py` +
`normalize.py` where normalization carries the pymorphy3/spaCy dependency). Flat
layout matches the project (`asyncTask.py`, `dataset.py`, `normalize.py`,
`runner.py`).

**Alternative**: Split into `metrics.py` (computation) + `metrics_artifact.py`
(I/O). Rejected — the I/O is trivial json read/write and does not warrant a
second module.

### D2: Metric computation via scikit-learn + numpy

Use `sklearn.metrics.precision_recall_fscore_support` and
`sklearn.metrics.confusion_matrix` for per-class / macro / weighted metrics and
the confusion matrix, and `numpy` for distribution shares. Both are imported
lazily inside the compute function so `metrics.py` imports without them; the
error message on first use names the install command.

**Rationale**: Hand-rolling precision/recall/F1 is error-prone (zero-division,
label ordering) and sklearn is the project's natural choice for classification
metrics. Lazy import keeps the module importable for artifact loading and
comparison without the heavy deps.

**Alternative**: Pure-Python implementation with no deps. Rejected — reinvents
well-tested metric logic and the spec explicitly lists numpy/sklearn as
dependencies.

### D3: Record filtering — `classify_status=ok` only for classification metrics

```python
ok_rows   = [r for r in rows if r["classify_status"] == "ok"]
parse_counts = Counter(r["classify_status"] for r in rows)  # failed/repaired/not_attempted
```

Accuracy, precision, recall, F1, confusion matrix, minority-class F1, prediction
distribution, and group breakdown are computed over `ok_rows`. `parse_counts`
is reported as a separate section and MUST NOT influence accuracy.

**Rationale**: The spec requires parse failures to be excluded from accuracy and
reported separately. A parse failure is a pipeline failure, not a model
classification mistake.

### D4: Minority class — from `true_label` distribution of the loaded rows

```python
label_counts = Counter(r["true_label"] for r in ok_rows)
minority_class = min(label_counts, key=label_counts.get)
```

The results artifact is split-specific (the filename carries `split`), so for a
dev artifact the `true_label` distribution is the dev distribution. The minority
class is the label with fewer examples; its F1 is reported as
`minority_class_f1` alongside `macro_f1`.

**Rationale**: Self-sufficient — no need to load the `dataset-and-prompt`
artifact at computation time. The results rows already carry `true_label`.

**Alternative**: Load the dataset artifact to count dev labels. Rejected —
duplicates data already present in the results artifact and adds a coupling.

### D5: Prediction distribution and bias — configurable threshold

```python
pred_share  = {c: pred_counts[c] / n_pred for c in (0, 1)}
true_share  = {c: true_counts[c] / n_true for c in (0, 1)}
bias_gap    = max(abs(pred_share[c] - true_share[c]) for c in (0, 1))
biased      = bias_gap > bias_threshold
bias_toward = 1 if pred_share[1] > true_share[1] else 0
```

`bias_threshold` defaults to `0.15` (15%) and is configurable via
`[metrics].bias_threshold`. The artifact stores `pred_distribution`,
`true_distribution`, `bias_gap`, `biased` (bool), and `bias_toward`.

**Rationale**: The open question (10/15/20%) is resolved with a configurable
default of 15% — a middle ground that catches the 50/50 → 10/90 case (40% gap)
without flagging mild natural variance. The config key lets the user tune it.

### D6: Group breakdown — configurable boundaries

```python
def length_group(word_count, short_max=10, long_min=30):
    if word_count < short_max:   return "short"
    if word_count <= long_min:   return "medium"
    return "long"
```

Boundaries default to `short_max=10`, `long_min=30` (short <10, medium 10–30,
long >30) and are configurable via `[metrics].group_short_max` and
`[metrics].group_long_min`. For each group, accuracy and F1 are computed over
the `ok_rows` in that group. A group with zero rows reports null metrics rather
than raising.

**Rationale**: The spec says groups are config-defined and MAY be overridden.
Word count (`len(text.split())`) is the grouping key; the `text` field is
present in every result row.

### D7: Confusion matrix — labeled dict + 2×2 array

```python
{
  "tp": int, "tn": int, "fp": int, "fn": int,
  "matrix": [[tn, fp], [fn, tp]]   # rows=true_label, cols=predicted
}
```

**Rationale**: The open question (plain four numbers vs. labeled) is resolved
with both — explicit `tp/tn/fp/fn` keys for direct access and a `matrix` array
for tooling that expects a 2×2 layout. Labels remove ambiguity about which
number is which.

### D8: Version comparison — changed-decision list with direction

```python
for rid in common_ids:
    p0 = pred_by_id_v0[rid]
    p1 = pred_by_id_v1[rid]
    if p0 != p1:
        correct0 = (p0 == true_by_id[rid])
        correct1 = (p1 == true_by_id[rid])
        direction = "fixed"  if (not correct0 and correct1) else \
                    "broke"  if (correct0 and not correct1) else \
                    "flip"
        changed.append({"id": rid, "v0": p0, "v1": p1, "direction": direction})
```

The comparison loads two results artifacts (not two metrics artifacts) on the
same split, aligns by `id`, and returns per-metric deltas (`v1 - v0` for
accuracy, F1, minority-class F1) plus the `changed` list. Direction is `fixed`
(v0 wrong → v1 right), `broke` (v0 right → v1 wrong), or `flip` (both wrong,
prediction changed).

**Rationale**: The spec requires the list of `id`s whose `predicted_decision`
changed with direction. Comparing results artifacts (not metrics artifacts) is
necessary because the metrics artifact does not store per-record predictions.
The open question (all records vs. only where at least one version was wrong) is
resolved as: report all changed predictions, with `flip` covering the both-wrong
case so nothing is hidden.

**Alternative**: Compare metrics artifacts only. Rejected — cannot recover
per-record prediction changes from aggregate metrics.

### D9: Config — `[metrics]` section

```toml
[metrics]
bias_threshold = 0.15
group_short_max = 10
group_long_min = 30
output_dir = "data/results"
```

Read with `tomllib` alongside the existing sections. Defaults apply when the
section is absent. `output_dir` defaults to the runner's output dir
(`data/results`) so metrics land beside results.

**Rationale**: Centralizes metrics knobs. No change to existing `[llm]`,
`[dataset]`, or `[runner]` keys.

### D10: Metrics artifact schema

```json
{
  "run_id": "...", "prompt_version": "...", "split": "dev", "timestamp": "...",
  "source_artifact": "results_*.jsonl",
  "counts": {"total": N, "ok": K, "failed": F, "repaired": R, "not_attempted": NA},
  "accuracy": float,
  "precision": {"0": float, "1": float},
  "recall":    {"0": float, "1": float},
  "f1":        {"0": float, "1": float, "macro": float, "weighted": float, "minority": float},
  "minority_class": int,
  "confusion_matrix": {"tp": int, "tn": int, "fp": int, "fn": int, "matrix": [[..],[..]]},
  "prediction_distribution": {"0": float, "1": float, "counts": {"0": int, "1": int}},
  "true_distribution":         {"0": float, "1": float, "counts": {"0": int, "1": int}},
  "bias": {"gap": float, "biased": bool, "toward": int},
  "groups": {
    "short":  {"count": int, "accuracy": float, "f1": float},
    "medium": {"count": int, "accuracy": float, "f1": float},
    "long":   {"count": int, "accuracy": float, "f1": float}
  },
  "comparison": null
}
```

`comparison` is `null` for a single-version artifact and populated with the
delta + changed list when the comparison command is run.

**Rationale**: One flat json object with every metric the acceptance criteria
require, keyed for direct access. `source_artifact` ties the metrics to the
exact results file they were derived from.

## Risks / Trade-offs

- **[sklearn/numpy install]** Metric computation requires `scikit-learn` and
  `numpy`. → Mitigation: lazy import; the module imports and loads/validates
  artifacts without them. The error on first compute names the install command.

- **[Zero-division on empty groups or single-class splits]** A length group or a
  split with no `ok` rows, or a split where one class is absent, can produce
  zero-division in precision/recall/F1. → Mitigation: sklearn's
  `precision_recall_fscore_support` accepts `zero_division=0`; empty groups
  report null metrics rather than raising; the artifact records `counts` so a
  consumer can detect degenerate cases.

- **[Minority class from results vs. dataset artifact]** Minority class is
  derived from the loaded results rows' `true_label`, not from the
  `dataset-and-prompt` artifact. → Mitigation: the results artifact is
  split-specific and carries `true_label` per row, so this is equivalent for a
  dev artifact. If a results artifact were ever mislabeled, the minority class
  would be wrong — but so would every other metric, so this is not an additional
  risk.

- **[Comparison requires both results artifacts]** Version comparison loads two
  results artifacts, not two metrics artifacts, to recover per-record changes.
  → Mitigation: the CLI takes two results paths; if a user only has metrics
  artifacts, per-record changes are unavailable and the command reports only
  aggregate deltas. This is documented in the comparison output.

- **[Bias threshold default is a guess]** 15% is a reasonable default but the
  right value is empirical. → Mitigation: it is configurable; the artifact
  stores the raw `bias_gap` so the threshold can be revisited without
  recomputation.

## Open Questions

- **Human-readable report**: Should the component emit a markdown or console
  summary in addition to the json artifact? It is cheap and useful for quick
  review, but no acceptance criterion requires it. Deferred — can be added
  without changing the specs.

- **Comparison input form**: Should comparison accept two metrics artifacts
  (aggregate deltas only) as a fallback when results artifacts are unavailable,
  or always require results artifacts for the changed-decision list? Deferred —
  the current design requires results artifacts for the full list and degrades
  to aggregate-only if only metrics artifacts are given.
