# Tasks

## 1. Config and module scaffolding

- [x] 1.1 Add a `[thesis_analyzer]` section to `config.toml` with keys
  `embedding_model`, `cosine_lower`, `cosine_upper`, `manhattan_threshold`,
  `max_clusters`, `output_dir`, and inline comments documenting each default.
  Verify the file parses with `tomllib.load` and the section round-trips.
- [x] 1.2 Create `thesis_analyzer.py` with module docstring, `from __future__ import annotations`,
  a `ThesisAnalyzerConfig` dataclass with `from_config()` reading `[thesis_analyzer]`
  from `config.toml` (mirroring `MetricsConfig`), and default constants. Verify
  `ThesisAnalyzerConfig.from_config()` returns the configured values and falls back
  to defaults for missing keys.

## 2. Artifact loading and thesis bank

- [x] 2.1 Implement `load_results(path)` that reads a `results_*.jsonl` artifact,
  validates required fields (`id`, `true_label`, `theses_norm`, `theses_raw`,
  `extract_status`, `classify_status`), and raises an error naming the first
  offending record's `id` and field on failure. Reuse `metrics.parse_artifact_name`
  for filename parsing. Verify it loads an existing results artifact and raises on
  a record missing `theses_norm`.
- [x] 2.2 Implement a `ThesisEntry` dataclass with fields `text_raw`, `text_norm`,
  `embedding`, `frequency`, `positive_hits`, `negative_hits`, `demand`, `selected`,
  `in_prompt`, `cluster_id`, and a `precision` property (`positive_hits / frequency`).
  Verify `precision` returns 0.8 for `positive_hits=8, negative_hits=2`.
- [x] 2.3 Implement `ThesisBank` keyed by `text_norm` with `add_or_update(text_norm, text_raw, true_label)`
  that creates a new entry (`frequency=1`, increment `positive_hits` or `negative_hits`
  by `true_label`) or updates an existing one (increment `frequency` and the hit
  counter, refresh `text_raw`). Verify a new thesis gets `frequency=1`, a repeat
  gets `frequency=2`, and morphological variants with the same `text_norm` merge
  into one entry.
- [x] 2.4 Implement `filter_ok_records(rows)` returning only records with
  `classify_status=ok` and `extract_status=ok`. Verify records with
  `extract_status=failed` are excluded and `ok` records are retained.

## 3. Embeddings

- [x] 3.1 Implement a lazy-loaded embedding model getter (matching `normalize.py`'s
  lazy-import pattern) that loads `sentence-transformers` `SentenceTransformer`
  with the configured `embedding_model` name, caching the instance. Verify the
  module imports without `sentence-transformers` installed and the getter raises
  a clear `ImportError` with install instructions only when called.
- [x] 3.2 Implement `compute_embedding(text_norm)` returning a numpy float32 vector,
  and wire it into `ThesisBank.add_or_update` so a new thesis gets its embedding
  computed once and stored, while an existing thesis skips recomputation. Verify a
  new entry has a non-None `embedding` and a re-added entry keeps its original
  embedding (not recomputed).

## 4. Clustering

- [x] 4.1 Implement a `Cluster` dataclass with `cluster_id`, `member_norms` (list),
  `centroid` (numpy array), `count`, and a method to add a thesis and update the
  centroid as an incremental running mean. Verify the centroid equals the first
  thesis's embedding on creation and updates to the mean after adding a second.
- [x] 4.2 Implement cosine similarity (`numpy.dot` / norms) and Manhattan distance
  (`numpy.abs(a - b).sum()`) helpers. Verify cosine of identical vectors is 1.0
  and Manhattan distance of identical vectors is 0.0.
- [x] 4.3 Implement `assign_cluster(thesis_entry)` with the three-zone logic: cosine
  above `cosine_upper` → attach directly; cosine below `cosine_lower` → not attached
  by cosine; cosine in the grey zone → attach only if Manhattan distance is below
  `manhattan_threshold`. When no cluster matches and the cluster count is below
  `max_clusters`, create a new cluster. When the limit is reached, attach to the
  nearest cluster by cosine and log the forced assignment. Verify: a high-cosine
  thesis joins the existing cluster; a below-lower-cosine thesis does not; a
  grey-zone thesis joins via Manhattan; a new cluster is created when nothing
  matches and the limit is not reached; a forced assignment happens at the limit.

## 5. Cluster counters and dump

- [x] 5.1 Implement `recompute_cluster_counters(cluster)` that aggregates
  `frequency`, `positive_hits`, `negative_hits`, `demand`, `selected` as sums and
  `precision` as the frequency-weighted mean of member thesis precisions. Wire it
  to run after every `add_or_update` on a thesis that already has a `cluster_id`.
  Verify a cluster with one thesis (`precision=1.0, frequency=10`) and another
  (`precision=0.5, frequency=2`) reports weighted precision `≈0.917` and summed
  `frequency=12`.
- [x] 5.2 Implement `write_dump(path)` that serializes the bank and clusters to
  `thesis_bank_{run_id}_{prompt_version}_{timestamp}.json` with top-level keys
  `theses` (dict keyed by `text_norm`), `clusters` (dict keyed by `cluster_id`),
  and `metadata` (`run_id`, `prompt_version`, `embedding_model`, config snapshot).
  Embeddings and centroids serialize as lists. Verify the file is valid JSON,
  contains all bank entries and clusters, and the filename matches the pattern.
- [x] 5.3 Implement `load_dump(path)` that restores `ThesisBank` and clusters from
  a dump file, converting embedding/centroid lists back to numpy arrays, without
  recomputing embeddings or clustering. Verify that loading a dump written from a
  bank reproduces identical thesis entries, counters, and cluster centroids.

## 6. CLI and integration

- [x] 6.1 Implement the argparse CLI (`python thesis_analyzer.py --results <path>
  [--config config.toml]`) that loads config, loads the results artifact, filters
  ok records, builds the bank with embeddings and clusters, and writes the dump.
  Print a summary (thesis count, cluster count, top clusters by frequency). Verify
  the CLI runs against an existing `results_*.jsonl` artifact and prints the
  summary without errors.
- [x] 6.2 Run the analyzer end-to-end on a real results artifact, then reload the
  written dump with `load_dump` and confirm the reloaded bank matches the in-memory
  bank (same thesis count, same counters, same cluster assignments). Verify the
  reload path does not invoke the embedding model.
