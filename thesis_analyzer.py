"""
Stage 1 thesis analyzer.

Loads a stage1-baseline-runner results artifact, builds a deduplicated thesis
bank with embeddings and semantic clusters, tracks precision/frequency/demand/
selected/in_prompt counters per thesis and per cluster, and persists a
reloadable thesis-bank dump that feeds candidate selection for prompt rules.

Usage:
    python thesis_analyzer.py --results data/results/results_*.jsonl
    python thesis_analyzer.py --results data/results/results_*.jsonl \
        --config config.toml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from metrics import parse_artifact_name, MetricsError

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_COSINE_LOWER = 0.60
DEFAULT_COSINE_UPPER = 0.75
DEFAULT_MANHATTAN_THRESHOLD = 8.0
DEFAULT_MAX_CLUSTERS = 20

REQUIRED_FIELDS = (
    "id",
    "true_label",
    "theses_norm",
    "theses_raw",
    "extract_status",
    "classify_status",
)


class ThesisAnalyzerError(ValueError):
    """Raised when a results artifact is invalid or cannot be loaded."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class ThesisAnalyzerConfig:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    cosine_lower: float = DEFAULT_COSINE_LOWER
    cosine_upper: float = DEFAULT_COSINE_UPPER
    manhattan_threshold: float = DEFAULT_MANHATTAN_THRESHOLD
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    output_dir: str = DEFAULT_OUTPUT_DIR

    @classmethod
    def from_config(
        cls, config_path: str = DEFAULT_CONFIG_PATH
    ) -> "ThesisAnalyzerConfig":
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        t = config.get("thesis_analyzer", {})
        return cls(
            embedding_model=t.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            cosine_lower=float(t.get("cosine_lower", DEFAULT_COSINE_LOWER)),
            cosine_upper=float(t.get("cosine_upper", DEFAULT_COSINE_UPPER)),
            manhattan_threshold=float(
                t.get("manhattan_threshold", DEFAULT_MANHATTAN_THRESHOLD)
            ),
            max_clusters=int(t.get("max_clusters", DEFAULT_MAX_CLUSTERS)),
            output_dir=t.get("output_dir", DEFAULT_OUTPUT_DIR),
        )


# --------------------------------------------------------------------------- #
#  Artifact loading and validation
# --------------------------------------------------------------------------- #
def load_results(path: str) -> List[Dict[str, Any]]:
    """Load and schema-validate a results_*.jsonl artifact.

    Raises ThesisAnalyzerError naming the first offending record's id and the
    missing field when a record does not conform.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _validate_row(row)
            rows.append(row)
    return rows


def _validate_row(row: Dict[str, Any]) -> None:
    rid = row.get("id")
    for fname in REQUIRED_FIELDS:
        if fname not in row or row[fname] is None:
            raise ThesisAnalyzerError(
                f"record id={rid!r} is missing required field '{fname}'"
            )


def filter_ok_records(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return only records with classify_status=ok and extract_status=ok."""
    return [
        r
        for r in rows
        if r.get("classify_status") == "ok" and r.get("extract_status") == "ok"
    ]


# --------------------------------------------------------------------------- #
#  Embeddings (lazy-loaded, matching normalize.py's pattern)
# --------------------------------------------------------------------------- #
_embedding_model: Optional[object] = None
_embedding_model_name: Optional[str] = None


def _get_embedding_model(model_name: str):
    """Lazy-load and cache a SentenceTransformer model.

    Raises ImportError with install instructions only when called.
    """
    global _embedding_model, _embedding_model_name
    if _embedding_model is None or _embedding_model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Thesis analysis requires 'sentence-transformers'; "
                "install with: pip install sentence-transformers"
            ) from exc
        _embedding_model = SentenceTransformer(model_name)
        _embedding_model_name = model_name
    return _embedding_model


def compute_embedding(text_norm: str, model_name: str) -> np.ndarray:
    """Compute a float32 embedding vector for a normalized thesis."""
    model = _get_embedding_model(model_name)
    vec = model.encode(text_norm, convert_to_numpy=True)
    return vec.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Distance helpers
# --------------------------------------------------------------------------- #
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0.0 if either is zero)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance between two vectors."""
    return float(np.abs(a - b).sum())


# --------------------------------------------------------------------------- #
#  Thesis entry
# --------------------------------------------------------------------------- #
@dataclass
class ThesisEntry:
    text_raw: str
    text_norm: str
    embedding: Optional[np.ndarray] = None
    frequency: int = 0
    positive_hits: int = 0
    negative_hits: int = 0
    demand: int = 0
    selected: int = 0
    in_prompt: bool = False
    cluster_id: Optional[int] = None

    @property
    def precision(self) -> float:
        """Share of positive_hits over total occurrences."""
        if self.frequency == 0:
            return 0.0
        return self.positive_hits / self.frequency


# --------------------------------------------------------------------------- #
#  Cluster
# --------------------------------------------------------------------------- #
@dataclass
class Cluster:
    cluster_id: int
    member_norms: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    count: int = 0
    frequency: int = 0
    positive_hits: int = 0
    negative_hits: int = 0
    demand: int = 0
    selected: int = 0
    precision: float = 0.0

    def add_member(self, text_norm: str, embedding: np.ndarray) -> None:
        """Add a thesis and update the centroid as an incremental running mean."""
        if self.centroid is None:
            self.centroid = embedding.copy()
        else:
            self.centroid = (
                self.centroid * self.count + embedding
            ) / (self.count + 1)
        self.member_norms.append(text_norm)
        self.count += 1


# --------------------------------------------------------------------------- #
#  Thesis bank
# --------------------------------------------------------------------------- #
class ThesisBank:
    def __init__(self, config: ThesisAnalyzerConfig) -> None:
        self._config = config
        self._theses: Dict[str, ThesisEntry] = {}
        self._clusters: Dict[int, Cluster] = {}
        self._next_cluster_id: int = 0

    @property
    def theses(self) -> Dict[str, ThesisEntry]:
        return self._theses

    @property
    def clusters(self) -> Dict[int, Cluster]:
        return self._clusters

    def add_or_update(
        self, text_norm: str, text_raw: str, true_label: int
    ) -> ThesisEntry:
        """Add a new thesis or update an existing one by normalized form.

        New theses get their embedding computed and are assigned to a cluster.
        Existing theses skip embedding recomputation; their cluster counters
        are refreshed.
        """
        entry = self._theses.get(text_norm)
        if entry is None:
            embedding = compute_embedding(
                text_norm, self._config.embedding_model
            )
            entry = ThesisEntry(
                text_raw=text_raw,
                text_norm=text_norm,
                embedding=embedding,
                frequency=1,
                positive_hits=1 if true_label == 1 else 0,
                negative_hits=1 if true_label == 0 else 0,
            )
            self._theses[text_norm] = entry
            self.assign_cluster(entry)
        else:
            entry.frequency += 1
            if true_label == 1:
                entry.positive_hits += 1
            else:
                entry.negative_hits += 1
            entry.text_raw = text_raw
            if entry.cluster_id is not None:
                self.recompute_cluster_counters(entry.cluster_id)
        return entry

    def assign_cluster(self, entry: ThesisEntry) -> None:
        """Assign a thesis to a cluster using cosine + Manhattan grey-zone logic."""
        if entry.embedding is None:
            return

        best_cluster_id: Optional[int] = None
        best_cosine: float = -1.0
        for cid, cluster in self._clusters.items():
            if cluster.centroid is None:
                continue
            cos = cosine_similarity(entry.embedding, cluster.centroid)
            if cos > best_cosine:
                best_cosine = cos
                best_cluster_id = cid

        assigned = False
        if best_cluster_id is not None:
            if best_cosine >= self._config.cosine_upper:
                assigned = True
            elif best_cosine >= self._config.cosine_lower:
                cluster = self._clusters[best_cluster_id]
                if cluster.centroid is not None:
                    md = manhattan_distance(
                        entry.embedding, cluster.centroid
                    )
                    if md < self._config.manhattan_threshold:
                        assigned = True

        if assigned:
            self._add_to_cluster(best_cluster_id, entry)
        elif len(self._clusters) < self._config.max_clusters:
            self._create_cluster(entry)
        else:
            if best_cluster_id is not None:
                log.info(
                    "Cluster limit reached; forcing thesis '%s' into "
                    "cluster %d (cosine=%.4f)",
                    entry.text_norm,
                    best_cluster_id,
                    best_cosine,
                )
                self._add_to_cluster(best_cluster_id, entry)
            else:
                self._create_cluster(entry)

    def _add_to_cluster(
        self, cluster_id: Optional[int], entry: ThesisEntry
    ) -> None:
        if cluster_id is None:
            self._create_cluster(entry)
            return
        cluster = self._clusters[cluster_id]
        cluster.add_member(entry.text_norm, entry.embedding)
        entry.cluster_id = cluster_id
        self.recompute_cluster_counters(cluster_id)

    def _create_cluster(self, entry: ThesisEntry) -> None:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        cluster = Cluster(cluster_id=cid)
        cluster.add_member(entry.text_norm, entry.embedding)
        entry.cluster_id = cid
        self._clusters[cid] = cluster
        self.recompute_cluster_counters(cid)

    def recompute_cluster_counters(self, cluster_id: int) -> None:
        """Aggregate cluster counters from member theses.

        Sums frequency/positive_hits/negative_hits/demand/selected;
        precision is the frequency-weighted mean of member precisions.
        """
        cluster = self._clusters[cluster_id]
        total_freq = 0
        total_pos = 0
        total_neg = 0
        total_demand = 0
        total_selected = 0
        weighted_precision_sum = 0.0
        for norm in cluster.member_norms:
            entry = self._theses.get(norm)
            if entry is None:
                continue
            total_freq += entry.frequency
            total_pos += entry.positive_hits
            total_neg += entry.negative_hits
            total_demand += entry.demand
            total_selected += entry.selected
            if entry.frequency > 0:
                weighted_precision_sum += entry.precision * entry.frequency
        cluster.frequency = total_freq
        cluster.positive_hits = total_pos
        cluster.negative_hits = total_neg
        cluster.demand = total_demand
        cluster.selected = total_selected
        cluster.precision = (
            weighted_precision_sum / total_freq if total_freq > 0 else 0.0
        )


# --------------------------------------------------------------------------- #
#  Dump persistence
# --------------------------------------------------------------------------- #
def write_dump(
    bank: ThesisBank,
    run_id: str,
    prompt_version: str,
    config: ThesisAnalyzerConfig,
    output_dir: str,
) -> str:
    """Serialize the thesis bank and clusters to a JSON dump file.

    Filename: thesis_bank_{run_id}_{prompt_version}_{timestamp}.json
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"thesis_bank_{run_id}_{prompt_version}_{timestamp}.json"
    path = os.path.join(output_dir, name)

    theses_data: Dict[str, Any] = {}
    for norm, entry in bank.theses.items():
        theses_data[norm] = {
            "text_raw": entry.text_raw,
            "text_norm": entry.text_norm,
            "embedding": (
                entry.embedding.tolist() if entry.embedding is not None else None
            ),
            "frequency": entry.frequency,
            "positive_hits": entry.positive_hits,
            "negative_hits": entry.negative_hits,
            "demand": entry.demand,
            "selected": entry.selected,
            "in_prompt": entry.in_prompt,
            "cluster_id": entry.cluster_id,
        }

    clusters_data: Dict[str, Any] = {}
    for cid, cluster in bank.clusters.items():
        clusters_data[str(cid)] = {
            "cluster_id": cluster.cluster_id,
            "member_norms": cluster.member_norms,
            "centroid": (
                cluster.centroid.tolist()
                if cluster.centroid is not None
                else None
            ),
            "count": cluster.count,
            "frequency": cluster.frequency,
            "positive_hits": cluster.positive_hits,
            "negative_hits": cluster.negative_hits,
            "demand": cluster.demand,
            "selected": cluster.selected,
            "precision": cluster.precision,
        }

    dump = {
        "metadata": {
            "run_id": run_id,
            "prompt_version": prompt_version,
            "timestamp": timestamp,
            "embedding_model": config.embedding_model,
            "config": {
                "cosine_lower": config.cosine_lower,
                "cosine_upper": config.cosine_upper,
                "manhattan_threshold": config.manhattan_threshold,
                "max_clusters": config.max_clusters,
            },
        },
        "theses": theses_data,
        "clusters": clusters_data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


def load_dump(
    path: str, config: ThesisAnalyzerConfig
) -> ThesisBank:
    """Restore a ThesisBank and clusters from a dump file.

    Embeddings and centroids are converted from lists back to numpy arrays.
    No embeddings are recomputed and no clustering is performed.
    """
    with open(path, "r", encoding="utf-8") as f:
        dump = json.load(f)

    bank = ThesisBank(config)

    for norm, data in dump["theses"].items():
        embedding = (
            np.array(data["embedding"], dtype=np.float32)
            if data["embedding"] is not None
            else None
        )
        entry = ThesisEntry(
            text_raw=data["text_raw"],
            text_norm=data["text_norm"],
            embedding=embedding,
            frequency=data["frequency"],
            positive_hits=data["positive_hits"],
            negative_hits=data["negative_hits"],
            demand=data["demand"],
            selected=data["selected"],
            in_prompt=data["in_prompt"],
            cluster_id=data["cluster_id"],
        )
        bank.theses[norm] = entry

    max_cid = -1
    for cid_str, data in dump["clusters"].items():
        cid = int(cid_str)
        centroid = (
            np.array(data["centroid"], dtype=np.float32)
            if data["centroid"] is not None
            else None
        )
        cluster = Cluster(
            cluster_id=cid,
            member_norms=data["member_norms"],
            centroid=centroid,
            count=data["count"],
            frequency=data["frequency"],
            positive_hits=data["positive_hits"],
            negative_hits=data["negative_hits"],
            demand=data["demand"],
            selected=data["selected"],
            precision=data["precision"],
        )
        bank.clusters[cid] = cluster
        if cid > max_cid:
            max_cid = cid

    bank._next_cluster_id = max_cid + 1
    return bank


# --------------------------------------------------------------------------- #
#  Analysis
# --------------------------------------------------------------------------- #
def analyze(
    rows: List[Dict[str, Any]],
    config: ThesisAnalyzerConfig,
    source_artifact: str,
) -> Tuple[ThesisBank, Dict[str, str]]:
    """Build a thesis bank from validated result rows.

    Returns (bank, name_parts) where name_parts has run_id, prompt_version,
    split, and timestamp parsed from the source artifact filename.
    """
    name_parts = parse_artifact_name(source_artifact)
    ok_rows = filter_ok_records(rows)

    bank = ThesisBank(config)
    for row in ok_rows:
        true_label = row["true_label"]
        theses_norm = row["theses_norm"]
        theses_raw = row["theses_raw"]
        for i, norm in enumerate(theses_norm):
            raw = theses_raw[i] if i < len(theses_raw) else norm
            bank.add_or_update(norm, raw, true_label)

    return bank, name_parts


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _print_summary(
    bank: ThesisBank, name_parts: Dict[str, str]
) -> None:
    print("\n" + "=" * 60)
    print(
        f"Thesis bank  (run_id={name_parts['run_id']}, "
        f"prompt_version={name_parts['prompt_version']}, "
        f"split={name_parts['split']})"
    )
    print("=" * 60)
    print(f"  Theses  : {len(bank.theses)}")
    print(f"  Clusters: {len(bank.clusters)}")

    sorted_clusters = sorted(
        bank.clusters.values(),
        key=lambda c: c.frequency,
        reverse=True,
    )
    for cluster in sorted_clusters[:10]:
        print(
            f"    cluster {cluster.cluster_id}: "
            f"freq={cluster.frequency}  precision={cluster.precision:.4f}  "
            f"members={cluster.count}"
        )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 thesis analyzer")
    parser.add_argument(
        "--results", required=True, help="Path to results_*.jsonl artifact"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ThesisAnalyzerConfig.from_config(args.config)
    rows = load_results(args.results)
    bank, name_parts = analyze(rows, config, args.results)
    dump_path = write_dump(
        bank,
        name_parts["run_id"],
        name_parts["prompt_version"],
        config,
        config.output_dir,
    )
    _print_summary(bank, name_parts)
    print(f"Dump: {dump_path}")


if __name__ == "__main__":
    main()
