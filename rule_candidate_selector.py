"""
Stage 2 rule candidate selector.

Loads a thesis artifact with theses and clusters, filters and ranks clusters
by frequency and precision, excludes clusters already in the prompt, selects
top-N candidates with representative theses, and persists a reloadable
candidate-list artifact for downstream rule formulation.

Usage:
    python rule_candidate_selector.py --artifact data/results/theses.json
    python rule_candidate_selector.py --artifact data/results/theses.json \
        --config config.toml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_FREQUENCY_THRESHOLD = 3
DEFAULT_TOP_N = 5
DEFAULT_MAX_REPRESENTATIVE_THESES = 3

REQUIRED_CLUSTER_FIELDS = (
    "cluster_id",
    "frequency",
    "precision",
    "positive_hits",
    "negative_hits",
)


class RuleCandidateSelectorError(ValueError):
    """Raised when a thesis artifact is invalid or cannot be loaded."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class RuleCandidateSelectorConfig:
    frequency_threshold: int = DEFAULT_FREQUENCY_THRESHOLD
    top_n: int = DEFAULT_TOP_N
    max_representative_theses: int = DEFAULT_MAX_REPRESENTATIVE_THESES
    output_dir: str = DEFAULT_OUTPUT_DIR

    @classmethod
    def from_config(
        cls, config_path: str = DEFAULT_CONFIG_PATH
    ) -> "RuleCandidateSelectorConfig":
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("rule_candidate_selector", {})
        return cls(
            frequency_threshold=int(
                s.get("frequency_threshold", DEFAULT_FREQUENCY_THRESHOLD)
            ),
            top_n=int(s.get("top_n", DEFAULT_TOP_N)),
            max_representative_theses=int(
                s.get("max_representative_theses", DEFAULT_MAX_REPRESENTATIVE_THESES)
            ),
            output_dir=s.get("output_dir", DEFAULT_OUTPUT_DIR),
        )


# --------------------------------------------------------------------------- #
#  Artifact loading and validation
# --------------------------------------------------------------------------- #
def _validate_cluster(cluster: Dict[str, Any]) -> None:
    cid = cluster.get("cluster_id")
    for fname in REQUIRED_CLUSTER_FIELDS:
        if fname not in cluster or cluster[fname] is None:
            raise RuleCandidateSelectorError(
                f"cluster cluster_id={cid!r} is missing required field '{fname}'"
            )


def load_thesis_artifact(path: str) -> Dict[str, Any]:
    """Load and schema-validate a thesis artifact (JSON with theses and clusters).

    Returns a dict with 'metadata', 'theses', and 'clusters' keys.
    Raises RuleCandidateSelectorError naming the first offending cluster's
    cluster_id and the missing field when a cluster does not conform.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "metadata" not in data:
        raise RuleCandidateSelectorError("artifact is missing top-level 'metadata' key")
    if "theses" not in data:
        raise RuleCandidateSelectorError("artifact is missing top-level 'theses' key")
    if "clusters" not in data:
        raise RuleCandidateSelectorError("artifact is missing top-level 'clusters' key")

    meta = data["metadata"]
    for key in ("run_id", "prompt_version"):
        if key not in meta or meta[key] is None:
            raise RuleCandidateSelectorError(
                f"artifact metadata is missing required field '{key}'"
            )

    for cid_str, cluster in data["clusters"].items():
        _validate_cluster(cluster)

    return data


def cluster_in_prompt(
    cluster: Dict[str, Any], theses: Dict[str, Any]
) -> bool:
    """Derive a cluster's in_prompt flag from its member theses.

    True when all member theses have in_prompt=true. False when the cluster
    has no members or any member is not in the prompt.
    """
    member_norms = cluster.get("member_norms", [])
    if not member_norms:
        return False
    for norm in member_norms:
        entry = theses.get(norm)
        if entry is None or not entry.get("in_prompt", False):
            return False
    return True


# --------------------------------------------------------------------------- #
#  Selection logic
# --------------------------------------------------------------------------- #
def filter_by_frequency(
    clusters: List[Dict[str, Any]], threshold: int
) -> List[Dict[str, Any]]:
    """Return clusters with frequency >= threshold."""
    return [c for c in clusters if c["frequency"] >= threshold]


def exclude_in_prompt(
    clusters: List[Dict[str, Any]], theses: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split clusters into (candidates, already_in_prompt).

    Candidates are clusters not already in the prompt; already_in_prompt are
    clusters whose members all have in_prompt=true.
    """
    candidates: List[Dict[str, Any]] = []
    already: List[Dict[str, Any]] = []
    for c in clusters:
        if cluster_in_prompt(c, theses):
            already.append(c)
        else:
            candidates.append(c)
    return candidates, already


def rank_by_precision(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort clusters by precision descending, frequency descending as tie-breaker."""
    return sorted(
        clusters,
        key=lambda c: (c["precision"], c["frequency"]),
        reverse=True,
    )


def select_top_n(
    ranked_clusters: List[Dict[str, Any]], top_n: int
) -> List[Dict[str, Any]]:
    """Return the first top_n entries, or all when fewer remain."""
    return ranked_clusters[:top_n]


def representative_theses(
    cluster: Dict[str, Any],
    theses: Dict[str, Any],
    max_theses: int,
) -> List[Dict[str, Any]]:
    """Return up to max_theses member theses sorted by descending frequency.

    Ties are broken by positive_hits descending. Each entry is
    {text_norm, text_raw, frequency}.
    """
    member_norms = cluster.get("member_norms", [])
    entries: List[Dict[str, Any]] = []
    for norm in member_norms:
        entry = theses.get(norm)
        if entry is None:
            continue
        entries.append(entry)
    entries.sort(
        key=lambda e: (e.get("frequency", 0), e.get("positive_hits", 0)),
        reverse=True,
    )
    result: List[Dict[str, Any]] = []
    for e in entries[:max_theses]:
        result.append(
            {
                "text_norm": e.get("text_norm", ""),
                "text_raw": e.get("text_raw", ""),
                "frequency": e.get("frequency", 0),
            }
        )
    return result


# --------------------------------------------------------------------------- #
#  Candidate selection orchestration
# --------------------------------------------------------------------------- #
def select_candidates(
    artifact: Dict[str, Any], config: RuleCandidateSelectorConfig
) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, int]]:
    """Run the full selection pipeline on a loaded thesis artifact.

    Returns (candidates, already_in_prompt_ids, counters) where each candidate
    has cluster_id, rank, precision, frequency, positive_hits, negative_hits,
    and representative_theses; counters has total_clusters, passed_frequency,
    excluded_in_prompt, selected_top_n.
    """
    theses = artifact["theses"]
    clusters: List[Dict[str, Any]] = list(artifact["clusters"].values())
    total_clusters = len(clusters)

    passed = filter_by_frequency(clusters, config.frequency_threshold)
    passed_frequency = len(passed)

    candidates, already = exclude_in_prompt(passed, theses)
    excluded_in_prompt = len(already)
    already_ids = [c["cluster_id"] for c in already]

    ranked = rank_by_precision(candidates)
    selected = select_top_n(ranked, config.top_n)
    selected_top_n = len(selected)

    candidate_objs: List[Dict[str, Any]] = []
    for rank, cluster in enumerate(selected):
        rep = representative_theses(
            cluster, theses, config.max_representative_theses
        )
        candidate_objs.append(
            {
                "cluster_id": cluster["cluster_id"],
                "rank": rank,
                "precision": cluster["precision"],
                "frequency": cluster["frequency"],
                "positive_hits": cluster["positive_hits"],
                "negative_hits": cluster["negative_hits"],
                "representative_theses": rep,
            }
        )

    counters = {
        "total_clusters": total_clusters,
        "passed_frequency": passed_frequency,
        "excluded_in_prompt": excluded_in_prompt,
        "selected_top_n": selected_top_n,
    }

    return candidate_objs, already_ids, counters


# --------------------------------------------------------------------------- #
#  Candidate artifact persistence
# --------------------------------------------------------------------------- #
def write_candidates(
    candidates: List[Dict[str, Any]],
    already_in_prompt: List[int],
    counters: Dict[str, int],
    run_id: str,
    prompt_version: str,
    config: RuleCandidateSelectorConfig,
    output_dir: str,
) -> str:
    """Serialize the candidate list to a JSON artifact.

    Filename: rule_candidates_{run_id}_{prompt_version}_{timestamp}.json
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"rule_candidates_{run_id}_{prompt_version}_{timestamp}.json"
    path = os.path.join(output_dir, name)

    dump = {
        "metadata": {
            "run_id": run_id,
            "prompt_version": prompt_version,
            "timestamp": timestamp,
            "config": {
                "frequency_threshold": config.frequency_threshold,
                "top_n": config.top_n,
                "max_representative_theses": config.max_representative_theses,
            },
        },
        "candidates": candidates,
        "already_in_prompt": already_in_prompt,
        "counters": counters,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


def load_candidates(path: str) -> Dict[str, Any]:
    """Restore the candidate list from a rule_candidates_*.json artifact.

    Returns a dict with 'metadata', 'candidates', 'already_in_prompt', and
    'counters' keys. No selection is re-run.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _print_summary(
    candidates: List[Dict[str, Any]],
    counters: Dict[str, int],
    run_id: str,
    prompt_version: str,
) -> None:
    print("\n" + "=" * 60)
    print(
        f"Rule candidates  (run_id={run_id}, "
        f"prompt_version={prompt_version})"
    )
    print("=" * 60)
    print(
        f"  total_clusters   : {counters['total_clusters']}\n"
        f"  passed_frequency : {counters['passed_frequency']}\n"
        f"  excluded_in_prompt: {counters['excluded_in_prompt']}\n"
        f"  selected_top_n   : {counters['selected_top_n']}"
    )
    print("-" * 60)
    for c in candidates:
        print(
            f"  rank {c['rank']}: cluster {c['cluster_id']}  "
            f"precision={c['precision']:.4f}  frequency={c['frequency']}  "
            f"rep={len(c['representative_theses'])}"
        )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 rule candidate selector"
    )
    parser.add_argument(
        "--artifact", required=True,
        help="Path to thesis artifact (JSON with theses and clusters)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    config = RuleCandidateSelectorConfig.from_config(args.config)
    artifact = load_thesis_artifact(args.artifact)
    run_id = artifact["metadata"]["run_id"]
    prompt_version = artifact["metadata"]["prompt_version"]

    candidates, already_in_prompt, counters = select_candidates(artifact, config)

    log.info(
        "selection: total_clusters=%d passed_frequency=%d "
        "excluded_in_prompt=%d selected_top_n=%d",
        counters["total_clusters"],
        counters["passed_frequency"],
        counters["excluded_in_prompt"],
        counters["selected_top_n"],
    )

    artifact_path = write_candidates(
        candidates,
        already_in_prompt,
        counters,
        run_id,
        prompt_version,
        config,
        config.output_dir,
    )
    _print_summary(candidates, counters, run_id, prompt_version)
    print(f"Candidates: {artifact_path}")


if __name__ == "__main__":
    main()
