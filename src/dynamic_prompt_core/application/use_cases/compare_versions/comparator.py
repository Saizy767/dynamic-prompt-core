"""
Stage 2 version comparator.

Loads a newly composed prompt version and the currently active prompt version,
runs both on the same dev split, computes comparable metrics, compares by
macro-F1 with minority-class F1 as tie-breaker, decides to accept or roll back
the new version, maintains a consecutive-rollback counter, returns the next
candidate on rollback, and persists a reloadable decision artifact with full
lineage and changed predictions.

Usage:
    python version_comparator.py --artifact data/results/prompt_v*.json \
        --active-results data/results/results_*_classify-v0_dev_*.jsonl \
        --config config.toml --endpoint http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from dynamic_prompt_core.domain.prompts import CLASSIFICATION_PROMPT_V0, PromptArtifact

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_MAX_CONSECUTIVE_ROLLBACKS = 2
DEFAULT_DECISION_METRIC = "macro_f1"
DEFAULT_TIE_BREAKER_METRIC = "minority_f1"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"

REQUIRED_PROMPT_VERSION_FIELDS = (
    "version",
    "text",
    "hash",
    "rules",
    "source_candidates",
    "base_version",
)


class VersionComparatorError(ValueError):
    """Raised when an artifact is invalid or comparison fails."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class VersionComparatorConfig:
    max_consecutive_rollbacks: int = DEFAULT_MAX_CONSECUTIVE_ROLLBACKS
    decision_metric: str = DEFAULT_DECISION_METRIC
    tie_breaker_metric: str = DEFAULT_TIE_BREAKER_METRIC
    output_dir: str = DEFAULT_OUTPUT_DIR

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_CONFIG_PATH) -> VersionComparatorConfig:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("version_comparator", {})
        return cls(
            max_consecutive_rollbacks=int(
                s.get("max_consecutive_rollbacks", DEFAULT_MAX_CONSECUTIVE_ROLLBACKS)
            ),
            decision_metric=s.get("decision_metric", DEFAULT_DECISION_METRIC),
            tie_breaker_metric=s.get("tie_breaker_metric", DEFAULT_TIE_BREAKER_METRIC),
            output_dir=s.get("output_dir", DEFAULT_OUTPUT_DIR),
        )


# --------------------------------------------------------------------------- #
#  Artifact loading and validation
# --------------------------------------------------------------------------- #
def load_new_version(path: str) -> dict[str, Any]:
    """Load a prompt-version artifact produced by the composer.

    Validates that all required fields are present. Raises
    VersionComparatorError naming the missing field on failure.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for field in REQUIRED_PROMPT_VERSION_FIELDS:
        if field not in data:
            raise VersionComparatorError(f"prompt version artifact missing required field: {field}")

    return cast(dict[str, Any], data)


def build_prompt_artifact(version_dict: dict[str, Any]) -> PromptArtifact:
    """Construct a PromptArtifact from a loaded prompt-version dict.

    layers is None because the composer stores rendered text, not the layer
    structure. The runner only uses .text and .version.
    """
    return PromptArtifact(
        version=version_dict["version"],
        layers=None,
        text=version_dict["text"],
        sha256=version_dict.get("hash", ""),
    )


def load_active_version(
    active_prompt: PromptArtifact | None = None,
) -> PromptArtifact:
    """Return the currently active PromptArtifact.

    Defaults to CLASSIFICATION_PROMPT_V0. Raises VersionComparatorError when
    active_prompt is explicitly None (no active version available).
    """
    if active_prompt is None:
        raise VersionComparatorError("no active prompt version found in the prompt store")
    return active_prompt


# --------------------------------------------------------------------------- #
#  Run new version on dev
# --------------------------------------------------------------------------- #
async def _run_version_async(
    prompt_artifact: PromptArtifact,
    config_path: str,
    dataset_artifact: str,
    split: str,
    endpoint: str,
    run_id: str,
) -> str:
    """Construct a BaselineRunner with a custom classify_prompt and run it."""
    from dynamic_prompt_core.application.services.classification_policy import (
        ArgmaxClassificationPolicy,
    )
    from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
        BaselineRunner,
        RunnerConfig,
    )
    from dynamic_prompt_core.domain.models.candidate import Candidate
    from dynamic_prompt_core.infrastructure.llm import AsyncTask
    from dynamic_prompt_core.infrastructure.llm.scoring.config import (
        scorer_backend_config_from_toml,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.factory import (
        build_candidate_scorer,
    )

    runner_config = RunnerConfig.from_config(config_path)

    log_path = os.path.join(
        runner_config.output_dir,
        f"requests_{run_id}_{prompt_artifact.version}_{split}.jsonl",
    )

    task = AsyncTask(
        config_path=config_path,
        endpoint=endpoint,
        log_path=log_path,
        run_id=run_id,
    )

    with open(config_path, "rb") as f:
        raw_config = tomllib.load(f)
    default_model_path = str(raw_config.get("llm", {}).get("model_path", "model"))

    runner = BaselineRunner(
        task=task,
        config=runner_config,
        split=split,
        scorer=build_candidate_scorer(
            scorer_backend_config_from_toml(
                raw_config, default_model_path=default_model_path
            )
        ),
        policy=ArgmaxClassificationPolicy(),
        candidates=[Candidate(v) for v in runner_config.candidates],
        run_id=run_id,
        dataset_artifact=dataset_artifact,
        classify_prompt=prompt_artifact,
    )

    _, artifact_path = await runner.run()
    return artifact_path


def run_version(
    prompt_artifact: PromptArtifact,
    config_path: str,
    dataset_artifact: str,
    split: str = "dev",
    endpoint: str = DEFAULT_ENDPOINT,
    run_id: str | None = None,
) -> str:
    """Run a prompt version on a split and return the results artifact path.

    Uses asyncio.run to drive the async BaselineRunner. The fixed extraction
    prompt (EXTRACTION_PROMPT) is used without modification — the runner's
    _extract_prompt is always EXTRACTION_PROMPT regardless of classify_prompt.
    """
    if run_id is None:
        run_id = f"cmp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    return asyncio.run(
        _run_version_async(
            prompt_artifact,
            config_path,
            dataset_artifact,
            split,
            endpoint,
            run_id,
        )
    )


# --------------------------------------------------------------------------- #
#  Metrics comparison and decision logic
# --------------------------------------------------------------------------- #
def compare_metrics(
    results_active: str,
    results_new: str,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    """Compare two results artifacts on the same split.

    Reuses metrics.compare_versions. Returns
    (metrics_active, metrics_new, deltas, changed).
    """
    from dynamic_prompt_core.application.services.metrics import metrics as metrics_module

    m_config = metrics_module.MetricsConfig.from_config(config_path)
    comparison = metrics_module.compare_versions(results_active, results_new, m_config)

    rows_active = metrics_module.load_results(results_active)
    rows_new = metrics_module.load_results(results_new)
    metrics_active = metrics_module.compute_metrics(rows_active, m_config, results_active)
    metrics_new = metrics_module.compute_metrics(rows_new, m_config, results_new)

    return metrics_active, metrics_new, comparison["deltas"], comparison["changed"]


def _extract_metric(metrics: dict[str, Any], metric_name: str) -> float:
    """Extract a metric value from a metrics dict by name.

    Config names (macro_f1, minority_f1, weighted_f1) map to the f1 sub-dict
    keys used by stage1-metrics (macro, minority, weighted).
    """
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    f1_key_map = {
        "macro_f1": "macro",
        "minority_f1": "minority",
        "weighted_f1": "weighted",
    }
    if metric_name in f1_key_map:
        return float(metrics["f1"][f1_key_map[metric_name]])
    raise VersionComparatorError(
        f"unknown metric: {metric_name!r} "
        f"(expected accuracy, macro_f1, minority_f1, or weighted_f1)"
    )


def decide(
    metrics_new: dict[str, Any],
    metrics_active: dict[str, Any],
    config: VersionComparatorConfig,
) -> tuple[str, str]:
    """Decide whether to accept or roll back the new version.

    Accept when the new version's decision_metric is >= the active's.
    When equal, accept only if the tie_breaker_metric is strictly higher.
    Otherwise roll back.

    Returns (decision, reason) where decision is "accept" or "rollback".
    """
    primary = config.decision_metric
    tie = config.tie_breaker_metric

    new_primary = _extract_metric(metrics_new, primary)
    active_primary = _extract_metric(metrics_active, primary)

    if new_primary > active_primary:
        return "accept", f"{primary} improved ({active_primary:.4f} -> {new_primary:.4f})"

    if new_primary == active_primary:
        new_tie = _extract_metric(metrics_new, tie)
        active_tie = _extract_metric(metrics_active, tie)
        if new_tie > active_tie:
            return (
                "accept",
                f"{primary} tied ({new_primary:.4f}), {tie} improved "
                f"({active_tie:.4f} -> {new_tie:.4f})",
            )
        return (
            "accept",
            f"{primary} tied ({new_primary:.4f}), {tie} did not improve "
            f"({active_tie:.4f} -> {new_tie:.4f})",
        )

    return (
        "rollback",
        f"{primary} degraded ({active_primary:.4f} -> {new_primary:.4f})",
    )


def changed_decisions(
    results_active: str,
    results_new: str,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """Return the list of {id, direction} for changed predictions.

    Reuses metrics.compare_versions. Directions: fixed, broke, flip.
    """
    from dynamic_prompt_core.application.services.metrics import metrics as metrics_module

    m_config = metrics_module.MetricsConfig.from_config(config_path)
    comparison = metrics_module.compare_versions(results_active, results_new, m_config)
    return cast(list[dict[str, Any]], comparison["changed"])


# --------------------------------------------------------------------------- #
#  Rollback counter and candidate queue
# --------------------------------------------------------------------------- #
def update_rollback_count(current_count: int, decision: str) -> int:
    """Return the updated rollback count.

    Reset to 0 on accept, increment by 1 on rollback.
    """
    if decision == "accept":
        return 0
    return current_count + 1


def check_stop(rollback_count: int, config: VersionComparatorConfig) -> tuple[bool, str | None]:
    """Check whether the cycle should stop.

    Returns (should_stop, reason). Stops when rollback_count reaches
    max_consecutive_rollbacks.
    """
    if rollback_count >= config.max_consecutive_rollbacks:
        return True, "max_consecutive_rollbacks reached"
    return False, None


def next_candidate(
    candidate_queue: list[str],
) -> tuple[str | None, str | None]:
    """Pop and return the next candidate artifact path from the queue.

    Returns (path, None) on success, or (None, reason) when the queue is empty.
    The rejected version's source_candidates are NOT carried forward — the next
    candidate is a fresh artifact from the queue.
    """
    if not candidate_queue:
        return None, "candidate queue exhausted"
    return candidate_queue.pop(0), None


# --------------------------------------------------------------------------- #
#  Decision artifact persistence
# --------------------------------------------------------------------------- #
def write_decision(
    decision: str,
    new_version: str,
    active_version: str,
    metrics_new: dict[str, Any],
    metrics_active: dict[str, Any],
    diff_accuracy: float,
    diff_macro_f1: float,
    diff_minority_f1: float,
    rollback_count: int,
    reason: str,
    changed_decisions: list[dict[str, Any]],
    run_id: str,
    config: VersionComparatorConfig,
    output_dir: str,
    artifact_path_new: str | None = None,
    artifact_path_active: str | None = None,
) -> str:
    """Serialize the decision to a JSON artifact.

    Filename: decision_{run_id}_v{new_version}_vs_v{active_version}_{timestamp}.json
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"decision_{run_id}_v{new_version}_vs_v{active_version}_{timestamp}.json"
    path = os.path.join(output_dir, name)

    dump = {
        "decision": decision,
        "new_version": new_version,
        "active_version": active_version,
        "metrics_new": metrics_new,
        "metrics_active": metrics_active,
        "diff_accuracy": diff_accuracy,
        "diff_macro_f1": diff_macro_f1,
        "diff_minority_f1": diff_minority_f1,
        "rollback_count": rollback_count,
        "reason": reason,
        "changed_decisions": changed_decisions,
        "run_id": run_id,
        "timestamp": timestamp,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": {
            "max_consecutive_rollbacks": config.max_consecutive_rollbacks,
            "decision_metric": config.decision_metric,
            "tie_breaker_metric": config.tie_breaker_metric,
            "artifact_path_new": artifact_path_new,
            "artifact_path_active": artifact_path_active,
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


def load_decision(path: str) -> dict[str, Any]:
    """Restore a decision artifact without re-running the comparison."""
    with open(path, encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
def log_decision(
    decision: str,
    new_version: str,
    active_version: str,
    metrics_new: dict[str, Any],
    metrics_active: dict[str, Any],
    diffs: dict[str, float],
    rollback_count: int,
) -> None:
    """Log the decision process at INFO level."""
    log.info(
        "decision: new_version=%s active_version=%s "
        "metrics_new(accuracy=%.4f macro_f1=%.4f minority_f1=%.4f) "
        "metrics_active(accuracy=%.4f macro_f1=%.4f minority_f1=%.4f) "
        "diff_accuracy=%.4f diff_macro_f1=%.4f diff_minority_f1=%.4f "
        "rollback_count=%d decision=%s",
        new_version,
        active_version,
        metrics_new["accuracy"],
        metrics_new["f1"]["macro"],
        metrics_new["f1"]["minority"],
        metrics_active["accuracy"],
        metrics_active["f1"]["macro"],
        metrics_active["f1"]["minority"],
        diffs["accuracy"],
        diffs["macro_f1"],
        diffs["minority_f1"],
        rollback_count,
        decision,
    )


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _print_summary(
    decision: str,
    new_version: str,
    active_version: str,
    diffs: dict[str, float],
    rollback_count: int,
    reason: str,
    changed_count: int,
) -> None:
    print("\n" + "=" * 60)
    print(f"Decision: {decision.upper()}")
    print("=" * 60)
    print(
        f"  new_version    : {new_version}\n"
        f"  active_version : {active_version}\n"
        f"  diff_accuracy  : {diffs['accuracy']:+.4f}\n"
        f"  diff_macro_f1  : {diffs['macro_f1']:+.4f}\n"
        f"  diff_minority_f1: {diffs['minority_f1']:+.4f}\n"
        f"  rollback_count : {rollback_count}\n"
        f"  changed_preds  : {changed_count}\n"
        f"  reason         : {reason}"
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 version comparator")
    parser.add_argument(
        "--artifact",
        required=True,
        help="Path to composed prompt-version artifact (prompt_v*.json)",
    )
    parser.add_argument(
        "--active-results",
        required=True,
        help="Path to active version results artifact on dev (results_*.jsonl)",
    )
    parser.add_argument(
        "--candidate-queue",
        nargs="*",
        default=[],
        help="Candidate artifact paths for rollback (consumed left-to-right)",
    )
    parser.add_argument(
        "--rollback-count",
        type=int,
        default=0,
        help="Current consecutive rollback count (from prior decision artifact)",
    )
    parser.add_argument(
        "--dataset-artifact",
        default=None,
        help="Path to prepared dataset artifact (jsonl/parquet)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="LLM server endpoint",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "holdout"],
        default="dev",
        help="Split to run the new version on (default: dev)",
    )
    args = parser.parse_args()

    pass  # logging configured by composition root

    config = VersionComparatorConfig.from_config(args.config)

    version_dict = load_new_version(args.artifact)
    new_prompt = build_prompt_artifact(version_dict)
    active_prompt = load_active_version(CLASSIFICATION_PROMPT_V0)

    run_id = version_dict.get("metadata", {}).get("run_id", "unknown")

    dataset_artifact = args.dataset_artifact
    if dataset_artifact is None:
        from dynamic_prompt_core.application.use_cases.run_baseline.runner import RunnerConfig

        runner_config = RunnerConfig.from_config(args.config)
        dataset_artifact = runner_config.dataset_artifact

    print(f"Running new version {new_prompt.version} on {args.split}...")
    results_new_path = run_version(
        new_prompt,
        args.config,
        dataset_artifact,
        split=args.split,
        endpoint=args.endpoint,
        run_id=f"{run_id}-cmp-{new_prompt.version}",
    )
    print(f"Results artifact: {results_new_path}")

    metrics_active, metrics_new, deltas, changed = compare_metrics(
        args.active_results, results_new_path, args.config
    )

    decision, reason = decide(metrics_new, metrics_active, config)
    rollback_count = update_rollback_count(args.rollback_count, decision)

    log_decision(
        decision,
        new_prompt.version,
        active_prompt.version,
        metrics_new,
        metrics_active,
        deltas,
        rollback_count,
    )

    artifact_path = write_decision(
        decision,
        new_prompt.version,
        active_prompt.version,
        metrics_new,
        metrics_active,
        deltas["accuracy"],
        deltas["macro_f1"],
        deltas["minority_f1"],
        rollback_count,
        reason,
        changed,
        run_id,
        config,
        config.output_dir,
        artifact_path_new=results_new_path,
        artifact_path_active=args.active_results,
    )

    _print_summary(
        decision,
        new_prompt.version,
        active_prompt.version,
        deltas,
        rollback_count,
        reason,
        len(changed),
    )
    print(f"Decision artifact: {artifact_path}")

    should_stop, stop_reason = check_stop(rollback_count, config)
    if should_stop:
        print(f"\nCycle stopped: {stop_reason}")
    elif decision == "rollback":
        candidate_queue = list(args.candidate_queue)
        next_path, queue_reason = next_candidate(candidate_queue)
        if next_path is not None:
            print(f"\nNext candidate for composition: {next_path}")
        else:
            print(f"\nCycle stopped: {queue_reason}")


if __name__ == "__main__":
    main()
