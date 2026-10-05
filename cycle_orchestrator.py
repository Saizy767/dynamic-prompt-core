"""
Stage 3 cycle orchestrator.

Drives the full optimization loop across a configured number of rounds. Each
round runs the active prompt version on dev, computes metrics, updates the
thesis bank and clusters, selects rule candidates, composes a new prompt
version, runs it on dev, and delegates the accept-or-rollback decision to the
version comparator. The orchestrator maintains the round counter, the candidate
queue, the rollback state, and produces a per-round report plus a final summary.
It is the top-level entry point of the system.

Usage:
    python cycle_orchestrator.py
    python cycle_orchestrator.py --config config.toml --endpoint http://127.0.0.1:8080/v1
    python cycle_orchestrator.py --resume data/results/state_*.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from prompts import PromptArtifact, CLASSIFICATION_PROMPT_V0

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_MAX_ROUNDS = 5
DEFAULT_MAX_CONSECUTIVE_ROLLBACKS = 2
DEFAULT_DEV_SPLIT = "dev"
DEFAULT_HOLDOUT_SPLIT = "holdout"
DEFAULT_STOP_ON_FIRST_ERROR = False
DEFAULT_DUMP_STATE_AFTER_EACH_ROUND = True
DEFAULT_REPORT_FORMAT = "json"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"

VALID_STOP_REASONS = (
    "max_rounds_reached",
    "max_rollbacks_reached",
    "candidate_queue_exhausted",
    "unrecoverable_error",
)


class CycleOrchestratorError(ValueError):
    """Raised when configuration is invalid or a cycle step fails."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class CycleOrchestratorConfig:
    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_consecutive_rollbacks: int = DEFAULT_MAX_CONSECUTIVE_ROLLBACKS
    dev_split: str = DEFAULT_DEV_SPLIT
    holdout_split: str = DEFAULT_HOLDOUT_SPLIT
    prompt_store_path: str = ""
    thesis_bank_path: str = ""
    candidate_queue_path: str = ""
    stop_on_first_error: bool = DEFAULT_STOP_ON_FIRST_ERROR
    dump_state_after_each_round: bool = DEFAULT_DUMP_STATE_AFTER_EACH_ROUND
    report_format: str = DEFAULT_REPORT_FORMAT
    output_dir: str = DEFAULT_OUTPUT_DIR
    config_path: str = DEFAULT_CONFIG_PATH
    duration: Optional[int] = None

    @classmethod
    def from_config(
        cls, config_path: str = DEFAULT_CONFIG_PATH
    ) -> "CycleOrchestratorConfig":
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("cycle_orchestrator", {})
        return cls(
            max_rounds=int(s.get("max_rounds", DEFAULT_MAX_ROUNDS)),
            max_consecutive_rollbacks=int(
                s.get("max_consecutive_rollbacks", DEFAULT_MAX_CONSECUTIVE_ROLLBACKS)
            ),
            dev_split=s.get("dev_split", DEFAULT_DEV_SPLIT),
            holdout_split=s.get("holdout_split", DEFAULT_HOLDOUT_SPLIT),
            prompt_store_path=s.get("prompt_store_path", ""),
            thesis_bank_path=s.get("thesis_bank_path", ""),
            candidate_queue_path=s.get("candidate_queue_path", ""),
            stop_on_first_error=bool(
                s.get("stop_on_first_error", DEFAULT_STOP_ON_FIRST_ERROR)
            ),
            dump_state_after_each_round=bool(
                s.get(
                    "dump_state_after_each_round",
                    DEFAULT_DUMP_STATE_AFTER_EACH_ROUND,
                )
            ),
            report_format=s.get("report_format", DEFAULT_REPORT_FORMAT),
            output_dir=s.get("output_dir", DEFAULT_OUTPUT_DIR),
            config_path=config_path,
        )


# --------------------------------------------------------------------------- #
#  Cycle state
# --------------------------------------------------------------------------- #
@dataclass
class CycleState:
    round_counter: int = 0
    active_version: str = "classify-v0"
    active_version_path: Optional[str] = None
    thesis_bank_path: Optional[str] = None
    clusters_path: Optional[str] = None
    candidate_queue: List[str] = field(default_factory=list)
    rollback_counter: int = 0
    latest_report_path: Optional[str] = None
    accepted_history: List[str] = field(default_factory=list)
    rollback_history: List[Dict[str, Any]] = field(default_factory=list)
    run_id: str = ""
    active_prompt: Optional[PromptArtifact] = None
    dataset_artifact: Optional[str] = None
    all_changed_decisions: List[Dict[str, Any]] = field(default_factory=list)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
#  Initial state loading
# --------------------------------------------------------------------------- #
def load_active_prompt(config: CycleOrchestratorConfig) -> PromptArtifact:
    """Return the active PromptArtifact for the cycle.

    On a fresh run (prompt_store_path empty) returns CLASSIFICATION_PROMPT_V0.
    When prompt_store_path points to a composed prompt_v*_*_*.json artifact,
    loads it via prompt_composer.load_prompt_version and builds a PromptArtifact
    via version_comparator.build_prompt_artifact.

    Raises CycleOrchestratorError when the path is set but the artifact cannot
    be loaded.
    """
    path = config.prompt_store_path
    if not path:
        return CLASSIFICATION_PROMPT_V0

    if not os.path.isfile(path):
        raise CycleOrchestratorError(
            f"active prompt artifact not found at prompt_store_path={path!r}"
        )

    import prompt_composer
    import version_comparator

    try:
        version_dict = prompt_composer.load_prompt_version(path)
        return version_comparator.build_prompt_artifact(version_dict)
    except Exception as exc:
        raise CycleOrchestratorError(
            f"failed to load active prompt from {path!r}: {exc}"
        ) from exc


def load_dataset_splits(
    config: CycleOrchestratorConfig,
) -> Tuple[str, str]:
    """Return the prepared dataset artifact path for dev and holdout splits.

    Reads the dataset_artifact path from [runner] config when the orchestrator
    config does not override it. Raises CycleOrchestratorError when the artifact
    is absent.
    """
    from runner import RunnerConfig

    runner_config = RunnerConfig.from_config(config.config_path)
    dataset_artifact = runner_config.dataset_artifact

    if not dataset_artifact or not os.path.isfile(dataset_artifact):
        raise CycleOrchestratorError(
            f"dataset artifact not found: {dataset_artifact!r} "
            "(set [runner].dataset_artifact in config.toml)"
        )

    return dataset_artifact, dataset_artifact


def load_initial_state(
    config: CycleOrchestratorConfig,
    run_id: Optional[str] = None,
) -> CycleState:
    """Build the initial CycleState for a fresh cycle.

    Loads the active prompt and dataset splits, starts with an empty thesis
    bank, round counter 0, rollback counter 0, and empty histories.
    """
    active_prompt = load_active_prompt(config)
    dataset_artifact, _ = load_dataset_splits(config)

    if run_id is None:
        run_id = f"cycle-{_utc_timestamp()}"

    return CycleState(
        round_counter=0,
        active_version=active_prompt.version,
        active_version_path=config.prompt_store_path or None,
        thesis_bank_path=config.thesis_bank_path or None,
        clusters_path=None,
        candidate_queue=[],
        rollback_counter=0,
        latest_report_path=None,
        accepted_history=[],
        rollback_history=[],
        run_id=run_id,
        active_prompt=active_prompt,
        dataset_artifact=dataset_artifact,
        all_changed_decisions=[],
    )


# --------------------------------------------------------------------------- #
#  Round execution — step helpers
# --------------------------------------------------------------------------- #
async def _run_on_split(
    prompt_artifact: PromptArtifact,
    split: str,
    config: CycleOrchestratorConfig,
    async_task: Any,
    run_id: str,
) -> str:
    """Run a prompt version on a dataset split and return the results path."""
    from runner import BaselineRunner, RunnerConfig

    runner_config = RunnerConfig.from_config(config.config_path)
    runner = BaselineRunner(
        task=async_task,
        config=runner_config,
        split=split,
        run_id=run_id,
        dataset_artifact=runner_config.dataset_artifact,
        duration=config.duration,
        classify_prompt=prompt_artifact,
    )

    from dataset import load_artifact

    ds = load_artifact(runner_config.dataset_artifact)
    examples = getattr(ds, split)
    if not examples:
        raise CycleOrchestratorError(
            f"split {split!r} has no examples in the dataset artifact"
        )

    _, artifact_path = await runner.run()
    return artifact_path


async def run_active_on_dev(
    state: CycleState,
    config: CycleOrchestratorConfig,
    async_task: Any,
) -> str:
    """Step 1: run the active prompt on dev and return the results path."""
    run_id = f"{state.run_id}-r{state.round_counter + 1}-{state.active_version}-dev"
    return await _run_on_split(
        state.active_prompt, config.dev_split, config, async_task, run_id
    )


async def run_on_holdout(
    prompt_artifact: PromptArtifact,
    version_label: str,
    state: CycleState,
    config: CycleOrchestratorConfig,
    async_task: Any,
) -> Optional[str]:
    """Run a prompt version on holdout. Returns the results path or None on failure."""
    run_id = (
        f"{state.run_id}-r{state.round_counter + 1}-{version_label}-holdout"
    )
    try:
        return await _run_on_split(
            prompt_artifact, config.holdout_split, config, async_task, run_id
        )
    except Exception as exc:
        log.warning("holdout run failed for %s: %s", version_label, exc)
        return None


def compute_dev_metrics(
    results_path: str,
    config: CycleOrchestratorConfig,
) -> Dict[str, Any]:
    """Step 2: compute metrics from a results artifact on dev."""
    import metrics as metrics_module

    m_config = metrics_module.MetricsConfig.from_config(config.config_path)
    rows = metrics_module.load_results(results_path)
    return metrics_module.compute_metrics(rows, m_config, results_path)


def compute_split_metrics(
    results_path: str,
    config: CycleOrchestratorConfig,
) -> Dict[str, Any]:
    """Compute metrics from a results artifact on any split."""
    return compute_dev_metrics(results_path, config)


def update_thesis_bank(
    results_path: str,
    config: CycleOrchestratorConfig,
    run_id: str,
    prompt_version: str,
) -> Tuple[str, str]:
    """Step 3: run thesis_analyzer on the results and return (thesis_bank_path, clusters_path).

    The thesis bank dump contains both theses and clusters in a single file;
    clusters_path is the same file.
    """
    import thesis_analyzer

    ta_config = thesis_analyzer.ThesisAnalyzerConfig.from_config(config.config_path)
    rows = thesis_analyzer.load_results(results_path)
    bank, _name_parts = thesis_analyzer.analyze(rows, ta_config, results_path)
    dump_path = thesis_analyzer.write_dump(
        bank, run_id, prompt_version, ta_config, ta_config.output_dir
    )
    return dump_path, dump_path


def select_candidates(
    thesis_bank_path: str,
    config: CycleOrchestratorConfig,
    run_id: str,
    prompt_version: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Step 4: run rule_candidate_selector and return (artifact_path, candidate_queue).

    The candidate_queue is the list of candidate dicts from the artifact. Each
    candidate is a dict with cluster_id, rank, precision, frequency, etc.
    """
    import rule_candidate_selector as rcs

    rcs_config = rcs.RuleCandidateSelectorConfig.from_config(config.config_path)
    artifact = rcs.load_thesis_artifact(thesis_bank_path)
    candidates, already_in_prompt, counters = rcs.select_candidates(
        artifact, rcs_config
    )
    artifact_path = rcs.write_candidates(
        candidates,
        already_in_prompt,
        counters,
        run_id,
        prompt_version,
        rcs_config,
        rcs_config.output_dir,
    )
    return artifact_path, candidates


async def compose_new_version(
    candidate: Any,
    active_prompt: PromptArtifact,
    config: CycleOrchestratorConfig,
    async_task: Any,
    thesis_bank_path: Optional[str],
    run_id: str,
) -> Tuple[PromptArtifact, str, List[int]]:
    """Step 5: compose a new prompt version from a candidate.

    Returns (new_prompt_artifact, artifact_path, source_candidates).

    The candidate may be a dict (from the selector) or a path to a candidate
    artifact. When it is a dict, a temporary single-candidate artifact is
    assembled for the composer.
    """
    import prompt_composer

    pc_config = prompt_composer.PromptComposerConfig.from_config(config.config_path)

    if isinstance(candidate, dict):
        candidates = [candidate]
    elif isinstance(candidate, str):
        artifact = prompt_composer.load_candidates(candidate)
        candidates = artifact["candidates"]
    else:
        raise CycleOrchestratorError(
            f"candidate must be a dict or a path string, got {type(candidate).__name__}"
        )

    base_layers = prompt_composer.load_base_layers(active_prompt)
    centroids = prompt_composer.load_centroids(thesis_bank_path)

    prompt_artifact, rules_with_lineage, counters, rejected_rules = await prompt_composer.compose(
        candidates, base_layers, centroids, pc_config, async_task,
        active_prompt.version,
    )

    tokenizer = getattr(async_task, "_tokenizer", None)
    prompt_composer.validate_limits(prompt_artifact, pc_config, tokenizer)

    source_candidates = [r["cluster_id"] for r in rules_with_lineage]
    artifact_path = prompt_composer.write_prompt_version(
        prompt_artifact,
        rules_with_lineage,
        source_candidates,
        active_prompt.version,
        run_id,
        counters,
        pc_config,
        pc_config.output_dir,
        rejected_rules,
    )
    return prompt_artifact, artifact_path, source_candidates


async def run_new_on_dev(
    new_prompt: PromptArtifact,
    state: CycleState,
    config: CycleOrchestratorConfig,
    async_task: Any,
) -> str:
    """Step 6: run the new prompt version on dev and return the results path."""
    run_id = f"{state.run_id}-r{state.round_counter + 1}-{new_prompt.version}-dev"
    return await _run_on_split(
        new_prompt, config.dev_split, config, async_task, run_id
    )


def decide_and_update(
    state: CycleState,
    new_prompt: PromptArtifact,
    new_version_path: str,
    source_candidates: List[int],
    metrics_new: Dict[str, Any],
    metrics_active: Dict[str, Any],
    results_active: str,
    results_new: str,
    config: CycleOrchestratorConfig,
) -> Tuple[str, str, CycleState, List[Dict[str, Any]]]:
    """Steps 7–8: decide accept/rollback and update the state.

    Returns (decision, reason, new_state, changed_decisions).
    On accept: updates the active version, resets the rollback counter, and
    updates the thesis bank in_prompt flags.
    On rollback: increments the rollback counter and pops the next candidate.
    """
    import version_comparator as vc

    vc_config = vc.VersionComparatorConfig.from_config(config.config_path)
    decision, reason = vc.decide(metrics_new, metrics_active, vc_config)

    changed: List[Dict[str, Any]] = []
    try:
        changed = vc.changed_decisions(results_active, results_new, config.config_path)
    except Exception as exc:
        log.warning("changed_decisions failed: %s", exc)

    new_state = _copy_state(state)

    if decision == "accept":
        new_state = update_active_version(new_state, new_prompt, new_version_path)
        new_state.rollback_counter = vc.update_rollback_count(
            state.rollback_counter, decision
        )
        if new_state.thesis_bank_path:
            try:
                update_in_prompt_flags(
                    new_state.thesis_bank_path, source_candidates, config
                )
            except Exception as exc:
                log.warning("update_in_prompt_flags failed: %s", exc)
    else:
        new_state.rollback_counter = vc.update_rollback_count(
            state.rollback_counter, decision
        )
        new_state.rollback_history.append(
            {
                "round": state.round_counter + 1,
                "version": new_prompt.version,
                "reason": reason,
            }
        )

    return decision, reason, new_state, changed


def _copy_state(state: CycleState) -> CycleState:
    """Return a shallow copy of the state with copied list fields."""
    return CycleState(
        round_counter=state.round_counter,
        active_version=state.active_version,
        active_version_path=state.active_version_path,
        thesis_bank_path=state.thesis_bank_path,
        clusters_path=state.clusters_path,
        candidate_queue=list(state.candidate_queue),
        rollback_counter=state.rollback_counter,
        latest_report_path=state.latest_report_path,
        accepted_history=list(state.accepted_history),
        rollback_history=list(state.rollback_history),
        run_id=state.run_id,
        active_prompt=state.active_prompt,
        dataset_artifact=state.dataset_artifact,
        all_changed_decisions=list(state.all_changed_decisions),
    )


# --------------------------------------------------------------------------- #
#  Round counter, candidate queue, rollback state
# --------------------------------------------------------------------------- #
def increment_round_counter(state: CycleState) -> CycleState:
    """Return a new state with round_counter + 1 (design D3)."""
    new_state = _copy_state(state)
    new_state.round_counter = state.round_counter + 1
    return new_state


def check_stop(
    state: CycleState, config: CycleOrchestratorConfig
) -> Tuple[bool, Optional[str]]:
    """Check whether the cycle should stop.

    Returns (should_stop, reason). Reasons:
    - max_rounds_reached when round_counter >= max_rounds
    - max_rollbacks_reached when rollback_counter >= max_consecutive_rollbacks
    - candidate_queue_exhausted when a rollback left the queue empty
    - (False, None) otherwise
    """
    if state.round_counter >= config.max_rounds:
        return True, "max_rounds_reached"
    if state.rollback_counter >= config.max_consecutive_rollbacks:
        return True, "max_rollbacks_reached"
    return False, None


def refill_queue(
    state: CycleState, candidates: List[Dict[str, Any]]
) -> CycleState:
    """Replace the candidate queue with the given candidates (design D4)."""
    new_state = _copy_state(state)
    new_state.candidate_queue = list(candidates)
    return new_state


# --------------------------------------------------------------------------- #
#  Accept handling and thesis bank update
# --------------------------------------------------------------------------- #
def update_active_version(
    state: CycleState,
    new_prompt: PromptArtifact,
    new_path: str,
) -> CycleState:
    """Mark the new version as active, archive the old, record lineage (D11)."""
    new_state = _copy_state(state)
    new_state.accepted_history.append(state.active_version)
    new_state.active_version = new_prompt.version
    new_state.active_version_path = new_path
    new_state.active_prompt = new_prompt
    return new_state


def update_in_prompt_flags(
    thesis_bank_path: str,
    accepted_source_candidates: List[int],
    config: CycleOrchestratorConfig,
) -> None:
    """Set in_prompt on theses in accepted clusters, clear others, and persist (D12)."""
    import thesis_analyzer

    ta_config = thesis_analyzer.ThesisAnalyzerConfig.from_config(config.config_path)
    bank = thesis_analyzer.load_dump(thesis_bank_path, ta_config)

    accepted_set = set(accepted_source_candidates)
    for entry in bank.theses.values():
        cid = entry.cluster_id
        entry.in_prompt = cid is not None and cid in accepted_set

    with open(thesis_bank_path, "r", encoding="utf-8") as f:
        original = json.load(f)
    run_id = original.get("metadata", {}).get("run_id", f"cycle-{_utc_timestamp()}")
    prompt_version = original.get("metadata", {}).get("prompt_version", "updated")

    new_path = thesis_analyzer.write_dump(
        bank, run_id, prompt_version, ta_config, ta_config.output_dir
    )
    with open(new_path, "r", encoding="utf-8") as f:
        updated = json.load(f)
    with open(thesis_bank_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
#  Per-round report and final summary
# --------------------------------------------------------------------------- #
def write_report(
    round_number: int,
    active_version_at_start: str,
    new_version: str,
    decision: str,
    metrics_dev_active: Dict[str, Any],
    metrics_dev_new: Dict[str, Any],
    metrics_holdout_active: Optional[Dict[str, Any]],
    metrics_holdout_new: Optional[Dict[str, Any]],
    rollback_counter: int,
    changed_decisions: List[Dict[str, Any]],
    next_action: str,
    run_id: str,
    config: CycleOrchestratorConfig,
) -> str:
    """Serialize a per-round report to JSON (design D7).

    Filename: report_{run_id}_round{N}_{timestamp}.json
    """
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = _utc_timestamp()
    name = f"report_{run_id}_round{round_number}_{timestamp}.json"
    path = os.path.join(config.output_dir, name)

    dump = {
        "round_number": round_number,
        "active_version_at_start": active_version_at_start,
        "new_version": new_version,
        "decision": decision,
        "metrics_dev_active": metrics_dev_active,
        "metrics_dev_new": metrics_dev_new,
        "metrics_holdout_active": metrics_holdout_active,
        "metrics_holdout_new": metrics_holdout_new,
        "rollback_counter": rollback_counter,
        "changed_decisions": changed_decisions,
        "next_action": next_action,
        "run_id": run_id,
        "timestamp": timestamp,
        "created_at": _utc_iso(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


def load_report(path: str) -> Dict[str, Any]:
    """Restore a per-round report without recomputation."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(
    state: CycleState,
    final_dev_metrics: Optional[Dict[str, Any]],
    final_holdout_metrics: Optional[Dict[str, Any]],
    stop_reason: str,
    run_id: str,
    config: CycleOrchestratorConfig,
) -> str:
    """Serialize the final summary to JSON (design D7).

    Filename: summary_{run_id}_{timestamp}.json
    """
    if stop_reason not in VALID_STOP_REASONS:
        raise CycleOrchestratorError(
            f"invalid stop_reason {stop_reason!r}; expected one of {VALID_STOP_REASONS}"
        )

    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = _utc_timestamp()
    name = f"summary_{run_id}_{timestamp}.json"
    path = os.path.join(config.output_dir, name)

    dump = {
        "total_rounds": state.round_counter,
        "stop_reason": stop_reason,
        "final_active_version": state.active_version,
        "final_dev_metrics": final_dev_metrics,
        "final_holdout_metrics": final_holdout_metrics,
        "accepted_history": state.accepted_history,
        "rollback_history": state.rollback_history,
        "all_changed_decisions": state.all_changed_decisions,
        "run_id": run_id,
        "timestamp": timestamp,
        "created_at": _utc_iso(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
#  State dump and resumability
# --------------------------------------------------------------------------- #
def _state_to_dict(state: CycleState) -> Dict[str, Any]:
    """Convert CycleState to a JSON-safe dict."""
    d = asdict(state)
    d["active_prompt"] = None
    return d


def dump_state(
    state: CycleState, config: CycleOrchestratorConfig
) -> str:
    """Serialize the full cycle state to JSON (design D6).

    Filename: state_{run_id}_{timestamp}.json
    """
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = _utc_timestamp()
    name = f"state_{state.run_id}_{timestamp}.json"
    path = os.path.join(config.output_dir, name)

    dump = _state_to_dict(state)
    dump["dump_timestamp"] = timestamp

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


def load_state(path: str) -> CycleState:
    """Restore a CycleState from a dumped JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    d.pop("dump_timestamp", None)
    d.pop("active_prompt", None)

    return CycleState(
        round_counter=d.get("round_counter", 0),
        active_version=d.get("active_version", "classify-v0"),
        active_version_path=d.get("active_version_path"),
        thesis_bank_path=d.get("thesis_bank_path"),
        clusters_path=d.get("clusters_path"),
        candidate_queue=d.get("candidate_queue", []),
        rollback_counter=d.get("rollback_counter", 0),
        latest_report_path=d.get("latest_report_path"),
        accepted_history=d.get("accepted_history", []),
        rollback_history=d.get("rollback_history", []),
        run_id=d.get("run_id", ""),
        active_prompt=None,
        dataset_artifact=d.get("dataset_artifact"),
        all_changed_decisions=d.get("all_changed_decisions", []),
    )


def _restore_active_prompt(state: CycleState, config: CycleOrchestratorConfig) -> CycleState:
    """Rebuild the active PromptArtifact on resume."""
    if state.active_version_path and os.path.isfile(state.active_version_path):
        import prompt_composer
        import version_comparator

        version_dict = prompt_composer.load_prompt_version(state.active_version_path)
        state.active_prompt = version_comparator.build_prompt_artifact(version_dict)
    else:
        state.active_prompt = CLASSIFICATION_PROMPT_V0
    return state


async def resume_from_state(
    state_path: str,
    config: CycleOrchestratorConfig,
    async_task: Any,
    endpoint: str = DEFAULT_ENDPOINT,
) -> CycleState:
    """Load a dumped state and continue the cycle from round_counter + 1.

    Skips already-completed rounds (design D8). Returns the final state.
    """
    state = load_state(state_path)
    state = _restore_active_prompt(state, config)

    if state.dataset_artifact is None:
        dataset_artifact, _ = load_dataset_splits(config)
        state.dataset_artifact = dataset_artifact

    log.info(
        "resuming cycle %s from round %d (counter=%d)",
        state.run_id, state.round_counter + 1, state.round_counter,
    )

    return await _run_cycle_loop(state, config, async_task, endpoint)


# --------------------------------------------------------------------------- #
#  Cycle logging
# --------------------------------------------------------------------------- #
def _cycle_log_path(run_id: str, config: CycleOrchestratorConfig) -> str:
    return os.path.join(config.output_dir, f"cycle_log_{run_id}.jsonl")


def log_event(
    event: str,
    round: int,
    details: Dict[str, Any],
    run_id: str,
    config: CycleOrchestratorConfig,
) -> None:
    """Append one event to the append-only cycle log (design D13)."""
    os.makedirs(config.output_dir, exist_ok=True)
    path = _cycle_log_path(run_id, config)
    entry = {
        "timestamp": _utc_iso(),
        "round": round,
        "event": event,
        "details": details,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_cycle_log(
    path: str, round: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Read the JSONL cycle log, optionally filtered by round number."""
    events: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if round is not None and entry.get("round") != round:
                continue
            events.append(entry)
    return events


# --------------------------------------------------------------------------- #
#  Round execution — the nine-step sequence
# --------------------------------------------------------------------------- #
async def run_round(
    state: CycleState,
    config: CycleOrchestratorConfig,
    async_task: Any,
    endpoint: str = DEFAULT_ENDPOINT,
) -> Tuple[CycleState, str]:
    """Execute a single round (design D2).

    Chains the nine steps: run active on dev, compute metrics, update thesis
    bank, select candidates, compose, run new on dev, decide, update state,
    write report. On rollback within the round, consume the candidate queue
    (steps 5–7) until a candidate is accepted or the queue is exhausted.

    Returns (updated_state, report_path).
    """
    round_number = state.round_counter + 1
    active_version_at_start = state.active_version
    log.info("=== round %d start (active=%s) ===", round_number, active_version_at_start)
    log_event("round_start", round_number, {"active_version": active_version_at_start},
              state.run_id, config)

    active_prompt = state.active_prompt or CLASSIFICATION_PROMPT_V0

    # Step 1: run active on dev
    results_active_dev = await run_active_on_dev(state, config, async_task)

    # Step 2: compute dev metrics for active
    metrics_dev_active = compute_dev_metrics(results_active_dev, config)

    # Step 3: update thesis bank
    thesis_run_id = f"{state.run_id}-r{round_number}-{active_version_at_start}"
    thesis_bank_path, clusters_path = update_thesis_bank(
        results_active_dev, config, thesis_run_id, active_version_at_start
    )

    # Step 4: select candidates
    candidate_run_id = f"{state.run_id}-r{round_number}-{active_version_at_start}"
    candidate_artifact_path, candidates = select_candidates(
        thesis_bank_path, config, candidate_run_id, active_version_at_start
    )

    working = _copy_state(state)
    working.thesis_bank_path = thesis_bank_path
    working.clusters_path = clusters_path
    working = refill_queue(working, candidates)

    # Holdout for active (best-effort, logged only)
    metrics_holdout_active: Optional[Dict[str, Any]] = None
    results_active_holdout = await run_on_holdout(
        active_prompt, active_version_at_start, state, config, async_task
    )
    if results_active_holdout:
        try:
            metrics_holdout_active = compute_split_metrics(
                results_active_holdout, config
            )
        except Exception as exc:
            log.warning("holdout metrics failed for active: %s", exc)

    # Steps 5–7: compose, run, decide — retry on rollback with next candidate
    decision = "rollback"
    reason = "no candidates"
    new_prompt: Optional[PromptArtifact] = None
    new_version_path: str = ""
    source_candidates: List[int] = []
    metrics_dev_new: Optional[Dict[str, Any]] = None
    metrics_holdout_new: Optional[Dict[str, Any]] = None
    changed: List[Dict[str, Any]] = []
    results_new_dev = ""

    if not working.candidate_queue:
        log.warning("round %d: no candidates produced by selector", round_number)
    else:
        while working.candidate_queue:
            candidate = working.candidate_queue.pop(0)

            # Step 5: compose
            compose_run_id = f"{state.run_id}-r{round_number}-compose"
            new_prompt, new_version_path, source_candidates = (
                await compose_new_version(
                    candidate, active_prompt, config, async_task,
                    thesis_bank_path, compose_run_id,
                )
            )

            # Step 6: run new on dev
            results_new_dev = await run_new_on_dev(
                new_prompt, state, config, async_task
            )
            metrics_dev_new = compute_dev_metrics(results_new_dev, config)

            # Holdout for new (best-effort)
            results_new_holdout = await run_on_holdout(
                new_prompt, new_prompt.version, state, config, async_task
            )
            if results_new_holdout:
                try:
                    metrics_holdout_new = compute_split_metrics(
                        results_new_holdout, config
                    )
                except Exception as exc:
                    log.warning("holdout metrics failed for new: %s", exc)

            # Step 7–8: decide
            decision, reason, working, changed = decide_and_update(
                working, new_prompt, new_version_path, source_candidates,
                metrics_dev_new, metrics_dev_active,
                results_active_dev, results_new_dev, config,
            )

            log_event("decision", round_number,
                      {"decision": decision, "reason": reason,
                       "new_version": new_prompt.version,
                       "active_version": active_version_at_start},
                      state.run_id, config)
            log_event(decision, round_number,
                      {"new_version": new_prompt.version, "reason": reason},
                      state.run_id, config)

            if decision == "accept":
                break

            log.info(
                "round %d: rolled back %s (%s); trying next candidate (%d remaining)",
                round_number, new_prompt.version, reason, len(working.candidate_queue),
            )

    # Determine next action
    if decision == "accept":
        next_action = "continue"
    elif not working.candidate_queue:
        next_action = "stop:candidate_queue_exhausted"
    else:
        next_action = "continue"

    # Step 9: write report
    report_path = write_report(
        round_number=round_number,
        active_version_at_start=active_version_at_start,
        new_version=new_prompt.version if new_prompt else "(none)",
        decision=decision,
        metrics_dev_active=metrics_dev_active,
        metrics_dev_new=metrics_dev_new or {},
        metrics_holdout_active=metrics_holdout_active,
        metrics_holdout_new=metrics_holdout_new,
        rollback_counter=working.rollback_counter,
        changed_decisions=changed,
        next_action=next_action,
        run_id=state.run_id,
        config=config,
    )
    working.latest_report_path = report_path
    working.all_changed_decisions.extend(changed)

    # Increment round counter (step complete)
    working = increment_round_counter(working)

    log.info("=== round %d end (decision=%s) ===", round_number, decision)
    log_event("round_end", round_number,
              {"decision": decision, "rollback_counter": working.rollback_counter,
               "next_action": next_action},
              state.run_id, config)

    return working, report_path


# --------------------------------------------------------------------------- #
#  Cycle loop
# --------------------------------------------------------------------------- #
async def _run_cycle_loop(
    state: CycleState,
    config: CycleOrchestratorConfig,
    async_task: Any,
    endpoint: str,
) -> CycleState:
    """Run rounds until a stop condition is reached."""
    stop_reason: str = "max_rounds_reached"
    final_dev_metrics: Optional[Dict[str, Any]] = None
    final_holdout_metrics: Optional[Dict[str, Any]] = None

    while True:
        should_stop, reason = check_stop(state, config)
        if should_stop:
            stop_reason = reason or "max_rounds_reached"
            log.info("cycle stopping before round %d: %s",
                     state.round_counter + 1, stop_reason)
            log_event("stop", state.round_counter,
                      {"stop_reason": stop_reason}, state.run_id, config)
            break

        try:
            state, report_path = await run_round(
                state, config, async_task, endpoint
            )
        except Exception as exc:
            log.error("unrecoverable error in round %d: %s",
                      state.round_counter + 1, exc)
            log_event("error", state.round_counter + 1,
                      {"step": "run_round", "error": str(exc)},
                      state.run_id, config)
            if config.stop_on_first_error or True:
                stop_reason = "unrecoverable_error"
                log_event("stop", state.round_counter,
                          {"stop_reason": stop_reason,
                           "error": str(exc)}, state.run_id, config)
                break

        if config.dump_state_after_each_round:
            try:
                state_path = dump_state(state, config)
                log.info("state dumped to %s", state_path)
            except Exception as exc:
                log.warning("state dump failed: %s", exc)

        # Check stop after the round (rollback / queue exhaustion)
        should_stop, reason = check_stop(state, config)
        if should_stop:
            stop_reason = reason or "max_rounds_reached"
            log.info("cycle stopping after round %d: %s",
                     state.round_counter, stop_reason)
            log_event("stop", state.round_counter,
                      {"stop_reason": stop_reason}, state.run_id, config)
            break

        if not state.candidate_queue and state.rollback_counter > 0:
            stop_reason = "candidate_queue_exhausted"
            log.info("cycle stopping: candidate queue exhausted")
            log_event("stop", state.round_counter,
                      {"stop_reason": stop_reason}, state.run_id, config)
            break

    # Compute final metrics for the active version
    if state.active_prompt is not None and state.round_counter > 0:
        try:
            final_run_id = f"{state.run_id}-final-{state.active_version}-dev"
            final_results = await _run_on_split(
                state.active_prompt, config.dev_split, config, async_task,
                final_run_id,
            )
            final_dev_metrics = compute_dev_metrics(final_results, config)

            final_holdout_results = await run_on_holdout(
                state.active_prompt, state.active_version, state, config,
                async_task,
            )
            if final_holdout_results:
                final_holdout_metrics = compute_split_metrics(
                    final_holdout_results, config
                )
        except Exception as exc:
            log.warning("final metrics computation failed: %s", exc)

    summary_path = write_summary(
        state, final_dev_metrics, final_holdout_metrics,
        stop_reason, state.run_id, config,
    )
    log.info("final summary written to %s", summary_path)
    _print_cycle_summary(state, stop_reason, summary_path)

    return state


async def run_cycle(
    config: CycleOrchestratorConfig,
    endpoint: str = DEFAULT_ENDPOINT,
    run_id: Optional[str] = None,
) -> CycleState:
    """Run a full optimization cycle from a fresh state."""
    from asyncTask import AsyncTask

    state = load_initial_state(config, run_id=run_id)

    async_task = AsyncTask(
        config_path=config.config_path,
        endpoint=endpoint,
    )

    log.info(
        "starting cycle %s: max_rounds=%d active=%s",
        state.run_id, config.max_rounds, state.active_version,
    )

    return await _run_cycle_loop(state, config, async_task, endpoint)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _print_cycle_summary(
    state: CycleState, stop_reason: str, summary_path: str
) -> None:
    print("\n" + "=" * 60)
    print(f"Cycle complete  (run_id={state.run_id})")
    print("=" * 60)
    print(
        f"  total_rounds     : {state.round_counter}\n"
        f"  stop_reason      : {stop_reason}\n"
        f"  final_version    : {state.active_version}\n"
        f"  accepted_history : {state.accepted_history}\n"
        f"  rollbacks        : {len(state.rollback_history)}"
    )
    print("-" * 60)
    print(f"  summary: {summary_path}")
    print("=" * 60)


def _print_round_summary(
    round_number: int, decision: str, active: str, new: str
) -> None:
    print(
        f"  round {round_number}: {decision:8s}  "
        f"{active} -> {new}"
    )


async def _main_async(args: argparse.Namespace) -> None:
    config = CycleOrchestratorConfig.from_config(args.config)
    if args.duration is not None:
        config.duration = args.duration

    if args.resume:
        from asyncTask import AsyncTask

        async_task = AsyncTask(
            config_path=config.config_path,
            endpoint=args.endpoint,
        )
        await resume_from_state(args.resume, config, async_task, args.endpoint)
    else:
        await run_cycle(config, endpoint=args.endpoint, run_id=args.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 cycle orchestrator — drives the full optimization loop"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="Path to config.toml (default: config.toml)",
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT,
        help="LLM server endpoint",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a dumped state JSON to resume from",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Override the cycle run id (default: cycle-<timestamp>)",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Limit examples per split (passed to BaselineRunner)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
