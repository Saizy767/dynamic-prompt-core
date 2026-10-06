"""
Stage 4 cross-task transfer.

Transfers an optimized system prompt's base layers, thesis collection, and
cluster centroids from a source task to a structurally compatible target task,
reformulates the task layer for the target domain, clears the rules, and
re-runs the optimization cycle on the target dataset. Produces a transfer
artifact, an effectiveness report, and an optional cold-start comparison,
while keeping source and target artifacts fully isolated.

Usage:
    python cross_task_transfer.py --config config.toml run
    python cross_task_transfer.py --config config.toml run --endpoint http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, NamedTuple, cast

from dynamic_prompt_core.domain.prompts import (
    PromptArtifact,
    PromptLayer,
    render,
)

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_LOG_PATH = "data/cross_task_transfer.jsonl"
DEFAULT_MAX_TRANSFER_THESES: int | None = None
DEFAULT_TRANSFER_CLUSTERS = True
DEFAULT_COMPARE_WITH_COLD_START = False
DEFAULT_MAX_ROUNDS_TARGET = 5
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"

VALID_METRICS = ("accuracy", "macro_f1", "minority_f1")

MAX_TASK_LAYER_CHARS = 500

_REFORMULATION_SYSTEM_PROMPT = (
    "You are a prompt-engineering assistant. You reformulate the 'Task' layer "
    "of a classification prompt for a new target domain. Keep the same "
    "structure: describe what to classify and how to interpret the two classes. "
    "Reply with only the reformulated task description, no markdown, no extra text."
)


class CrossTaskTransferError(ValueError):
    """Raised when configuration is invalid or a transfer step fails."""


# --------------------------------------------------------------------------- #
#  Task spec and result dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class TaskSpec:
    task_id: str = ""
    dataset_path: str = ""
    num_classes: int = 0
    class_labels: list[str] = field(default_factory=list)
    metric: str = "macro_f1"


@dataclass
class TransferResult:
    source_version: int = 0
    target_final_version: str = ""
    transferred_theses_count: int = 0
    transferred_clusters_count: int = 0
    target_metrics_start: dict[str, Any] = field(default_factory=dict)
    target_metrics_final: dict[str, Any] = field(default_factory=dict)
    improvement: dict[str, float] = field(
        default_factory=lambda: {"absolute": 0.0, "relative": 0.0}
    )
    stop_reason: str = ""


class SourceArtifacts(NamedTuple):
    prompt_record: dict[str, Any]
    thesis_bank: Any
    output_contract: str


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class CrossTaskTransferConfig:
    source_task: TaskSpec = field(default_factory=TaskSpec)
    target_task: TaskSpec = field(default_factory=TaskSpec)
    source_artifacts_path: str = ""
    target_artifacts_path: str = ""
    source_store_path: str = ""
    target_store_path: str = ""
    source_thesis_bank_path: str = ""
    max_transfer_theses: int | None = DEFAULT_MAX_TRANSFER_THESES
    transfer_clusters: bool = DEFAULT_TRANSFER_CLUSTERS
    compare_with_cold_start: bool = DEFAULT_COMPARE_WITH_COLD_START
    max_rounds_target: int = DEFAULT_MAX_ROUNDS_TARGET
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_path: str = DEFAULT_LOG_PATH
    config_path: str = DEFAULT_CONFIG_PATH

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_CONFIG_PATH) -> CrossTaskTransferConfig:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("cross_task_transfer", {})

        source_task = _build_task_spec(
            config.get("cross_task_transfer", {}).get("source_task", {}),
            "source",
        )
        target_task = _build_task_spec(
            config.get("cross_task_transfer", {}).get("target_task", {}),
            "target",
        )

        required_keys = {
            "source_artifacts_path": s.get("source_artifacts_path"),
            "target_artifacts_path": s.get("target_artifacts_path"),
            "source_store_path": s.get("source_store_path"),
            "target_store_path": s.get("target_store_path"),
            "source_thesis_bank_path": s.get("source_thesis_bank_path"),
        }
        for key, value in required_keys.items():
            if not value:
                raise CrossTaskTransferError(
                    f"cross_task_transfer config missing required key: {key}"
                )

        max_transfer_theses = s.get("max_transfer_theses")
        if max_transfer_theses is not None:
            max_transfer_theses = int(max_transfer_theses)

        return cls(
            source_task=source_task,
            target_task=target_task,
            source_artifacts_path=s["source_artifacts_path"],
            target_artifacts_path=s["target_artifacts_path"],
            source_store_path=s["source_store_path"],
            target_store_path=s["target_store_path"],
            source_thesis_bank_path=s["source_thesis_bank_path"],
            max_transfer_theses=max_transfer_theses,
            transfer_clusters=bool(s.get("transfer_clusters", DEFAULT_TRANSFER_CLUSTERS)),
            compare_with_cold_start=bool(
                s.get("compare_with_cold_start", DEFAULT_COMPARE_WITH_COLD_START)
            ),
            max_rounds_target=int(s.get("max_rounds_target", DEFAULT_MAX_ROUNDS_TARGET)),
            output_dir=s.get("output_dir", DEFAULT_OUTPUT_DIR),
            log_path=s.get("log_path", DEFAULT_LOG_PATH),
            config_path=config_path,
        )


def _build_task_spec(raw: dict[str, Any], role: str) -> TaskSpec:
    """Build a TaskSpec from a config sub-table, validating fields."""
    spec = TaskSpec(
        task_id=raw.get("task_id", ""),
        dataset_path=raw.get("dataset_path", ""),
        num_classes=int(raw.get("num_classes", 0)),
        class_labels=list(raw.get("class_labels", [])),
        metric=raw.get("metric", ""),
    )
    _validate_task_spec(spec, role)
    return spec


# --------------------------------------------------------------------------- #
#  Task spec validation and compatibility
# --------------------------------------------------------------------------- #
def _validate_task_spec(spec: TaskSpec, role: str) -> None:
    """Validate that *spec* has all required fields (design D2).

    Raises CrossTaskTransferError naming the missing or invalid field and the
    role ('source' or 'target').
    """
    if not spec.task_id:
        raise CrossTaskTransferError(f"{role} task missing required field: task_id")
    if not spec.dataset_path:
        raise CrossTaskTransferError(f"{role} task missing required field: dataset_path")
    if spec.num_classes <= 0:
        raise CrossTaskTransferError(f"{role} task missing or invalid field: num_classes")
    if not spec.class_labels:
        raise CrossTaskTransferError(f"{role} task missing required field: class_labels")
    if not spec.metric:
        raise CrossTaskTransferError(f"{role} task missing required field: metric")
    if spec.metric not in VALID_METRICS:
        raise CrossTaskTransferError(
            f"{role} task has invalid metric {spec.metric!r}; expected one of {VALID_METRICS}"
        )


def _check_compatibility(
    source_spec: TaskSpec,
    target_spec: TaskSpec,
    source_output_contract: str,
    target_output_contract: str = "",
) -> None:
    """Check that source and target have the same structure (design D3).

    Raises CrossTaskTransferError naming the mismatch when num_classes differs
    or the output contract strings differ.
    """
    if source_spec.num_classes != target_spec.num_classes:
        raise CrossTaskTransferError(
            f"structure mismatch: source has {source_spec.num_classes} classes "
            f"but target has {target_spec.num_classes} classes"
        )
    if target_output_contract and source_output_contract != target_output_contract:
        raise CrossTaskTransferError(
            "structure mismatch: output contract differs between source and "
            f"target:\n  source: {source_output_contract!r}\n  "
            f"target: {target_output_contract!r}"
        )


# --------------------------------------------------------------------------- #
#  Source artifact loading
# --------------------------------------------------------------------------- #
def _load_source_prompt(config: CrossTaskTransferConfig) -> dict[str, Any]:
    """Load the source active prompt version read-only (design D4).

    Raises CrossTaskTransferError naming the missing artifact when no active
    version exists.
    """
    from dynamic_prompt_core.infrastructure.storage import (
        PromptStore,
        PromptStoreConfig,
        PromptStoreError,
    )

    store_config = PromptStoreConfig(store_path=config.source_store_path)
    store = PromptStore(store_config)
    try:
        record = store.get_active()
    except PromptStoreError as exc:
        raise CrossTaskTransferError(f"source prompt store has no active version: {exc}") from exc
    return record


def _load_source_thesis_bank(config: CrossTaskTransferConfig) -> Any:
    """Load the source thesis bank dump and validate embedding model (design D4).

    Raises CrossTaskTransferError when the file is missing or the embedding
    model does not match the target's configured model.
    """
    from dynamic_prompt_core.application.use_cases.analyze_theses import thesis_analyzer

    path = config.source_thesis_bank_path
    if not os.path.isfile(path):
        raise CrossTaskTransferError(f"source thesis bank not found: {path}")

    target_ta_config = thesis_analyzer.ThesisAnalyzerConfig.from_config(config.config_path)

    with open(path, encoding="utf-8") as f:
        dump = json.load(f)
    source_model = dump.get("metadata", {}).get("embedding_model", "")
    target_model = target_ta_config.embedding_model
    if source_model and source_model != target_model:
        raise CrossTaskTransferError(
            f"embedding model mismatch: source thesis bank uses "
            f"{source_model!r} but target is configured for "
            f"{target_model!r}"
        )

    bank = thesis_analyzer.load_dump(path, target_ta_config)
    return bank


def _load_source_artifacts(
    config: CrossTaskTransferConfig,
    log_fn: Any | None = None,
) -> SourceArtifacts:
    """Load all source artifacts read-only (design D4).

    Returns a SourceArtifacts namedtuple with prompt_record, thesis_bank, and
    output_contract.
    """
    prompt_record = _load_source_prompt(config)
    thesis_bank = _load_source_thesis_bank(config)

    layers = _extract_source_layers(prompt_record)
    output_contract = layers.output_contract

    if log_fn:
        log_fn(
            "load_source",
            config.source_task.task_id,
            {
                "source_version": prompt_record.get("version"),
                "theses_count": len(thesis_bank.theses),
                "clusters_count": len(thesis_bank.clusters),
            },
        )

    return SourceArtifacts(
        prompt_record=prompt_record,
        thesis_bank=thesis_bank,
        output_contract=output_contract,
    )


# --------------------------------------------------------------------------- #
#  Transfer base layers
# --------------------------------------------------------------------------- #
def _extract_source_layers(prompt_record: dict[str, Any]) -> PromptLayer:
    """Reconstruct a PromptLayer from the source prompt text (design D5).

    Parses the ## Role, ## Task, ## Rules, ## Output contract, and ## Fallback
    sections produced by prompts.render.
    """
    text = prompt_record.get("text", "")
    sections: dict[str, str] = {}
    headers = [
        "## Role",
        "## Task",
        "## Rules",
        "## Output contract",
        "## Fallback",
    ]
    remaining = text
    for i, header in enumerate(headers):
        marker = header + "\n"
        idx = remaining.find(marker)
        if idx == -1:
            raise CrossTaskTransferError(f"source prompt text missing section header: {header}")
        start = idx + len(marker)
        rest = remaining[start:]
        next_idx = len(rest)
        for j in range(i + 1, len(headers)):
            nxt = rest.find(headers[j] + "\n")
            if nxt != -1:
                next_idx = min(next_idx, nxt)
        section_text = rest[:next_idx].rstrip("\n")
        sections[header] = section_text
        remaining = rest

    rules_text = sections["## Rules"].strip()
    rules: list[str] = []
    if rules_text:
        for line in rules_text.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                rules.append(line[2:])
            elif line:
                rules.append(line)

    return PromptLayer(
        role=sections["## Role"],
        task=sections["## Task"],
        rules=rules,
        output_contract=sections["## Output contract"],
        fallback=sections["## Fallback"],
    )


async def _reformulate_task_layer(
    source_task_layer: str,
    target_spec: TaskSpec,
    async_task: Any,
) -> str:
    """Reformulate the task layer for the target domain via a single LLM call (D5).

    Uses temperature 0 for determinism. Raises CrossTaskTransferError on failure.
    """
    import aiohttp

    user_message = (
        f"Source task layer:\n{source_task_layer}\n\n"
        f"Target domain:\n"
        f"  class_labels: {target_spec.class_labels}\n"
        f"  dataset_path: {target_spec.dataset_path}\n"
        f"  num_classes: {target_spec.num_classes}\n\n"
        "Reformulate the task layer for the target domain. Keep it concise "
        "(1-3 sentences). Describe what to classify and how to interpret "
        f"the {target_spec.num_classes} classes."
    )

    connector = aiohttp.TCPConnector(limit=1)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            response = await async_task.analyze_raw(
                session,
                user_message,
                model=None,
                system_prompt=_REFORMULATION_SYSTEM_PROMPT,
                max_tokens=128,
            )
    except Exception as exc:
        raise CrossTaskTransferError(f"task layer reformulation failed: {exc}") from exc

    content = response.content
    if not content:
        raise CrossTaskTransferError("task layer reformulation failed: LLM returned empty content")

    task_layer = content.strip()
    if len(task_layer) > MAX_TASK_LAYER_CHARS:
        task_layer = task_layer[:MAX_TASK_LAYER_CHARS]
    return cast(str, task_layer)


def _build_transferred_prompt(source_layers: PromptLayer, target_task_layer: str) -> PromptArtifact:
    """Assemble the transferred prompt with empty rules (design D6).

    Bypasses build_classification_prompt's 3-5 rule guard by constructing the
    PromptArtifact directly via PromptLayer + render.
    """
    transferred_layers = PromptLayer(
        role=source_layers.role,
        task=target_task_layer,
        rules=[],
        output_contract=source_layers.output_contract,
        fallback=source_layers.fallback,
    )
    text = render(transferred_layers)
    return PromptArtifact(
        version="classify-transfer-seed",
        layers=transferred_layers,
        text=text,
    )


# --------------------------------------------------------------------------- #
#  Seed target thesis collection and clusters
# --------------------------------------------------------------------------- #
def _seed_target_theses(
    source_bank: Any,
    target_bank: Any,
    max_transfer_theses: int | None,
) -> int:
    """Transfer theses from source to target with source=transfer tag (design D7).

    Transferred theses have frequency=0 and zeroed hit counters so they do not
    participate in precision computation. Returns the count transferred.
    """
    from dynamic_prompt_core.application.use_cases.analyze_theses import thesis_analyzer

    entries = list(source_bank.theses.values())
    if max_transfer_theses is not None:
        entries.sort(key=lambda e: e.frequency, reverse=True)
        entries = entries[:max_transfer_theses]

    count = 0
    for entry in entries:
        new_entry = thesis_analyzer.ThesisEntry(
            text_raw=entry.text_raw,
            text_norm=entry.text_norm,
            embedding=entry.embedding.copy() if entry.embedding is not None else None,
            frequency=0,
            positive_hits=0,
            negative_hits=0,
            demand=0,
            selected=0,
            in_prompt=False,
            cluster_id=entry.cluster_id,
            source="transfer",
        )
        target_bank.theses[entry.text_norm] = new_entry
        count += 1

    return count


def _seed_target_clusters(source_bank: Any, target_bank: Any) -> int:
    """Transfer cluster centroids as initial clustering points (design D8).

    Clusters are created with count=0 and zeroed counters. Transferred theses
    are linked by cluster_id without calling assign_cluster. Returns the count.
    """
    from dynamic_prompt_core.application.use_cases.analyze_theses import thesis_analyzer

    count = 0
    for cid, cluster in source_bank.clusters.items():
        new_cluster = thesis_analyzer.Cluster(
            cluster_id=cid,
            member_norms=[],
            centroid=cluster.centroid.copy() if cluster.centroid is not None else None,
            count=0,
            frequency=0,
            positive_hits=0,
            negative_hits=0,
            demand=0,
            selected=0,
            precision=0.0,
        )
        target_bank.clusters[cid] = new_cluster
        count += 1

    if source_bank.clusters:
        target_bank._next_cluster_id = max(source_bank.clusters.keys()) + 1

    return count


# --------------------------------------------------------------------------- #
#  Run target cycle
# --------------------------------------------------------------------------- #
def _write_transferred_seed(
    config: CrossTaskTransferConfig,
    transferred_prompt: PromptArtifact,
    target_bank: Any,
    run_id: str,
) -> tuple[str, str]:
    """Write the transferred prompt and seeded thesis bank to target paths (D9).

    Returns (target_store_path, target_thesis_bank_path).
    """
    from dynamic_prompt_core.application.use_cases.analyze_theses import thesis_analyzer
    from dynamic_prompt_core.infrastructure.storage import prompt_store

    os.makedirs(os.path.dirname(config.target_store_path) or ".", exist_ok=True)
    store_config = prompt_store.PromptStoreConfig(
        store_path=config.target_store_path,
        log_path=os.path.join(
            os.path.dirname(config.target_store_path) or ".",
            "target_prompt_store.jsonl",
        ),
    )
    store = prompt_store.init_store(store_config)

    prompt_version_dict: dict[str, Any] = {
        "text": transferred_prompt.text,
        "hash": prompt_store._sha256_hex(transferred_prompt.text),
        "rules": [],
        "source_candidates": [],
        "base_version": None,
        "created_at": datetime.now(UTC).isoformat(),
        "version": "classify-transfer-seed",
    }
    record = store.save(prompt_version_dict, reason="transfer-seed")
    store.activate(record["version"])

    ta_config = thesis_analyzer.ThesisAnalyzerConfig.from_config(config.config_path)
    thesis_bank_path = thesis_analyzer.write_dump(
        target_bank,
        run_id=run_id,
        prompt_version="classify-transfer-seed",
        config=ta_config,
        output_dir=config.target_artifacts_path,
    )

    return config.target_store_path, thesis_bank_path


def _build_target_cycle_config(
    config: CrossTaskTransferConfig,
    target_store_path: str,
    target_thesis_bank_path: str,
) -> Any:
    """Build a CycleOrchestratorConfig for the target cycle (design D9)."""
    import dynamic_prompt_core.application.use_cases.run_cycle.orchestrator as cycle_orchestrator  # noqa: E402

    return cycle_orchestrator.CycleOrchestratorConfig(
        max_rounds=config.max_rounds_target,
        prompt_store_path=target_store_path,
        thesis_bank_path=target_thesis_bank_path,
        output_dir=config.target_artifacts_path,
        config_path=config.config_path,
    )


def _transfer_run_id(config: CrossTaskTransferConfig, timestamp: str) -> str:
    """Build a distinct run_id for the transfer target cycle (design D9)."""
    return f"transfer-{config.source_task.task_id}-to-{config.target_task.task_id}-{timestamp}"


async def _run_target_cycle(
    config: CrossTaskTransferConfig,
    transferred_prompt: PromptArtifact,
    target_bank: Any,
    async_task: Any,
    endpoint: str,
    run_id: str,
    log_fn: Any | None = None,
) -> tuple[Any, dict[str, Any], str]:
    """Run the target optimization cycle (design D9).

    Returns (final_state, final_metrics, stop_reason).
    """
    import dynamic_prompt_core.application.use_cases.run_cycle.orchestrator as cycle_orchestrator  # noqa: E402

    target_store_path, target_thesis_bank_path = _write_transferred_seed(
        config, transferred_prompt, target_bank, run_id
    )
    target_config = _build_target_cycle_config(config, target_store_path, target_thesis_bank_path)

    if log_fn:
        log_fn(
            "run_target_cycle",
            config.target_task.task_id,
            {"run_id": run_id, "max_rounds": config.max_rounds_target},
        )

    state = await cycle_orchestrator.run_cycle(target_config, endpoint=endpoint, run_id=run_id)

    final_metrics, stop_reason = _load_cycle_summary_metrics(config.target_artifacts_path, run_id)

    return state, final_metrics, stop_reason


def _load_cycle_summary_metrics(output_dir: str, run_id: str) -> tuple[dict[str, Any], str]:
    """Load the final dev metrics and stop reason from a cycle's summary file."""
    pattern = os.path.join(output_dir, f"summary_{run_id}_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return {}, ""
    with open(matches[-1], encoding="utf-8") as f:
        summary = json.load(f)
    return (
        summary.get("final_dev_metrics") or {},
        summary.get("stop_reason", ""),
    )


def _load_first_round_metrics(output_dir: str, run_id: str) -> dict[str, Any]:
    """Load the new version's dev metrics from the first round's report."""
    pattern = os.path.join(output_dir, f"report_{run_id}_round1_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return {}
    with open(matches[-1], encoding="utf-8") as f:
        report = json.load(f)
    return report.get("metrics_dev_new") or {}


# --------------------------------------------------------------------------- #
#  Transfer artifact
# --------------------------------------------------------------------------- #
def _compute_improvement(
    metrics_start: dict[str, Any],
    metrics_final: dict[str, Any],
    metric: str,
) -> dict[str, float]:
    """Compute absolute and relative improvement on *metric* (design D10)."""
    start_val = float(metrics_start.get(metric, 0.0) or 0.0)
    final_val = float(metrics_final.get(metric, 0.0) or 0.0)
    absolute = final_val - start_val
    relative = absolute / start_val if start_val > 0 else 0.0
    return {"absolute": absolute, "relative": relative}


def write_transfer_artifact(
    config: CrossTaskTransferConfig,
    result: TransferResult,
    timestamp: str,
) -> str:
    """Write the transfer artifact JSON (design D10).

    Filename: transfer_{source_task_id}_to_{target_task_id}_{timestamp}.json
    """
    os.makedirs(config.output_dir, exist_ok=True)
    name = f"transfer_{config.source_task.task_id}_to_{config.target_task.task_id}_{timestamp}.json"
    path = os.path.join(config.output_dir, name)

    dump = {
        "source_task_id": config.source_task.task_id,
        "target_task_id": config.target_task.task_id,
        "source_version": result.source_version,
        "target_final_version": result.target_final_version,
        "transferred_theses_count": result.transferred_theses_count,
        "transferred_clusters_count": result.transferred_clusters_count,
        "target_metrics_start": result.target_metrics_start,
        "target_metrics_final": result.target_metrics_final,
        "improvement": result.improvement,
        "stop_reason": result.stop_reason,
        "timestamp": timestamp,
        "created_at": datetime.now(UTC).isoformat(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
#  Effectiveness report
# --------------------------------------------------------------------------- #
async def _compute_target_baseline(
    config: CrossTaskTransferConfig,
    async_task: Any,
    endpoint: str,
    run_id: str,
) -> dict[str, Any]:
    """Run the target dataset with v0 baseline prompt and compute metrics (D11)."""
    import dynamic_prompt_core.application.use_cases.run_cycle.orchestrator as cycle_orchestrator  # noqa: E402

    baseline_config = cycle_orchestrator.CycleOrchestratorConfig(
        max_rounds=1,
        prompt_store_path="",
        thesis_bank_path="",
        output_dir=config.target_artifacts_path,
        config_path=config.config_path,
    )
    state = cycle_orchestrator.load_initial_state(baseline_config, run_id=run_id)
    results_path = await cycle_orchestrator._run_on_split(
        state.active_prompt,
        baseline_config.dev_split,
        baseline_config,
        async_task,
        f"{run_id}-baseline-dev",
    )
    return cast(
        dict[str, Any],
        cycle_orchestrator.compute_dev_metrics(results_path, baseline_config),
    )


def write_effectiveness_report(
    config: CrossTaskTransferConfig,
    baseline_metrics: dict[str, Any],
    transfer_result: TransferResult,
    target_state: Any,
    timestamp: str,
    cold_start_section: dict[str, Any] | None = None,
) -> str:
    """Write the effectiveness report JSON (design D11).

    Filename: transfer_report_{source_task_id}_to_{target_task_id}_{timestamp}.json
    """
    os.makedirs(config.output_dir, exist_ok=True)
    name = (
        f"transfer_report_{config.source_task.task_id}_to_"
        f"{config.target_task.task_id}_{timestamp}.json"
    )
    path = os.path.join(config.output_dir, name)

    report: dict[str, Any] = {
        "source_task_id": config.source_task.task_id,
        "target_task_id": config.target_task.task_id,
        "metric": config.target_task.metric,
        "baseline": baseline_metrics,
        "after_first_round": transfer_result.target_metrics_start,
        "after_final_round": transfer_result.target_metrics_final,
        "rounds_to_stop": target_state.round_counter,
        "improvement": transfer_result.improvement,
        "timestamp": timestamp,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if cold_start_section is not None:
        report["cold_start"] = cold_start_section

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


# --------------------------------------------------------------------------- #
#  Cold-start comparison
# --------------------------------------------------------------------------- #
async def _run_cold_start_cycle(
    config: CrossTaskTransferConfig,
    async_task: Any,
    endpoint: str,
    timestamp: str,
    log_fn: Any | None = None,
) -> tuple[Any, dict[str, Any], str]:
    """Run the target cycle from a cold start (design D11).

    Returns (cold_start_state, final_metrics, stop_reason).
    """
    import dynamic_prompt_core.application.use_cases.run_cycle.orchestrator as cycle_orchestrator  # noqa: E402

    run_id = f"coldstart-{config.target_task.task_id}-{timestamp}"

    cold_config = cycle_orchestrator.CycleOrchestratorConfig(
        max_rounds=config.max_rounds_target,
        prompt_store_path="",
        thesis_bank_path="",
        output_dir=config.target_artifacts_path,
        config_path=config.config_path,
    )

    if log_fn:
        log_fn(
            "cold_start_run",
            config.target_task.task_id,
            {"run_id": run_id, "max_rounds": config.max_rounds_target},
        )

    state = await cycle_orchestrator.run_cycle(cold_config, endpoint=endpoint, run_id=run_id)

    final_metrics, stop_reason = _load_cycle_summary_metrics(config.target_artifacts_path, run_id)

    return state, final_metrics, stop_reason


def _write_comparison_section(
    cold_start_state: Any,
    cold_start_metrics: dict[str, Any],
    transfer_state: Any,
    transfer_result: TransferResult,
) -> dict[str, Any]:
    """Build the cold-start comparison section for the report (design D11)."""
    acceleration = cold_start_state.round_counter - transfer_state.round_counter
    return {
        "metrics_final": cold_start_metrics,
        "rounds_to_stop": cold_start_state.round_counter,
        "stop_reason": getattr(cold_start_state, "_stop_reason", ""),
        "acceleration_rounds": acceleration,
    }


# --------------------------------------------------------------------------- #
#  Isolation
# --------------------------------------------------------------------------- #
def _check_target_path_clean(config: CrossTaskTransferConfig) -> None:
    """Verify the target path does not contain prior run artifacts (design D12).

    Raises CrossTaskTransferError naming the conflict when prior artifacts exist.
    """
    if os.path.exists(config.target_store_path):
        raise CrossTaskTransferError(
            f"target store path already exists: {config.target_store_path} "
            "(archive or delete the prior target run before transferring)"
        )


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
def _log_operation(
    config: CrossTaskTransferConfig,
    operation: str,
    task_id: str,
    details: dict[str, Any],
) -> None:
    """Append one JSON line to the transfer log (design D14).

    Best-effort: a write failure is printed to stderr and never blocks.
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "task_id": task_id,
        "details": details,
    }
    try:
        with open(config.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(
            f"cross_task_transfer: failed to write log: {exc}",
            file=sys.stderr,
        )


def _make_logger(config: CrossTaskTransferConfig) -> Callable[[str, str, dict[str, Any]], None]:
    """Return a closure that logs with the given config."""

    def _log(operation: str, task_id: str, details: dict[str, Any]) -> None:
        _log_operation(config, operation, task_id, details)

    return _log


# --------------------------------------------------------------------------- #
#  Main transfer orchestration
# --------------------------------------------------------------------------- #
async def run_transfer(
    config: CrossTaskTransferConfig,
    endpoint: str = DEFAULT_ENDPOINT,
) -> tuple[TransferResult, str]:
    """Run the full cross-task transfer.

    Returns (transfer_result, timestamp).
    """
    from dynamic_prompt_core.application.use_cases.analyze_theses import thesis_analyzer
    from dynamic_prompt_core.infrastructure.llm import AsyncTask

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_fn = _make_logger(config)

    _check_target_path_clean(config)

    _check_compatibility(
        config.source_task,
        config.target_task,
        "",
    )

    source_artifacts = _load_source_artifacts(config, log_fn=log_fn)

    _check_compatibility(
        config.source_task,
        config.target_task,
        source_artifacts.output_contract,
    )
    log_fn(
        "check_compatibility",
        f"{config.source_task.task_id}->{config.target_task.task_id}",
        {"num_classes": config.source_task.num_classes},
    )

    source_layers = _extract_source_layers(source_artifacts.prompt_record)

    async_task = AsyncTask(
        config_path=config.config_path,
        endpoint=endpoint,
    )

    target_task_layer = await _reformulate_task_layer(
        source_layers.task, config.target_task, async_task
    )
    transferred_prompt = _build_transferred_prompt(source_layers, target_task_layer)
    log_fn(
        "transfer_layers",
        config.target_task.task_id,
        {
            "rules_cleared": True,
            "task_layer_chars": len(target_task_layer),
        },
    )

    ta_config = thesis_analyzer.ThesisAnalyzerConfig.from_config(config.config_path)
    target_bank = thesis_analyzer.ThesisBank(ta_config)

    transferred_theses_count = _seed_target_theses(
        source_artifacts.thesis_bank,
        target_bank,
        config.max_transfer_theses,
    )
    log_fn(
        "transfer_theses",
        config.target_task.task_id,
        {"count": transferred_theses_count},
    )

    transferred_clusters_count = 0
    if config.transfer_clusters:
        transferred_clusters_count = _seed_target_clusters(
            source_artifacts.thesis_bank, target_bank
        )
    log_fn(
        "transfer_clusters",
        config.target_task.task_id,
        {"count": transferred_clusters_count},
    )

    run_id = _transfer_run_id(config, timestamp)
    target_state, final_metrics, stop_reason = await _run_target_cycle(
        config,
        transferred_prompt,
        target_bank,
        async_task,
        endpoint,
        run_id,
        log_fn=log_fn,
    )

    start_metrics = _load_first_round_metrics(config.target_artifacts_path, run_id)

    improvement = _compute_improvement(start_metrics, final_metrics, config.target_task.metric)

    result = TransferResult(
        source_version=source_artifacts.prompt_record.get("version", 0),
        target_final_version=target_state.active_version,
        transferred_theses_count=transferred_theses_count,
        transferred_clusters_count=transferred_clusters_count,
        target_metrics_start=start_metrics,
        target_metrics_final=final_metrics,
        improvement=improvement,
        stop_reason=stop_reason,
    )

    artifact_path = write_transfer_artifact(config, result, timestamp)
    log_fn(
        "write_artifact",
        config.target_task.task_id,
        {"path": artifact_path},
    )

    baseline_metrics = await _compute_target_baseline(config, async_task, endpoint, run_id)

    cold_start_section: dict[str, Any] | None = None
    if config.compare_with_cold_start:
        cs_state, cs_metrics, _ = await _run_cold_start_cycle(
            config, async_task, endpoint, timestamp, log_fn=log_fn
        )
        cold_start_section = _write_comparison_section(cs_state, cs_metrics, target_state, result)

    report_path = write_effectiveness_report(
        config,
        baseline_metrics,
        result,
        target_state,
        timestamp,
        cold_start_section=cold_start_section,
    )
    log_fn(
        "write_report",
        config.target_task.task_id,
        {"path": report_path},
    )

    return result, timestamp


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
async def _main_async(args: argparse.Namespace) -> None:
    config = CrossTaskTransferConfig.from_config(args.config)

    print("=" * 60)
    print("Cross-task transfer")
    print("=" * 60)
    print(
        f"  source: {config.source_task.task_id} "
        f"({config.source_task.num_classes} classes, "
        f"metric={config.source_task.metric})"
    )
    print(
        f"  target: {config.target_task.task_id} "
        f"({config.target_task.num_classes} classes, "
        f"metric={config.target_task.metric})"
    )
    print(
        f"  max_transfer_theses: {config.max_transfer_theses}\n"
        f"  transfer_clusters: {config.transfer_clusters}\n"
        f"  compare_with_cold_start: {config.compare_with_cold_start}\n"
        f"  max_rounds_target: {config.max_rounds_target}"
    )
    print("-" * 60)

    result, timestamp = await run_transfer(config, endpoint=args.endpoint)

    print("-" * 60)
    print(
        f"  transferred_theses : {result.transferred_theses_count}\n"
        f"  transferred_clusters: {result.transferred_clusters_count}\n"
        f"  target_final_version: {result.target_final_version}\n"
        f"  stop_reason         : {result.stop_reason}\n"
        f"  improvement         : {result.improvement}"
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 cross-task transfer — transfer an optimized "
        "prompt from a source task to a target task"
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.toml (default: config.toml)",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="LLM server endpoint (default: http://127.0.0.1:8080/v1)",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "run",
        help="Run the full cross-task transfer: validate, load source, "
        "transfer layers/theses/clusters, run target cycle, write artifacts",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    pass  # logging configured by composition root

    if args.command == "run":
        asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
