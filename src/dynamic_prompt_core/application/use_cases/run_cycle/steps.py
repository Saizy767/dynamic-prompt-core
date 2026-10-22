"""The nine-step round sequence executed by ``run_cycle``.

Each step calls an existing use case or service via its public API.  The
orchestrator does not duplicate their logic.  Sub-step configs are loaded from
``config_path`` via each module's own ``XxxConfig.from_config()`` classmethod.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from dynamic_prompt_core.application.services.metrics import (
    MetricsConfig,
    compute_metrics,
)
from dynamic_prompt_core.application.services.metrics import (
    load_results as load_metrics_results,
)
from dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer import (
    ThesisAnalyzerConfig,
    analyze,
    write_dump,
)
from dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer import (
    load_results as load_analyzer_results,
)
from dynamic_prompt_core.application.use_cases.compare_versions.comparator import (
    VersionComparatorConfig,
    decide,
)
from dynamic_prompt_core.application.use_cases.compose_prompt.composer import (
    PromptComposerConfig,
    assemble_prompt,
    compose,
    finalized_version,
    intermediate_version,
    load_base_layers,
    load_centroids,
    write_prompt_version,
)
from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
    BaselineRunner,
    RunnerConfig,
)
from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.report import (
    log_event,
    write_report,
)
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
    RunCycleDeps,
)
from dynamic_prompt_core.application.use_cases.run_cycle.state import (
    CycleState,
    update_in_prompt_flags,
)
from dynamic_prompt_core.application.use_cases.select_candidates.selector import (
    RuleCandidateSelectorConfig,
    load_thesis_artifact,
    write_candidates,
)
from dynamic_prompt_core.application.use_cases.select_candidates.selector import (
    select_candidates as select_candidates_fn,
)
from dynamic_prompt_core.domain.prompts.base import PromptArtifact

log = logging.getLogger(__name__)


def _extract_metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract ``accuracy``, ``macro_f1``, ``minority_f1`` from a metrics dict."""
    f1 = metrics.get("f1", {})
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(f1.get("macro", 0.0)),
        "minority_f1": float(f1.get("minority", 0.0)),
    }


# --------------------------------------------------------------------------- #
#  Step 1: Run the active prompt on dev
# --------------------------------------------------------------------------- #
async def run_active_on_dev(
    state: CycleState,
    deps: RunCycleDeps,
    config: CycleConfig,
    config_path: str,
) -> str:
    """Run the active prompt version on the dev split.

    Returns the results artifact path.
    """
    runner_config = RunnerConfig.from_config(config_path)
    cycle_number = state.cycle_counter + 1
    runner = BaselineRunner(
        task=cast(Any, deps.llm_client),
        config=runner_config,
        split=config.dev_split,
        scorer=deps.candidate_scorer,
        policy=deps.classification_policy,
        candidates=list(deps.candidates),
        run_id=state.run_id,
        classify_prompt=state.active_version,
        duration=config.duration_for_cycle(cycle_number),
    )
    _, artifact_path = await runner.run()
    return artifact_path


# --------------------------------------------------------------------------- #
#  Step 2: Compute dev metrics
# --------------------------------------------------------------------------- #
def compute_dev_metrics(results_path: str, config_path: str) -> dict[str, Any]:
    """Compute metrics from a dev results artifact."""
    m_config = MetricsConfig.from_config(config_path)
    rows = load_metrics_results(results_path)
    return compute_metrics(rows, m_config, results_path)


# --------------------------------------------------------------------------- #
#  Step 3: Update thesis collection and clusters
# --------------------------------------------------------------------------- #
def update_thesis_collection(
    results_path: str,
    config_path: str,
    refined_path: str | None = None,
) -> tuple[str, str]:
    """Run the thesis analyzer on the active results.

    Returns ``(thesis_bank_path, clusters_path)``.  When ``refined_path`` is
    provided, the refined theses artifact is used instead of the raw results.
    """
    ta_config = ThesisAnalyzerConfig.from_config(config_path)
    source = refined_path if refined_path is not None else results_path
    rows = load_analyzer_results(source)
    bank, name_parts = analyze(rows, ta_config, source)
    run_id = name_parts.get("run_id", "unknown")
    prompt_version = name_parts.get("prompt_version", "unknown")
    thesis_bank_path = write_dump(bank, run_id, prompt_version, ta_config, ta_config.output_dir)
    clusters_path = thesis_bank_path
    return thesis_bank_path, clusters_path


# --------------------------------------------------------------------------- #
#  Step 4: Select rule candidates
# --------------------------------------------------------------------------- #
def select_candidates(
    thesis_bank_path: str,
    config_path: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the rule-candidate selector on the thesis bank.

    Returns ``(candidate_artifact_path, candidate_queue)``.
    """
    sel_config = RuleCandidateSelectorConfig.from_config(config_path)
    artifact = load_thesis_artifact(thesis_bank_path)
    candidates, already_in_prompt, counters = select_candidates_fn(artifact, sel_config)
    run_id = artifact["metadata"].get("run_id", "unknown")
    prompt_version = artifact["metadata"].get("prompt_version", "unknown")
    candidate_path = write_candidates(
        candidates,
        already_in_prompt,
        counters,
        run_id,
        prompt_version,
        sel_config,
        sel_config.output_dir,
    )
    return candidate_path, candidates


# --------------------------------------------------------------------------- #
#  Step 5: Compose a new prompt version
# --------------------------------------------------------------------------- #
async def compose_new_version(
    candidate: dict[str, Any],
    active_prompt: PromptArtifact,
    config_path: str,
    deps: RunCycleDeps,
    run_id: str,
    version_str: str | None = None,
    thesis_bank_path: str | None = None,
) -> tuple[PromptArtifact, str, list[int]]:
    """Compose a new prompt version from a candidate and the active base layers.

    Returns ``(new_prompt, artifact_path, source_candidates)``.

    ``thesis_bank_path`` is the path to the thesis-bank dump artifact produced
    by ``update_thesis_collection``.  It supplies cluster centroids for the
    semantic distortion check.  When ``None``, the distortion check is skipped
    for every rule and a warning is logged.
    """
    pc_config = PromptComposerConfig.from_config(config_path)
    base_layers = load_base_layers(active_prompt)
    if thesis_bank_path is None:
        log.warning(
            "compose_new_version: thesis_bank_path is None — "
            "distortion check will be skipped for all rules"
        )
    centroids = load_centroids(thesis_bank_path)
    candidates_list = [candidate]
    existing_rules = list(active_prompt.layers.rules) if active_prompt.layers else []
    new_artifact, rules_with_lineage, counters, rejected = await compose(
        candidates_list,
        base_layers,
        centroids,
        pc_config,
        cast(Any, deps.llm_client),
        active_prompt.version,
        existing_rules=existing_rules,
        version_str=version_str,
    )
    source_candidates = [c["cluster_id"] for c in rules_with_lineage]
    artifact_path = write_prompt_version(
        new_artifact,
        rules_with_lineage,
        source_candidates,
        active_prompt.version,
        run_id,
        counters,
        pc_config,
        pc_config.output_dir,
        rejected_rules=rejected,
    )
    return new_artifact, artifact_path, source_candidates


# --------------------------------------------------------------------------- #
#  Step 6: Run the new version on dev
# --------------------------------------------------------------------------- #
async def run_new_on_dev(
    new_prompt: PromptArtifact,
    config: CycleConfig,
    config_path: str,
    deps: RunCycleDeps,
    run_id: str,
    round_number: int = 1,
) -> str:
    """Run the new prompt version on the dev split.

    Returns the results artifact path.
    """
    runner_config = RunnerConfig.from_config(config_path)
    runner = BaselineRunner(
        task=cast(Any, deps.llm_client),
        config=runner_config,
        split=config.dev_split,
        scorer=deps.candidate_scorer,
        policy=deps.classification_policy,
        candidates=list(deps.candidates),
        run_id=run_id,
        classify_prompt=new_prompt,
        duration=config.duration_for_cycle(round_number),
    )
    _, artifact_path = await runner.run()
    return artifact_path


# --------------------------------------------------------------------------- #
#  Holdout evaluation (logged only)
# --------------------------------------------------------------------------- #
async def run_holdout(
    prompt_artifact: PromptArtifact,
    config: CycleConfig,
    config_path: str,
    deps: RunCycleDeps,
    run_id: str,
    round_number: int = 1,
) -> dict[str, Any]:
    """Run a prompt on the holdout split and compute metrics.

    Returns the holdout metrics dict.  Logged only — not used in the decision.
    """
    runner_config = RunnerConfig.from_config(config_path)
    runner = BaselineRunner(
        task=cast(Any, deps.llm_client),
        config=runner_config,
        split=config.holdout_split,
        scorer=deps.candidate_scorer,
        policy=deps.classification_policy,
        candidates=list(deps.candidates),
        run_id=run_id,
        classify_prompt=prompt_artifact,
        duration=config.duration_for_cycle(round_number),
        skip_extraction=True,
    )
    _, artifact_path = await runner.run()
    return compute_dev_metrics(artifact_path, config_path)


# --------------------------------------------------------------------------- #
#  Optional teacher refinement (step between 1 and 3)
# --------------------------------------------------------------------------- #
async def refine_theses_step(
    results_path: str,
    run_id: str,
    prompt_version: str,
    deps: RunCycleDeps,
    config_path: str,
) -> str:
    """Optionally refine theses via the teacher model.

    Returns the refined artifact path when refinement is enabled, or the
    original ``results_path`` when ``deps.teacher_llm_client is None``.
    """
    if deps.teacher_llm_client is None:
        return results_path

    from dynamic_prompt_core.application.use_cases.refine_theses.refine_theses_deps import (
        RefineThesesDeps,
    )
    from dynamic_prompt_core.application.use_cases.refine_theses.refiner import (
        refine_theses,
    )
    from dynamic_prompt_core.application.use_cases.refine_theses.result import (
        RefineThesesInput,
    )

    refine_deps = RefineThesesDeps(
        teacher_llm_client=deps.teacher_llm_client,
        run_repository=deps.run_repository,
    )
    inp = RefineThesesInput(
        results_path=results_path,
        run_id=run_id,
        prompt_version=prompt_version,
        output_dir=os.path.join(os.path.dirname(results_path) or "data/results", ""),
    )
    result = await refine_theses(refine_deps, inp)
    return result.artifact_path


# --------------------------------------------------------------------------- #
#  Step 9: Run a full cycle (steps 1–8)
# --------------------------------------------------------------------------- #
async def run_cycle_step(
    state: CycleState,
    deps: RunCycleDeps,
    config: CycleConfig,
    config_path: str,
    log_path: str,
) -> CycleState:
    """Execute one full cycle and return the updated state.

    Greedily accumulates rules from the candidate queue: each candidate's
    rule is composed on top of the previously accepted rules, run on
    holdout, and compared against the current best. Accumulation continues
    on accept (strict improvement) and stops on degradation or when
    max_rules is reached. The best version is finalized as ``vN``.
    """
    cycle_number = state.cycle_counter + 1
    state.cycle_number = cycle_number
    state.cluster_index = 0
    state.any_accepted = False
    active_version_at_start = state.active_version.version
    log_event("cycle_start", cycle_number, {"active_version": active_version_at_start}, log_path)

    # Step 1: Run active on dev (thesis collection + dev metrics for logging)
    active_dev_path = await run_active_on_dev(state, deps, config, config_path)

    # Step 2: Compute dev metrics for active (logging only)
    metrics_dev_active = compute_dev_metrics(active_dev_path, config_path)
    _dev_summary = _extract_metric_summary(metrics_dev_active)
    log.info(
        "cycle %d active dev metrics: version=%s accuracy=%.4f macro_f1=%.4f minority_f1=%.4f",
        cycle_number,
        active_version_at_start,
        _dev_summary["accuracy"],
        _dev_summary["macro_f1"],
        _dev_summary["minority_f1"],
    )

    # Optional teacher refinement (between step 1 and step 3)
    refined_path: str | None = None
    if config.use_teacher_refinement:
        refined_path = await refine_theses_step(
            active_dev_path,
            state.run_id,
            active_version_at_start,
            deps,
            config_path,
        )

    # Step 3: Update thesis collection
    thesis_bank_path, clusters_path = update_thesis_collection(
        active_dev_path,
        config_path,
        refined_path=refined_path,
    )
    state.thesis_bank_path = thesis_bank_path
    state.clusters_path = clusters_path

    # Step 4: Select candidates
    _, candidate_queue = select_candidates(thesis_bank_path, config_path)
    state.candidate_queue = list(candidate_queue)
    state.last_cycle_new_candidates = len(candidate_queue)

    # Step 5: Run active on holdout → cached baseline for the accumulation loop
    metrics_holdout_active: dict[str, Any] = {}
    try:
        metrics_holdout_active = await run_holdout(
            state.active_version,
            config,
            config_path,
            deps,
            state.run_id,
            round_number=cycle_number,
        )
    except Exception as exc:
        log.warning("active holdout evaluation failed: %s", exc)

    if metrics_holdout_active:
        _hold_summary = _extract_metric_summary(metrics_holdout_active)
        log.info(
            "cycle %d active holdout metrics: version=%s accuracy=%.4f "
            "macro_f1=%.4f minority_f1=%.4f",
            cycle_number,
            active_version_at_start,
            _hold_summary["accuracy"],
            _hold_summary["macro_f1"],
            _hold_summary["minority_f1"],
        )

    # Steps 6–7: Greedy incremental rule accumulation (holdout only)
    best_prompt = state.active_version
    best_artifact_path = state.active_version_path
    best_holdout_metrics = metrics_holdout_active
    accumulated_source_candidates: list[int] = []
    decision = "rollback"
    reason = ""
    new_version = ""
    attempts: list[dict[str, Any]] = []

    pc_config = PromptComposerConfig.from_config(config_path)

    while state.candidate_queue:
        current_rule_count = len(best_prompt.layers.rules) if best_prompt.layers else 0
        if current_rule_count >= pc_config.max_rules:
            log.info(
                "max_rules limit (%d) reached; stopping accumulation",
                pc_config.max_rules,
            )
            break

        candidate = state.candidate_queue.pop(0)
        state.cluster_index += 1

        # Step 6: Compose vN_k (carries over existing rules from best_prompt)
        version_str = intermediate_version(cycle_number, state.cluster_index)
        new_prompt, new_artifact_path, source_candidates = await compose_new_version(
            candidate,
            best_prompt,
            config_path,
            deps,
            state.run_id,
            version_str=version_str,
            thesis_bank_path=state.thesis_bank_path,
        )
        new_version = new_prompt.version

        old_rules = list(best_prompt.layers.rules) if best_prompt.layers else []
        new_rules = list(new_prompt.layers.rules) if new_prompt.layers else []

        if len(new_rules) <= len(old_rules):
            log.warning(
                "candidate cluster_id=%s did not add a rule; "
                "version=%s has %d rules, previous=%d; skipping evaluation",
                candidate.get("cluster_id"),
                new_prompt.version,
                len(new_rules),
                len(old_rules),
            )
            continue

        # Run vN_k on holdout only (no dev run per candidate)
        holdout_metrics_new: dict[str, Any] = {}
        try:
            holdout_metrics_new = await run_holdout(
                new_prompt,
                config,
                config_path,
                deps,
                state.run_id,
                round_number=cycle_number,
            )
        except Exception as exc:
            log.warning("holdout evaluation failed for %s: %s", new_version, exc)
            break

        # Step 7: Decide by holdout (strict improvement)
        vc_config = VersionComparatorConfig.from_config(config_path)
        decision, reason = decide(holdout_metrics_new, best_holdout_metrics, vc_config)

        new_summary = _extract_metric_summary(holdout_metrics_new)
        best_summary = _extract_metric_summary(best_holdout_metrics)
        log.info(
            "cycle %d holdout decision: version=%s cluster_id=%s decision=%s "
            "new(accuracy=%.4f macro_f1=%.4f minority_f1=%.4f) "
            "best(accuracy=%.4f macro_f1=%.4f minority_f1=%.4f)",
            cycle_number,
            new_version,
            candidate.get("cluster_id"),
            decision,
            new_summary["accuracy"],
            new_summary["macro_f1"],
            new_summary["minority_f1"],
            best_summary["accuracy"],
            best_summary["macro_f1"],
            best_summary["minority_f1"],
        )

        attempts.append(
            {
                "version": new_version,
                "cluster_id": candidate.get("cluster_id"),
                "holdout_metric": _extract_metric_summary(holdout_metrics_new),
                "decision": decision,
            }
        )

        log_event(
            "decision",
            cycle_number,
            {
                "decision": decision,
                "reason": reason,
                "new_version": new_version,
                "cluster_id": candidate.get("cluster_id"),
                "accumulated_rules": len(accumulated_source_candidates),
            },
            log_path,
        )

        if decision == "accept":
            state.any_accepted = True
            best_prompt = new_prompt
            best_artifact_path = new_artifact_path
            best_holdout_metrics = holdout_metrics_new
            accumulated_source_candidates.extend(source_candidates)
            log_event(
                "accept",
                cycle_number,
                {
                    "version": new_version,
                    "cluster_id": candidate.get("cluster_id"),
                    "total_rules": len(accumulated_source_candidates),
                },
                log_path,
            )
            continue

        log_event(
            "rollback",
            cycle_number,
            {"version": new_version, "cluster_id": candidate.get("cluster_id")},
            log_path,
        )
        break

    # Finalize: rename best version to vN
    if state.any_accepted:
        finalized_str = finalized_version(cycle_number)
        base_layers = load_base_layers(best_prompt)
        all_rules = list(best_prompt.layers.rules) if best_prompt.layers else []
        best_prompt = assemble_prompt(
            base_layers, all_rules, active_version_at_start, version_str=finalized_str
        )

    # Update state with the best version
    state.active_version = best_prompt
    state.active_version_path = best_artifact_path

    # Persist the accepted version so load_initial_state recovers it next run.
    if state.any_accepted:
        try:
            saved = deps.prompt_repository.save(
                {
                    "text": best_prompt.text,
                    "rules": list(best_prompt.layers.rules)
                    if best_prompt.layers
                    else [],
                    "source_candidates": accumulated_source_candidates,
                },
                reason="accepted",
            )
            deps.prompt_repository.activate(saved["version"])
        except Exception as exc:
            log.warning("failed to persist accepted prompt to store: %s", exc)

    # Update histories
    if state.any_accepted:
        state.accepted_history.append(active_version_at_start)
    else:
        state.rollback_history.append(
            {
                "cycle": cycle_number,
                "version": new_version or active_version_at_start,
                "reason": reason or "no candidates available",
            }
        )

    # Update in_prompt flags for all accepted candidates
    if accumulated_source_candidates and state.thesis_bank_path:
        accepted_cluster_ids = {str(cid) for cid in accumulated_source_candidates}
        update_in_prompt_flags(state.thesis_bank_path, accepted_cluster_ids)

    # Step 8: Write per-cycle report
    final_decision = "accept" if state.any_accepted else "rollback"
    next_action = "continue" if state.any_accepted else "rollback"
    report_path = write_report(
        round_number=cycle_number,
        active_version_at_start=active_version_at_start,
        new_version=best_prompt.version,
        decision=final_decision,
        metrics_dev_active=metrics_dev_active,
        metrics_dev_new={},
        metrics_holdout_active=metrics_holdout_active,
        metrics_holdout_new=best_holdout_metrics,
        rollback_counter=len(state.rollback_history),
        changed_decisions=[],
        next_action=next_action,
        run_id=state.run_id,
        run_repository=deps.run_repository,
        output_dir=config.output_dir,
    )
    state.latest_report_path = report_path

    log_event(
        "cycle_end",
        cycle_number,
        {"decision": final_decision, "report": report_path},
        log_path,
    )
    return state
