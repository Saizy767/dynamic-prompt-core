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
    update_rollback_count,
)
from dynamic_prompt_core.application.use_cases.compose_prompt.composer import (
    PromptComposerConfig,
    compose,
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
    round_number = state.round_counter + 1
    runner = BaselineRunner(
        task=cast(Any, deps.llm_client),
        config=runner_config,
        split=config.dev_split,
        run_id=state.run_id,
        classify_prompt=state.active_version,
        duration=config.duration_for_round(round_number),
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
) -> tuple[PromptArtifact, str, list[int]]:
    """Compose a new prompt version from a candidate and the active base layers.

    Returns ``(new_prompt, artifact_path, source_candidates)``.
    """
    pc_config = PromptComposerConfig.from_config(config_path)
    base_layers = load_base_layers(active_prompt)
    centroids = load_centroids(None)
    candidates_list = [candidate]
    new_artifact, rules_with_lineage, counters, rejected = await compose(
        candidates_list,
        base_layers,
        centroids,
        pc_config,
        cast(Any, deps.llm_client),
        active_prompt.version,
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
        run_id=run_id,
        classify_prompt=new_prompt,
        duration=config.duration_for_round(round_number),
    )
    _, artifact_path = await runner.run()
    return artifact_path


# --------------------------------------------------------------------------- #
#  Step 7–8: Decide and update state
# --------------------------------------------------------------------------- #
def decide_and_update(
    state: CycleState,
    new_prompt: PromptArtifact,
    new_artifact_path: str,
    metrics_new: dict[str, Any],
    metrics_active: dict[str, Any],
    config: CycleConfig,
    config_path: str,
    source_candidates: list[int],
) -> tuple[str, str, list[dict[str, Any]]]:
    """Decide accept/rollback and update state.

    Returns ``(decision, reason, changed)``.  On accept, activates the new
    version via ``deps.prompt_repository`` and updates ``in_prompt`` flags.
    On rollback, increments the rollback counter.
    """
    vc_config = VersionComparatorConfig.from_config(config_path)
    decision, reason = decide(metrics_new, metrics_active, vc_config)
    changed: list[dict[str, Any]] = []

    if decision == "accept":
        state.accepted_history.append(state.active_version.version)
        state.active_version = new_prompt
        state.active_version_path = new_artifact_path
        state.rollback_counter = update_rollback_count(state.rollback_counter, decision)
        accepted_cluster_ids = {str(cid) for cid in source_candidates}
        if state.thesis_bank_path:
            update_in_prompt_flags(state.thesis_bank_path, accepted_cluster_ids)
    else:
        state.rollback_counter = update_rollback_count(state.rollback_counter, decision)
        state.rollback_history.append(
            {
                "round": state.round_counter + 1,
                "version": new_prompt.version,
                "reason": reason,
            }
        )

    return decision, reason, changed


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
        run_id=run_id,
        classify_prompt=prompt_artifact,
        duration=config.duration_for_round(round_number),
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
#  Step 9: Run a full round (steps 1–9)
# --------------------------------------------------------------------------- #
async def run_round(
    state: CycleState,
    deps: RunCycleDeps,
    config: CycleConfig,
    config_path: str,
    log_path: str,
) -> CycleState:
    """Execute one full round (steps 1–9) and return the updated state.

    On rollback within the round, consume the candidate queue (steps 5–7)
    until a candidate is accepted or the queue is exhausted.
    """
    round_number = state.round_counter + 1
    active_version_at_start = state.active_version.version
    log_event("round_start", round_number, {"active_version": active_version_at_start}, log_path)

    # Step 1: Run active on dev
    active_dev_path = await run_active_on_dev(state, deps, config, config_path)

    # Step 2: Compute dev metrics for active
    metrics_dev_active = compute_dev_metrics(active_dev_path, config_path)

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
    state.last_round_new_candidates = len(candidate_queue)

    # Steps 5–7: Compose, run, decide — retry on rollback
    decision = "rollback"
    reason = ""
    new_version = ""
    changed: list[dict[str, Any]] = []
    metrics_dev_new: dict[str, Any] = {}
    source_candidates: list[int] = []

    while state.candidate_queue:
        candidate = state.candidate_queue.pop(0)

        # Step 5: Compose new version
        new_prompt, new_artifact_path, source_candidates = await compose_new_version(
            candidate,
            state.active_version,
            config_path,
            deps,
            state.run_id,
        )
        new_version = new_prompt.version

        # Step 6: Run new on dev
        new_dev_path = await run_new_on_dev(
            new_prompt,
            config,
            config_path,
            deps,
            state.run_id,
            round_number=round_number,
        )
        metrics_dev_new = compute_dev_metrics(new_dev_path, config_path)

        # Step 7–8: Decide
        decision, reason, changed = decide_and_update(
            state,
            new_prompt,
            new_artifact_path,
            metrics_dev_new,
            metrics_dev_active,
            config,
            config_path,
            source_candidates,
        )

        log_event(
            "decision",
            round_number,
            {
                "decision": decision,
                "reason": reason,
                "new_version": new_version,
            },
            log_path,
        )

        if decision == "accept":
            log_event("accept", round_number, {"version": new_version}, log_path)
            break
        log_event("rollback", round_number, {"version": new_version}, log_path)

    # Holdout evaluation for both versions (logged only)
    metrics_holdout_active: dict[str, Any] = {}
    metrics_holdout_new: dict[str, Any] = {}
    try:
        active_for_holdout = (
            state.active_version
            if decision == "accept"
            else _restore_active(state, active_version_at_start)
        )
        metrics_holdout_active = await run_holdout(
            active_for_holdout,
            config,
            config_path,
            deps,
            state.run_id,
            round_number=round_number,
        )
        if new_version:
            metrics_holdout_new = await run_holdout(
                state.active_version,
                config,
                config_path,
                deps,
                state.run_id,
                round_number=round_number,
            )
    except Exception as exc:
        log.warning("holdout evaluation failed: %s", exc)

    # Step 9: Write per-round report
    next_action = "continue" if decision == "accept" else "rollback"
    report_path = write_report(
        round_number=round_number,
        active_version_at_start=active_version_at_start,
        new_version=new_version,
        decision=decision,
        metrics_dev_active=metrics_dev_active,
        metrics_dev_new=metrics_dev_new,
        metrics_holdout_active=metrics_holdout_active,
        metrics_holdout_new=metrics_holdout_new,
        rollback_counter=state.rollback_counter,
        changed_decisions=changed,
        next_action=next_action,
        run_id=state.run_id,
        run_repository=deps.run_repository,
        output_dir=config.output_dir,
    )
    state.latest_report_path = report_path

    log_event("round_end", round_number, {"decision": decision, "report": report_path}, log_path)
    return state


def _restore_active(state: CycleState, version: str) -> PromptArtifact:
    """Return the active version for holdout (the accepted one or the original)."""
    return state.active_version
