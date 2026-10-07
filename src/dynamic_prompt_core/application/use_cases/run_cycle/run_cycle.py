"""run_cycle: the central optimization loop use case.

Threads the round counter, candidate queue, and rollback counter across
rounds, calling the nine-step round sequence from ``steps.py``.  Produces
per-round reports, a final summary, and reloadable state dumps.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.application.ports.inbound.run_cycle_input import RunCycleInput
from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.report import (
    CycleOrchestratorError,
    log_event,
    write_summary,
)
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
    RunCycleDeps,
)
from dynamic_prompt_core.application.use_cases.run_cycle.state import (
    CycleState,
    dump_state,
    load_state,
)
from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_round
from dynamic_prompt_core.domain.prompts.fixed import CLASSIFICATION_PROMPT_V0

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunCycleResult:
    """Result of a run_cycle execution."""

    total_rounds: int
    stop_reason: str
    final_active_version: str
    accepted_history: list[str]
    rollback_history: list[dict[str, str | int]]


def check_stop(state: CycleState, config: CycleConfig) -> tuple[bool, str | None]:
    """Check whether the cycle should stop.

    Returns ``(should_stop, reason)``.
    """
    if state.round_counter >= config.max_rounds:
        return True, "max_rounds_reached"
    if state.rollback_counter >= config.max_consecutive_rollbacks:
        return True, "max_rollbacks_reached"
    if not state.candidate_queue and state.rollback_counter > 0:
        return True, "candidate_queue_exhausted"
    return False, None


def _extract_metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract ``accuracy``, ``macro_f1``, ``minority_f1`` from a metrics dict."""
    f1 = metrics.get("f1", {})
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(f1.get("macro", 0.0)),
        "minority_f1": float(f1.get("minority", 0.0)),
    }


def _extract_rule_set(state: CycleState) -> frozenset[str]:
    """Extract the set of rule ids from the active prompt version."""
    layers = state.active_version.layers
    if layers is not None:
        return frozenset(layers.rules)
    return frozenset()


def load_initial_state(
    deps: RunCycleDeps,
    config: CycleConfig,
    config_path: str,
) -> CycleState:
    """Load the initial cycle state.

    Loads the active prompt via ``deps.prompt_repository`` (or v0 when no
    active version exists) and the dataset via ``deps.dataset_repository``.
    Raises ``CycleOrchestratorError`` when a required artifact is missing.
    """
    run_id = config.run_id or f"cycle-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    active_version = CLASSIFICATION_PROMPT_V0
    active_version_path = ""
    try:
        active_record = deps.prompt_repository.get_active()
        version_str = str(active_record.get("version", "classify-v0"))
        text = active_record.get("text", CLASSIFICATION_PROMPT_V0.text)
        sha = active_record.get("hash", "")
        active_version = type(CLASSIFICATION_PROMPT_V0)(
            version=version_str,
            layers=None,
            text=text,
            sha256=sha,
        )
        active_version_path = str(active_record.get("path", ""))
    except Exception as exc:
        log.debug("no active prompt in repository, using v0: %s", exc)

    dataset_artifact = ""
    try:
        from dynamic_prompt_core.application.use_cases.run_baseline.runner import (
            RunnerConfig,
        )

        runner_config = RunnerConfig.from_config(config_path)
        dataset_artifact = runner_config.dataset_artifact
    except Exception:
        pass

    if dataset_artifact:
        try:
            deps.dataset_repository.load_artifact(dataset_artifact)
        except Exception as exc:
            raise CycleOrchestratorError(
                f"dataset artifact not found at {dataset_artifact!r}: {exc}"
            ) from exc
    else:
        raise CycleOrchestratorError(
            "dataset artifact path is required (config [runner].dataset_artifact)"
        )

    return CycleState(
        round_counter=0,
        active_version=active_version,
        active_version_path=active_version_path,
        run_id=run_id,
    )


async def run_cycle(deps: RunCycleDeps, run_input: RunCycleInput) -> RunCycleResult:
    """Execute the optimization loop for a configured number of rounds.

    This use case receives all dependencies via the typed ``deps`` object.
    It does not instantiate infrastructure classes directly.
    """
    config = run_input.config or CycleConfig.from_toml(run_input.config_path)
    if run_input.max_rounds != config.max_rounds:
        config = type(config)(**{**config.__dict__, "max_rounds": run_input.max_rounds})

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(
        output_dir,
        f"cycle_log_{config.run_id or 'default'}_{ts}.jsonl",
    )

    all_changed_decisions: list[dict[str, Any]] = []
    stop_reason = "unrecoverable_error"

    try:
        if run_input.resume_from:
            state = load_state(run_input.resume_from)
            log.info("resuming from round %d", state.round_counter)
        else:
            state = load_initial_state(deps, config, run_input.config_path)

        log_event(
            "cycle_start",
            0,
            {"run_id": state.run_id, "max_rounds": config.max_rounds},
            log_path,
        )

        metric_history: list[dict[str, float]] = []
        decision_history: list[str] = []
        rule_set_history: list[frozenset[str]] = []

        while True:
            should_stop, reason = check_stop(state, config)
            if should_stop:
                stop_reason = reason or "max_rounds_reached"
                break

            state = await run_round(state, deps, config, run_input.config_path, log_path)
            state.round_counter += 1

            if config.dump_state_after_each_round:
                dump_state(state, deps.run_repository, output_dir)

            if state.latest_report_path:
                try:
                    from dynamic_prompt_core.application.use_cases.run_cycle.report import (
                        load_report,
                    )

                    report = load_report(state.latest_report_path)
                    metrics_dev = report.get("metrics_dev_new") or report.get(
                        "metrics_dev_active", {}
                    )
                    metric_history.append(_extract_metric_summary(metrics_dev))
                    decision_history.append(str(report.get("decision", "rollback")))
                except Exception as exc:
                    log.debug("could not load report for history: %s", exc)
            rule_set_history.append(_extract_rule_set(state))

            if deps.stop_criteria_deps is not None:
                from dynamic_prompt_core.application.ports.outbound.stop_criteria import (
                    StopEvaluationContext,
                )
                from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria import (
                    evaluate_stop_criteria,
                )

                stop_config = config.stop_criteria_config
                ctx = StopEvaluationContext(
                    round_counter=state.round_counter,
                    rollback_counter=state.rollback_counter,
                    max_consecutive_rollbacks=config.max_consecutive_rollbacks,
                    candidate_queue_size=len(state.candidate_queue),
                    new_candidates_found=state.last_round_new_candidates,
                    metric_history=list(metric_history),
                    decision_history=list(decision_history),
                    rule_set_history=list(rule_set_history),
                    total_teacher_tokens=0,
                    run_id=state.run_id,
                    log_path=log_path,
                    config=stop_config,
                )
                stop_decision = await evaluate_stop_criteria(
                    deps.stop_criteria_deps, ctx, output_dir=output_dir
                )
                if stop_decision.should_stop:
                    stop_reason = stop_decision.reason or "plateau_detected"
                    break

        final_dev_metrics: dict[str, Any] = {}
        final_holdout_metrics: dict[str, Any] = {}
        try:
            if state.latest_report_path:
                from dynamic_prompt_core.application.use_cases.run_cycle.report import (
                    load_report,
                )

                report = load_report(state.latest_report_path)
                final_dev_metrics = report.get("metrics_dev_new") or report.get(
                    "metrics_dev_active", {}
                )
                final_holdout_metrics = report.get("metrics_holdout_new") or report.get(
                    "metrics_holdout_active", {}
                )
        except Exception as exc:
            log.warning("could not load final metrics: %s", exc)

        write_summary(
            state,
            final_dev_metrics,
            final_holdout_metrics,
            stop_reason,
            all_changed_decisions,
            state.run_id,
            deps.run_repository,
            output_dir,
        )

        log_event("stop", state.round_counter, {"reason": stop_reason}, log_path)

        return RunCycleResult(
            total_rounds=state.round_counter,
            stop_reason=stop_reason,
            final_active_version=state.active_version.version,
            accepted_history=state.accepted_history,
            rollback_history=state.rollback_history,
        )

    except CycleOrchestratorError:
        raise
    except Exception as exc:
        log.exception("unrecoverable error during cycle execution")
        log_event("error", 0, {"error": str(exc)}, log_path)
        raise
