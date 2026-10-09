"""Tests for the nine-step round sequence and step functions."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamic_prompt_core.application.use_cases.run_cycle.state import CycleState
from dynamic_prompt_core.domain.prompts.fixed import CLASSIFICATION_PROMPT_V0

STEPS = "dynamic_prompt_core.application.use_cases.run_cycle.steps"


def test_compute_dev_metrics_returns_expected_keys(tmp_path):
    """Step 2: compute_dev_metrics returns accuracy and f1."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        compute_dev_metrics,
    )

    results_path = tmp_path / "results_test_v0_dev_20260101T000000Z.jsonl"
    rows = [
        {
            "id": 1,
            "text": "a",
            "true_label": 1,
            "predicted_decision": 1,
            "classify_status": "ok",
            "theses_norm": ["t1"],
        },
        {
            "id": 2,
            "text": "b",
            "true_label": 0,
            "predicted_decision": 0,
            "classify_status": "ok",
            "theses_norm": ["t2"],
        },
    ]
    with open(results_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg_path = "dynamic_prompt_core.application.services.metrics"
    with patch(f"{cfg_path}.metrics.MetricsConfig.from_config") as mock_cfg:
        mock_cfg.return_value = MagicMock(
            output_dir=str(tmp_path),
            bias_threshold=0.15,
            group_short_max=10,
            group_long_min=30,
        )
        metrics = compute_dev_metrics(str(results_path), "config.toml")

    assert "accuracy" in metrics
    assert "f1" in metrics


def test_cycle_state_round_counter_starts_at_zero():
    """CycleState starts with round_counter=0."""
    state = CycleState(run_id="test")
    assert state.round_counter == 0
    assert state.rollback_counter == 0
    assert state.candidate_queue == []


def test_cycle_state_active_version_defaults_to_v0():
    """CycleState defaults to CLASSIFICATION_PROMPT_V0."""
    state = CycleState(run_id="test")
    assert state.active_version.version == CLASSIFICATION_PROMPT_V0.version


def test_run_round_executes_steps_and_writes_report(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """run_round chains all nine steps and writes a report."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_round

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection", return_value=("thesis.json", "clusters.json")
            )
        )
        stack.enter_context(
            patch(f"{STEPS}.select_candidates", return_value=("cands.json", [{"cluster_id": 1}]))
        )
        stack.enter_context(
            patch(
                f"{STEPS}.compose_new_version",
                new_callable=AsyncMock,
                return_value=(CLASSIFICATION_PROMPT_V0, "prompt.json", [1]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.run_new_on_dev", new_callable=AsyncMock, return_value="new_results.jsonl"
            )
        )
        stack.enter_context(
            patch(f"{STEPS}.decide_and_update", return_value=("accept", "improved", []))
        )
        stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))

        updated = asyncio.run(run_round(state, deps, cycle_config, "config.toml", log_path))

    assert updated.latest_report_path == "report.json"


def test_run_active_on_dev_threads_scorer_policy_candidates(build_mock_deps):
    """run_active_on_dev passes scorer, policy, and candidates from deps to BaselineRunner."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        run_active_on_dev,
    )

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    config = CycleConfig(max_rounds=1, run_id="test-run")

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with patch(f"{STEPS}.BaselineRunner", _CapturingRunner), patch(
        f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()
    ):
        asyncio.run(run_active_on_dev(state, deps, config, "config.toml"))

    assert captured["scorer"] is deps.candidate_scorer
    assert captured["policy"] is deps.classification_policy
    assert list(deps.candidates) == captured["candidates"]


def test_run_new_on_dev_threads_scorer_policy_candidates(build_mock_deps):
    """run_new_on_dev passes scorer, policy, and candidates from deps to BaselineRunner."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        run_new_on_dev,
    )

    deps = build_mock_deps()
    config = CycleConfig(max_rounds=1, run_id="test-run")

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with patch(f"{STEPS}.BaselineRunner", _CapturingRunner), patch(
        f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()
    ):
        asyncio.run(
            run_new_on_dev(
                CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run"
            )
        )

    assert captured["scorer"] is deps.candidate_scorer
    assert captured["policy"] is deps.classification_policy
    assert list(deps.candidates) == captured["candidates"]


def test_run_holdout_threads_scorer_policy_candidates(build_mock_deps):
    """run_holdout passes scorer, policy, and candidates from deps to BaselineRunner."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_holdout

    deps = build_mock_deps()
    config = CycleConfig(max_rounds=1, run_id="test-run")

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with patch(f"{STEPS}.BaselineRunner", _CapturingRunner), patch(
        f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()
    ), patch(f"{STEPS}.compute_dev_metrics", return_value={"accuracy": 0.5}):
        asyncio.run(
            run_holdout(
                CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run"
            )
        )

    assert captured["scorer"] is deps.candidate_scorer
    assert captured["policy"] is deps.classification_policy
    assert list(deps.candidates) == captured["candidates"]


def test_all_runners_receive_same_scorer_and_policy_instances(build_mock_deps):
    """All three BaselineRunner constructions during a round receive the same
    scorer and policy object instances from RunCycleDeps."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        run_active_on_dev,
        run_holdout,
        run_new_on_dev,
    )

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    config = CycleConfig(max_rounds=1, run_id="test-run")

    captured_list: list[dict] = []

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured_list.append(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with patch(f"{STEPS}.BaselineRunner", _CapturingRunner), patch(
        f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()
    ), patch(f"{STEPS}.compute_dev_metrics", return_value={"accuracy": 0.5}):
        asyncio.run(run_active_on_dev(state, deps, config, "config.toml"))
        asyncio.run(
            run_new_on_dev(
                CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run"
            )
        )
        asyncio.run(
            run_holdout(
                CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run"
            )
        )

    assert len(captured_list) == 3
    scorers = [c["scorer"] for c in captured_list]
    policies = [c["policy"] for c in captured_list]
    candidates = [c["candidates"] for c in captured_list]
    assert all(s is scorers[0] for s in scorers)
    assert all(p is policies[0] for p in policies)
    assert all(c == candidates[0] for c in candidates)


def test_run_cycle_deps_requires_candidate_scorer():
    """RunCycleDeps construction without candidate_scorer raises TypeError."""
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
        RunCycleDeps,
    )

    with pytest.raises(TypeError):
        RunCycleDeps(  # type: ignore[call-arg]
            llm_client=MagicMock(),
            prompt_repository=MagicMock(),
            run_repository=MagicMock(),
            dataset_repository=MagicMock(),
            embedding_client=MagicMock(),
            normalizer=MagicMock(),
            classification_policy=MagicMock(),
            candidates=(MagicMock(),),
        )


def test_run_cycle_deps_requires_classification_policy():
    """RunCycleDeps construction without classification_policy raises TypeError."""
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
        RunCycleDeps,
    )

    with pytest.raises(TypeError):
        RunCycleDeps(  # type: ignore[call-arg]
            llm_client=MagicMock(),
            prompt_repository=MagicMock(),
            run_repository=MagicMock(),
            dataset_repository=MagicMock(),
            embedding_client=MagicMock(),
            normalizer=MagicMock(),
            candidate_scorer=MagicMock(),
            candidates=(MagicMock(),),
        )


def test_run_cycle_deps_requires_candidates():
    """RunCycleDeps construction without candidates raises TypeError."""
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
        RunCycleDeps,
    )

    with pytest.raises(TypeError):
        RunCycleDeps(  # type: ignore[call-arg]
            llm_client=MagicMock(),
            prompt_repository=MagicMock(),
            run_repository=MagicMock(),
            dataset_repository=MagicMock(),
            embedding_client=MagicMock(),
            normalizer=MagicMock(),
            candidate_scorer=MagicMock(),
            classification_policy=MagicMock(),
        )


def test_candidate_source_of_truth_in_steps():
    """No BaselineRunner construction in steps.py supplies a candidates= argument
    other than deps.candidates. Inspect BaselineRunner(...) calls specifically."""
    import ast
    from pathlib import Path

    steps_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "dynamic_prompt_core"
        / "application"
        / "use_cases"
        / "run_cycle"
        / "steps.py"
    )
    tree = ast.parse(steps_path.read_text(encoding="utf-8"))

    baseline_runner_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "BaselineRunner":
                baseline_runner_calls.append(node)
            elif isinstance(func, ast.Attribute) and func.attr == "BaselineRunner":
                baseline_runner_calls.append(node)

    assert len(baseline_runner_calls) >= 3, (
        f"Expected at least 3 BaselineRunner calls in steps.py, found {len(baseline_runner_calls)}"
    )

    for call in baseline_runner_calls:
        candidates_arg = None
        for kw in call.keywords:
            if kw.arg == "candidates":
                candidates_arg = kw.value
                break

        assert candidates_arg is not None, (
            "BaselineRunner construction in steps.py must pass candidates="
        )

        assert isinstance(candidates_arg, ast.Call), (
            "candidates= argument must be a function call (e.g., list(deps.candidates))"
        )

        func = candidates_arg.func
        is_deps_candidates = False
        if isinstance(func, ast.Name) and func.id == "list":
            if candidates_arg.args:
                arg = candidates_arg.args[0]
                if (
                    isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Name)
                    and arg.value.id == "deps"
                    and arg.attr == "candidates"
                ):
                    is_deps_candidates = True
        elif isinstance(func, ast.Attribute) and func.attr == "candidates":
            if isinstance(func.value, ast.Name) and func.value.id == "deps":
                is_deps_candidates = True

        assert is_deps_candidates, (
            "BaselineRunner candidates= must be derived from deps.candidates, "
            f"not a step-local construction: {ast.dump(candidates_arg)}"
        )
