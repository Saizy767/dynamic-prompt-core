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


def test_cycle_state_cycle_counter_starts_at_zero():
    """CycleState starts with cycle_counter=0."""
    state = CycleState(run_id="test")
    assert state.cycle_counter == 0
    assert state.any_accepted is False
    assert state.candidate_queue == []


def test_cycle_state_active_version_defaults_to_v0():
    """CycleState defaults to CLASSIFICATION_PROMPT_V0."""
    state = CycleState(run_id="test")
    assert state.active_version.version == CLASSIFICATION_PROMPT_V0.version


def test_run_cycle_step_executes_steps_and_writes_report(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """run_cycle_step chains all nine steps and writes a report."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    mock_pc_config = MagicMock(max_rules=10, max_prompt_tokens=100_000)

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
        stack.enter_context(patch(f"{STEPS}.decide", return_value=("accept", "improved")))
        stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(
            patch(f"{STEPS}.PromptComposerConfig.from_config", return_value=mock_pc_config)
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))
        stack.enter_context(patch(f"{STEPS}.update_in_prompt_flags"))

        updated = asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert updated.latest_report_path == "report.json"


def test_run_cycle_step_persists_accepted_version_to_store(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """On accept, run_cycle_step calls prompt_repository.save then activate."""
    from dataclasses import replace

    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step

    deps = build_mock_deps()

    saved_calls: list[dict] = []
    activated: list[int] = []

    class _TrackingRepo:
        def save(self, prompt_version_dict, reason="composed", base_version=None):
            saved_calls.append(prompt_version_dict)
            return {"version": 7, "text": prompt_version_dict["text"]}

        def activate(self, version_number):
            activated.append(version_number)
            return {"version": version_number}

        def get_active(self):
            raise KeyError("no active version")

        def get(self, version_number):
            return {}

        def list_versions(self):
            return []

        def lineage(self, version_number):
            return []

    deps = replace(deps, prompt_repository=_TrackingRepo())

    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    prompt_v1 = _make_prompt_with_rules("classify-v1", _v0_rules() + ["rule-1"])
    mock_pc_config = MagicMock(max_rules=10, max_prompt_tokens=100_000)

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection",
                return_value=("thesis.json", "clusters.json"),
            )
        )
        stack.enter_context(
            patch(f"{STEPS}.select_candidates", return_value=("cands.json", [{"cluster_id": 1}]))
        )
        stack.enter_context(
            patch(
                f"{STEPS}.compose_new_version",
                new_callable=AsyncMock,
                return_value=(prompt_v1, "prompt_v1.json", [1]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.run_new_on_dev", new_callable=AsyncMock, return_value="new_results.jsonl"
            )
        )
        stack.enter_context(patch(f"{STEPS}.decide", return_value=("accept", "improved")))
        stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(
            patch(f"{STEPS}.PromptComposerConfig.from_config", return_value=mock_pc_config)
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))
        stack.enter_context(patch(f"{STEPS}.update_in_prompt_flags"))

        asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert len(saved_calls) == 1
    assert activated == [7]


def test_run_active_on_dev_threads_scorer_policy_candidates(build_mock_deps):
    """run_active_on_dev passes scorer, policy, and candidates from deps to BaselineRunner."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        run_active_on_dev,
    )

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    config = CycleConfig(max_cycles=1, run_id="test-run")

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with (
        patch(f"{STEPS}.BaselineRunner", _CapturingRunner),
        patch(f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()),
    ):
        asyncio.run(run_active_on_dev(state, deps, config, "config.toml"))

    assert captured["classifier"] is deps.classifier
    assert captured["extractor"] is deps.extractor
    assert list(deps.candidates) == captured["candidates"]


def test_run_active_on_dev_passes_active_prompt_with_rules(build_mock_deps):
    """run_active_on_dev passes state.active_version as classify_prompt so the
    runner can extract rules from it."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        run_active_on_dev,
    )

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    config = CycleConfig(max_cycles=1, run_id="test-run")

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with (
        patch(f"{STEPS}.BaselineRunner", _CapturingRunner),
        patch(f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()),
    ):
        asyncio.run(run_active_on_dev(state, deps, config, "config.toml"))

    assert captured["classify_prompt"] is state.active_version
    """run_new_on_dev passes scorer, policy, and candidates from deps to BaselineRunner."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        run_new_on_dev,
    )

    deps = build_mock_deps()
    config = CycleConfig(max_cycles=1, run_id="test-run")

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with (
        patch(f"{STEPS}.BaselineRunner", _CapturingRunner),
        patch(f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()),
    ):
        asyncio.run(
            run_new_on_dev(CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run")
        )

    assert captured["classifier"] is deps.classifier
    assert captured["extractor"] is deps.extractor
    assert list(deps.candidates) == captured["candidates"]


def test_run_holdout_threads_scorer_policy_candidates(build_mock_deps):
    """run_holdout passes scorer, policy, and candidates from deps to BaselineRunner."""
    from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_holdout

    deps = build_mock_deps()
    config = CycleConfig(max_cycles=1, run_id="test-run")

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with (
        patch(f"{STEPS}.BaselineRunner", _CapturingRunner),
        patch(f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()),
        patch(f"{STEPS}.compute_dev_metrics", return_value={"accuracy": 0.5}),
    ):
        asyncio.run(run_holdout(CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run"))

    assert captured["classifier"] is deps.classifier
    assert captured["extractor"] is deps.extractor
    assert list(deps.candidates) == captured["candidates"]
    assert captured["skip_extraction"] is True


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
    config = CycleConfig(max_cycles=1, run_id="test-run")

    captured_list: list[dict] = []

    class _CapturingRunner:
        def __init__(self, **kwargs):
            captured_list.append(kwargs)

        async def run(self):
            return [], "fake.jsonl"

    with (
        patch(f"{STEPS}.BaselineRunner", _CapturingRunner),
        patch(f"{STEPS}.RunnerConfig.from_config", return_value=MagicMock()),
        patch(f"{STEPS}.compute_dev_metrics", return_value={"accuracy": 0.5}),
    ):
        asyncio.run(run_active_on_dev(state, deps, config, "config.toml"))
        asyncio.run(
            run_new_on_dev(CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run")
        )
        asyncio.run(run_holdout(CLASSIFICATION_PROMPT_V0, config, "config.toml", deps, "test-run"))

    assert len(captured_list) == 3
    classifiers = [c["classifier"] for c in captured_list]
    extractors = [c["extractor"] for c in captured_list]
    candidates = [c["candidates"] for c in captured_list]
    assert all(cl is classifiers[0] for cl in classifiers)
    assert all(ex is extractors[0] for ex in extractors)
    assert all(c == candidates[0] for c in candidates)


def test_run_cycle_deps_requires_classifier():
    """RunCycleDeps construction without classifier raises TypeError."""
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
        RunCycleDeps,
    )

    with pytest.raises(TypeError):
        RunCycleDeps(  # type: ignore[call-arg]
            llm_client=MagicMock(),
            prompt_repository=MagicMock(),
            run_repository=MagicMock(),
            normalizer=MagicMock(),
            extractor=MagicMock(),
            candidates=(MagicMock(),),
        )


def test_run_cycle_deps_requires_extractor():
    """RunCycleDeps construction without extractor raises TypeError."""
    from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import (
        RunCycleDeps,
    )

    with pytest.raises(TypeError):
        RunCycleDeps(  # type: ignore[call-arg]
            llm_client=MagicMock(),
            prompt_repository=MagicMock(),
            run_repository=MagicMock(),
            normalizer=MagicMock(),
            classifier=MagicMock(),
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
            normalizer=MagicMock(),
            classifier=MagicMock(),
            extractor=MagicMock(),
        )


def _make_prompt_with_rules(version: str, rules: list[str]):
    """Build a PromptArtifact with the given rules for testing."""
    from dynamic_prompt_core.domain.prompts.base import PromptArtifact, PromptLayer, render

    layers = PromptLayer(
        role="test role",
        task="test task",
        rules=rules,
        output_contract="test contract",
        fallback="test fallback",
    )
    return PromptArtifact(version=version, layers=layers, text=render(layers))


def _v0_rules() -> list[str]:
    """Return the default V0 classification rules."""
    from dynamic_prompt_core.domain.prompts import CLASSIFICATION_PROMPT_V0

    layers = CLASSIFICATION_PROMPT_V0.layers
    return list(layers.rules) if layers else []


def test_run_round_accumulates_rules_on_accept(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """run_cycle_step accumulates rules from multiple candidates when both improve metrics."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    prompt_v1 = _make_prompt_with_rules("classify-v1", _v0_rules() + ["rule-1"])
    prompt_v2 = _make_prompt_with_rules("classify-v2", _v0_rules() + ["rule-1", "rule-2"])

    mock_pc_config = MagicMock(max_rules=10, max_prompt_tokens=100_000)

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection",
                return_value=("thesis.json", "clusters.json"),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.select_candidates",
                return_value=("cands.json", [{"cluster_id": 1}, {"cluster_id": 2}]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.compose_new_version",
                new_callable=AsyncMock,
                side_effect=[
                    (prompt_v1, "prompt_v1.json", [1]),
                    (prompt_v2, "prompt_v2.json", [2]),
                ],
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.run_new_on_dev",
                new_callable=AsyncMock,
                return_value="new_results.jsonl",
            )
        )
        stack.enter_context(
            patch(f"{STEPS}.decide", side_effect=[("accept", "improved"), ("accept", "improved")])
        )
        stack.enter_context(
            patch(f"{STEPS}.PromptComposerConfig.from_config", return_value=mock_pc_config)
        )
        stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))
        stack.enter_context(patch(f"{STEPS}.update_in_prompt_flags"))

        updated = asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert updated.active_version.version == "v1"
    assert updated.any_accepted is True


def test_run_round_stops_on_degradation(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """run_cycle_step keeps the first accepted rule when the second candidate degrades."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    prompt_v1 = _make_prompt_with_rules("classify-v1", _v0_rules() + ["rule-1"])
    prompt_v2 = _make_prompt_with_rules("classify-v2", _v0_rules() + ["rule-1", "rule-2"])

    mock_pc_config = MagicMock(max_rules=10, max_prompt_tokens=100_000)

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection",
                return_value=("thesis.json", "clusters.json"),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.select_candidates",
                return_value=("cands.json", [{"cluster_id": 1}, {"cluster_id": 2}]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.compose_new_version",
                new_callable=AsyncMock,
                side_effect=[
                    (prompt_v1, "prompt_v1.json", [1]),
                    (prompt_v2, "prompt_v2.json", [2]),
                ],
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.run_new_on_dev",
                new_callable=AsyncMock,
                return_value="new_results.jsonl",
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.decide",
                side_effect=[("accept", "improved"), ("rollback", "degraded")],
            )
        )
        stack.enter_context(
            patch(f"{STEPS}.PromptComposerConfig.from_config", return_value=mock_pc_config)
        )
        stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))
        stack.enter_context(patch(f"{STEPS}.update_in_prompt_flags"))

        updated = asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert updated.active_version.version == "v1"
    assert updated.any_accepted is True


def test_run_round_no_acceptance_increments_rollback(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """run_cycle_step keeps active version when no candidate is accepted."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    prompt_v1 = _make_prompt_with_rules("classify-v1", _v0_rules() + ["rule-1"])

    mock_pc_config = MagicMock(max_rules=10, max_prompt_tokens=100_000)

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection",
                return_value=("thesis.json", "clusters.json"),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.select_candidates",
                return_value=("cands.json", [{"cluster_id": 1}, {"cluster_id": 2}]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.compose_new_version",
                new_callable=AsyncMock,
                return_value=(prompt_v1, "prompt_v1.json", [1]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.run_new_on_dev",
                new_callable=AsyncMock,
                return_value="new_results.jsonl",
            )
        )
        stack.enter_context(patch(f"{STEPS}.decide", return_value=("rollback", "degraded")))
        stack.enter_context(
            patch(f"{STEPS}.PromptComposerConfig.from_config", return_value=mock_pc_config)
        )
        stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))
        stack.enter_context(patch(f"{STEPS}.update_in_prompt_flags"))

        updated = asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert updated.any_accepted is False
    assert updated.active_version.version == CLASSIFICATION_PROMPT_V0.version


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


def test_refine_theses_step_merges_refined_into_full_rows(tmp_path, build_mock_deps):
    """Regression: teacher refinement must not crash the analyzer.

    The refined artifact alone lacks the analyzer's required fields
    (true_label, statuses, theses_norm); refine_theses_step must return the
    original rows with the refined theses merged in.
    """
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        refine_theses_step,
    )
    from dynamic_prompt_core.domain.services.thesis_refinement import (
        RefinementReview,
    )

    class _DroppingTeacher:
        async def review_theses(self, text, theses):
            return RefinementReview(
                keep=["thesis one"],
                reformulate=[],
                drop=[{"thesis": "thesis two", "reason": "noisy"}],
                add=["thesis three"],
                latency_ms=1.0,
                usage={},
            )

    results_path = tmp_path / "results_test-run_v1_dev_20260101T000000Z.jsonl"
    row = {
        "id": 1,
        "text": "a",
        "true_label": 1,
        "predicted_decision": 1,
        "selected_candidate": "1",
        "judgment_scores": {"0": 0.1, "1": 0.9},
        "theses_raw": ["thesis one", "thesis two"],
        "theses_norm": ["thesis one", "thesis two"],
        "extract_status": "ok",
        "classify_status": "ok",
        "extract_latency_ms": 1.0,
        "classify_latency_ms": 1.0,
        "raw_extract": None,
    }
    with open(results_path, "w") as f:
        f.write(json.dumps(row) + "\n")

    deps = build_mock_deps(teacher=_DroppingTeacher())
    language = MagicMock()
    language.language = "ru"
    with patch(f"{STEPS}.RunnerConfig.from_config", return_value=language):
        merged = asyncio.run(
            refine_theses_step(str(results_path), "test-run", "v1", deps, "config.toml")
        )

    assert merged is not None
    assert len(merged) == 1
    merged_row = merged[0]
    assert merged_row["id"] == 1
    assert merged_row["true_label"] == 1
    assert merged_row["classify_status"] == "ok"
    assert merged_row["extract_status"] == "ok"
    assert merged_row["theses_raw"] == ["thesis one", "thesis three"]
    assert merged_row["theses_norm"] == ["thesis one", "thesis three"]


def test_refine_theses_step_without_teacher_returns_none(build_mock_deps):
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        refine_theses_step,
    )

    deps = build_mock_deps(teacher=None)
    merged = asyncio.run(refine_theses_step("results.jsonl", "test-run", "v1", deps, "config.toml"))
    assert merged is None


def test_update_thesis_collection_uses_refined_rows(tmp_path):
    """With refined_rows given, the raw artifact is not loaded; names still
    come from results_path."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import (
        update_thesis_collection,
    )

    results_path = str(tmp_path / "results_test-run_v1_dev_20260101T000000Z.jsonl")
    refined_rows = [
        {
            "id": 1,
            "true_label": 1,
            "theses_norm": ["t1"],
            "theses_raw": ["t1"],
            "extract_status": "ok",
            "classify_status": "ok",
        }
    ]
    with (
        patch(f"{STEPS}.ThesisAnalyzerConfig.from_config", return_value=MagicMock()),
        patch(
            f"{STEPS}.analyze",
            return_value=(MagicMock(), {"run_id": "test-run", "prompt_version": "v1"}),
        ) as mock_analyze,
        patch(f"{STEPS}.write_dump", return_value="thesis_bank.json") as mock_dump,
    ):
        bank_path, clusters_path = update_thesis_collection(
            results_path, "config.toml", refined_rows=refined_rows
        )

    rows_arg, _cfg, source_arg = mock_analyze.call_args[0]
    assert rows_arg is refined_rows
    assert source_arg == results_path
    assert bank_path == "thesis_bank.json"
    assert clusters_path == "thesis_bank.json"
    assert mock_dump.called


def test_active_holdout_failure_is_retried(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """Regression: one transient failure of the active holdout run must not
    doom the round — it is retried once and the round proceeds."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}
    holdout_metrics = {
        "accuracy": 0.5,
        "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5},
    }

    flaky_holdout = AsyncMock(side_effect=[RuntimeError("server hiccup"), holdout_metrics])

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection",
                return_value=("thesis.json", "clusters.json"),
            )
        )
        stack.enter_context(patch(f"{STEPS}.select_candidates", return_value=("cands.json", [])))
        stack.enter_context(patch(f"{STEPS}.run_holdout", flaky_holdout))
        stack.enter_context(
            patch(f"{STEPS}.PromptComposerConfig.from_config", return_value=MagicMock(max_rules=10))
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))

        updated = asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert flaky_holdout.await_count == 2
    assert updated.latest_report_path == "report.json"


def test_active_holdout_persistent_failure_raises(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """Two failed attempts abort the round with a clear error instead of
    silently rolling every candidate back and stopping with no_improvement."""
    from dynamic_prompt_core.application.use_cases.run_cycle.report import (
        CycleOrchestratorError,
    )
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    failing_holdout = AsyncMock(side_effect=RuntimeError("server down"))

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection",
                return_value=("thesis.json", "clusters.json"),
            )
        )
        stack.enter_context(patch(f"{STEPS}.select_candidates", return_value=("cands.json", [])))
        stack.enter_context(patch(f"{STEPS}.run_holdout", failing_holdout))

        with pytest.raises(CycleOrchestratorError, match="holdout baseline unavailable"):
            asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert failing_holdout.await_count == 2


def test_run_round_skips_candidate_rejected_by_limits(
    build_mock_deps,
    cycle_config,
    tmp_output_dir,
):
    """A candidate whose composed prompt fails validate_limits is skipped;
    the next candidate is still evaluated and can be accepted."""
    from dynamic_prompt_core.application.use_cases.run_cycle.steps import run_cycle_step
    from dynamic_prompt_core.domain.prompts.base import (
        PromptArtifact,
        PromptLayer,
        render,
    )

    deps = build_mock_deps()
    state = CycleState(run_id="test-run")
    log_path = os.path.join(tmp_output_dir, "cycle_log.jsonl")
    os.makedirs(tmp_output_dir, exist_ok=True)

    mock_runner = AsyncMock()
    mock_runner.run = AsyncMock(return_value=([], "fake_results.jsonl"))
    metrics = {"accuracy": 0.5, "f1": {"macro": 0.5, "minority": 0.5, "weighted": 0.5}}

    bad_layers = PromptLayer(
        role="test role",
        task="test task",
        rules=_v0_rules() + ["rule-1"],
        output_contract="",
        fallback="test fallback",
    )
    prompt_bad = PromptArtifact(version="v1_1", layers=bad_layers, text=render(bad_layers))
    prompt_good = _make_prompt_with_rules("classify-v1", _v0_rules() + ["rule-2"])

    mock_pc_config = MagicMock(max_rules=10, max_prompt_tokens=100_000)

    with ExitStack() as stack:
        stack.enter_context(patch(f"{STEPS}.BaselineRunner", return_value=mock_runner))
        stack.enter_context(patch(f"{STEPS}.compute_dev_metrics", return_value=metrics))
        stack.enter_context(
            patch(
                f"{STEPS}.update_thesis_collection",
                return_value=("thesis.json", "clusters.json"),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.select_candidates",
                return_value=("cands.json", [{"cluster_id": 1}, {"cluster_id": 2}]),
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.compose_new_version",
                new_callable=AsyncMock,
                side_effect=[
                    (prompt_bad, "prompt_bad.json", [1]),
                    (prompt_good, "prompt_good.json", [2]),
                ],
            )
        )
        stack.enter_context(
            patch(
                f"{STEPS}.run_new_on_dev",
                new_callable=AsyncMock,
                return_value="new_results.jsonl",
            )
        )
        stack.enter_context(patch(f"{STEPS}.decide", side_effect=[("accept", "improved")]))
        stack.enter_context(
            patch(f"{STEPS}.PromptComposerConfig.from_config", return_value=mock_pc_config)
        )
        holdout_mock = stack.enter_context(
            patch(f"{STEPS}.run_holdout", new_callable=AsyncMock, return_value={"accuracy": 0.5})
        )
        stack.enter_context(patch(f"{STEPS}.write_report", return_value="report.json"))
        stack.enter_context(patch(f"{STEPS}.update_in_prompt_flags"))

        updated = asyncio.run(run_cycle_step(state, deps, cycle_config, "config.toml", log_path))

    assert updated.active_version.version == "v1"
    assert updated.any_accepted is True
    # active-baseline holdout + the one valid candidate; the rejected
    # candidate never reaches holdout
    assert holdout_mock.await_count == 2
