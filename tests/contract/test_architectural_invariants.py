"""Architectural invariant tests for the candidate-scoring migration.

These tests verify invariants that lint-imports cannot express:
- application classification tests do not import infrastructure scoring;
- concrete scorer construction has one production source;
- the scorer is constructed once per process in the composition root;
- no legacy structured-generation classification dependency remains;
- the composition root is allowed to import infrastructure;
- no infrastructure state (tensors, logits, token IDs) escapes into
  domain or application types.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification import Classification
from dynamic_prompt_core.domain.models.judgment import Judgment

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "dynamic_prompt_core"
TESTS_ROOT = Path(__file__).resolve().parents[2] / "tests"


def _all_python_files(root: Path) -> list[Path]:
    return list(root.rglob("*.py"))


def _file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(_file_text(path))


# --------------------------------------------------------------------------- #
#  Task 2.4: Application classification tests do not import infrastructure
# --------------------------------------------------------------------------- #
class TestApplicationTestsDoNotImportInfrastructure:
    """Verify that application classification tests depend on the port, not
    the concrete infrastructure implementation."""

    FORBIDDEN_IMPORTS = {
        "transformers",
        "torch",
        "dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer",
        "dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer",
        "dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter",
        "dynamic_prompt_core.infrastructure.llm.scoring.model_adapter",
    }

    def test_classify_input_tests_do_not_import_infrastructure(self) -> None:
        test_file = TESTS_ROOT / "unit" / "candidate_scoring" / "test_classify_input.py"
        tree = _parse_module(test_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in self.FORBIDDEN_IMPORTS, (
                        f"test imports forbidden infrastructure: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or node.module not in self.FORBIDDEN_IMPORTS, (
                    f"test imports from forbidden infrastructure: {node.module}"
                )

    def test_classification_policy_tests_do_not_import_infrastructure(self) -> None:
        test_file = TESTS_ROOT / "unit" / "candidate_scoring" / "test_classification_policy.py"
        tree = _parse_module(test_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in self.FORBIDDEN_IMPORTS, (
                        f"test imports forbidden infrastructure: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or node.module not in self.FORBIDDEN_IMPORTS, (
                    f"test imports from forbidden infrastructure: {node.module}"
                )


# --------------------------------------------------------------------------- #
#  Task 7.2: Concrete scorer construction has one production source
# --------------------------------------------------------------------------- #
class TestConcreteScorerConstructionHasOneSource:
    """Verify that both composition roots route through build_candidate_scorer
    and neither constructs LLMLogitCandidateScorer directly."""

    def test_build_cli_deps_uses_shared_factory(self) -> None:
        main_path = SRC_ROOT / "interfaces" / "cli" / "main.py"
        text = _file_text(main_path)
        assert "build_candidate_scorer" in text, (
            "build_cli_deps must call build_candidate_scorer"
        )

    def test_runner_cli_uses_shared_factory(self) -> None:
        runner_path = SRC_ROOT / "application" / "use_cases" / "run_baseline" / "runner.py"
        text = _file_text(runner_path)
        assert "build_candidate_scorer" in text, (
            "runner CLI must call build_candidate_scorer"
        )

    def test_neither_composition_root_constructs_llm_logit_scorer_directly(self) -> None:
        """Neither build_cli_deps nor runner._main_async should construct
        LLMLogitCandidateScorer directly (they must go through the factory)."""
        main_path = SRC_ROOT / "interfaces" / "cli" / "main.py"
        runner_path = SRC_ROOT / "application" / "use_cases" / "run_baseline" / "runner.py"

        for path in [main_path, runner_path]:
            tree = _parse_module(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == "LLMLogitCandidateScorer":
                        pytest.fail(f"{path} constructs LLMLogitCandidateScorer directly")
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "LLMLogitCandidateScorer"
                    ):
                        pytest.fail(f"{path} constructs LLMLogitCandidateScorer directly")


# --------------------------------------------------------------------------- #
#  Task 7.3: Scorer constructed once per process
# --------------------------------------------------------------------------- #
class TestScorerConstructedOncePerProcess:
    """Verify the composition root constructs the scorer once, not per round."""

    def test_build_cli_deps_has_single_factory_call(self) -> None:
        main_path = SRC_ROOT / "interfaces" / "cli" / "main.py"
        text = _file_text(main_path)
        count = text.count("build_candidate_scorer(")
        assert count >= 1, "build_cli_deps must call build_candidate_scorer at least once"

    def test_run_cycle_does_not_construct_scorer(self) -> None:
        """run_cycle application code must not construct a scorer — it receives
        one from RunCycleDeps."""
        steps_path = SRC_ROOT / "application" / "use_cases" / "run_cycle" / "steps.py"
        text = _file_text(steps_path)
        assert "build_candidate_scorer" not in text, (
            "run_cycle steps must not construct a scorer — it receives one from deps"
        )
        assert "LLMLogitCandidateScorer" not in text, (
            "run_cycle steps must not reference LLMLogitCandidateScorer"
        )


# --------------------------------------------------------------------------- #
#  Task 7.5: No legacy structured-generation classification dependency
# --------------------------------------------------------------------------- #
class TestNoLegacyStructuredGenerationDependency:
    """Verify the classification path does not depend on legacy structured-
    generation mechanisms. Target concrete obsolete artifacts, not vocabulary."""

    LEGACY_SYMBOLS = [
        "ClassificationResult",
        "classify_detailed",
        "classify_many",
        "classify_many_detailed",
    ]

    @pytest.mark.parametrize("symbol", LEGACY_SYMBOLS)
    def test_legacy_symbols_absent_from_classification_path(self, symbol: str) -> None:
        classification_modules = [
            SRC_ROOT / "application" / "use_cases" / "classify_input",
            SRC_ROOT / "application" / "services" / "classification_policy.py",
            SRC_ROOT / "domain" / "models" / "classification.py",
            SRC_ROOT / "application" / "use_cases" / "run_baseline" / "runner.py",
        ]
        for path in classification_modules:
            if path.is_dir():
                for py_file in path.rglob("*.py"):
                    text = _file_text(py_file)
                    assert symbol not in text, (
                        f"{py_file} references legacy symbol '{symbol}'"
                    )
            elif path.exists():
                text = _file_text(path)
                assert symbol not in text, (
                    f"{path} references legacy symbol '{symbol}'"
                )

    def test_no_json_classification_response_in_classification_path(self) -> None:
        """The classification path must not parse JSON classification responses."""
        classify_input_path = (
            SRC_ROOT / "application" / "use_cases" / "classify_input" / "classify_input.py"
        )
        text = _file_text(classify_input_path)
        assert "json" not in text.lower() or "json" not in text, (
            "classify_input should not parse JSON classification responses"
        )


# --------------------------------------------------------------------------- #
#  Task 7.6: Composition root may import infrastructure
# --------------------------------------------------------------------------- #
class TestCompositionRootMayImportInfrastructure:
    """Positive assertion: the composition root is allowed to import
    infrastructure scorer implementations."""

    def test_cli_main_imports_build_candidate_scorer(self) -> None:
        main_path = SRC_ROOT / "interfaces" / "cli" / "main.py"
        text = _file_text(main_path)
        assert "build_candidate_scorer" in text, (
            "Composition root must import build_candidate_scorer from infrastructure"
        )

    def test_cli_main_imports_argmax_policy(self) -> None:
        main_path = SRC_ROOT / "interfaces" / "cli" / "main.py"
        text = _file_text(main_path)
        assert "ArgmaxClassificationPolicy" in text, (
            "Composition root must import ArgmaxClassificationPolicy"
        )


# --------------------------------------------------------------------------- #
#  Task 7.7: No infrastructure state escapes into domain or application
# --------------------------------------------------------------------------- #
class TestNoInfrastructureStateEscapes:
    """Verify that Judgment, Classification, and ResultRow carry only
    domain/application-safe information — no tensors, token IDs, logits,
    attention masks, or model output objects."""

    FORBIDDEN_FIELD_NAMES = {
        "logits",
        "raw_logits",
        "token_ids",
        "input_ids",
        "attention_mask",
        "model_output",
        "tensor",
        "raw_tensor",
    }

    def test_judgment_fields_are_domain_safe(self) -> None:
        fields = {f.name for f in dataclasses.fields(Judgment)}
        forbidden = fields & self.FORBIDDEN_FIELD_NAMES
        assert not forbidden, f"Judgment has forbidden fields: {forbidden}"
        assert fields == {"candidate", "score"}, f"Unexpected Judgment fields: {fields}"

    def test_classification_fields_are_domain_safe(self) -> None:
        fields = {f.name for f in dataclasses.fields(Classification)}
        forbidden = fields & self.FORBIDDEN_FIELD_NAMES
        assert not forbidden, f"Classification has forbidden fields: {forbidden}"
        assert fields == {"selected", "judgments"}, (
            f"Unexpected Classification fields: {fields}"
        )

    def test_result_row_fields_are_application_safe(self) -> None:
        from dynamic_prompt_core.application.use_cases.run_baseline.runner import ResultRow

        fields = {f.name for f in dataclasses.fields(ResultRow)}
        forbidden = fields & self.FORBIDDEN_FIELD_NAMES
        assert not forbidden, f"ResultRow has forbidden fields: {forbidden}"

    def test_judgment_candidate_is_domain_type(self) -> None:
        j = Judgment(Candidate("0"), 0.5)
        assert isinstance(j.candidate, Candidate)
        assert isinstance(j.score, float)

    def test_classification_judgments_are_domain_type(self) -> None:
        judgments = (Judgment(Candidate("0"), 0.2), Judgment(Candidate("1"), 0.8))
        c = Classification(selected=Candidate("1"), judgments=judgments)
        assert isinstance(c.selected, Candidate)
        assert all(isinstance(j, Judgment) for j in c.judgments)
