"""Unit tests for results artifact filename parsing."""

from __future__ import annotations

import pytest

from dynamic_prompt_core.application.services.metrics.metrics import (
    MetricsError,
    parse_artifact_name,
)


class TestMetricsParseArtifactName:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            (
                "results_cycle-20261027T174033Z_v1_1_holdout_20261027T174135Z.jsonl",
                {
                    "run_id": "cycle-20261027T174033Z",
                    "prompt_version": "v1_1",
                    "split": "holdout",
                    "timestamp": "20261027T174135Z",
                },
            ),
            (
                "results_cycle-20261027T174033Z_v2_1_dev_20261027T174208Z.jsonl",
                {
                    "run_id": "cycle-20261027T174033Z",
                    "prompt_version": "v2_1",
                    "split": "dev",
                    "timestamp": "20261027T174208Z",
                },
            ),
            (
                "results_run-20260101T000000Z_v1_dev_20260101T000000Z.jsonl",
                {
                    "run_id": "run-20260101T000000Z",
                    "prompt_version": "v1",
                    "split": "dev",
                    "timestamp": "20260101T000000Z",
                },
            ),
            (
                "results_run-20260101T000000Z_0_holdout_20260101T000000Z.jsonl",
                {
                    "run_id": "run-20260101T000000Z",
                    "prompt_version": "0",
                    "split": "holdout",
                    "timestamp": "20260101T000000Z",
                },
            ),
        ],
    )
    def test_parses_run_id_version_split(self, filename: str, expected: dict[str, str]) -> None:
        assert parse_artifact_name(f"/some/dir/{filename}") == expected

    def test_rejects_non_results_filename(self) -> None:
        with pytest.raises(MetricsError, match="does not match"):
            parse_artifact_name("data/results/foo.jsonl")
