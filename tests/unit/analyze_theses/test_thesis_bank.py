"""Incremental cluster counters and atomic thesis-bank dump writes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer import (
    ThesisAnalyzerConfig,
    ThesisBank,
    write_dump,
)


def _bank_with_two_clusters() -> ThesisBank:
    """Build a bank with one 2-member cluster and one 1-member cluster.

    alpha/alpha2 embeddings are near-identical (cosine >= cosine_upper), beta
    is orthogonal, so it forms its own cluster. Repeats with mixed labels
    exercise the incremental counter path in add_or_update.
    """
    bank = ThesisBank(ThesisAnalyzerConfig())
    a = np.array([1.0, 0.0], dtype=np.float32)
    a2 = np.array([0.99, 0.01], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)

    bank.add_or_update("alpha", "Alpha raw", 1, precomputed_embedding=a)
    bank.add_or_update("alpha2", "Alpha2 raw", 0, precomputed_embedding=a2)
    bank.add_or_update("beta", "Beta raw", 1, precomputed_embedding=b)
    bank.add_or_update("alpha", "Alpha raw 2", 0, precomputed_embedding=a)
    bank.add_or_update("alpha", "Alpha raw 3", 1, precomputed_embedding=a)
    bank.add_or_update("beta", "Beta raw 2", 0, precomputed_embedding=b)
    bank.add_or_update("alpha2", "Alpha2 raw 2", 1, precomputed_embedding=a2)
    return bank


class TestIncrementalClusterCounters:
    def test_counters_match_full_recompute(self) -> None:
        bank = _bank_with_two_clusters()
        assert len(bank.clusters) == 2

        for cid, cluster in bank.clusters.items():
            before = (
                cluster.frequency,
                cluster.positive_hits,
                cluster.negative_hits,
                cluster.demand,
                cluster.selected,
                cluster.precision,
            )
            bank.recompute_cluster_counters(cid)
            after = (
                cluster.frequency,
                cluster.positive_hits,
                cluster.negative_hits,
                cluster.demand,
                cluster.selected,
                cluster.precision,
            )
            assert before == pytest.approx(after), f"cluster {cid} drifted from recompute"

    def test_expected_counter_values(self) -> None:
        bank = _bank_with_two_clusters()
        c0 = bank.clusters[0]
        assert (c0.count, c0.frequency, c0.positive_hits, c0.negative_hits) == (2, 5, 3, 2)
        assert c0.precision == pytest.approx(0.6)
        c1 = bank.clusters[1]
        assert (c1.count, c1.frequency, c1.positive_hits, c1.negative_hits) == (1, 2, 1, 1)
        assert c1.precision == pytest.approx(0.5)

        alpha = bank.theses["alpha"]
        assert (alpha.frequency, alpha.positive_hits, alpha.negative_hits) == (3, 2, 1)


class TestWriteDumpAtomic:
    def test_round_trip_and_no_temp_leftovers(self, tmp_path: Path) -> None:
        bank = _bank_with_two_clusters()
        config = ThesisAnalyzerConfig(output_dir=str(tmp_path))
        path = write_dump(bank, "run1", "v1", config, str(tmp_path))

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["metadata"]["run_id"] == "run1"
        assert data["theses"]["alpha"]["frequency"] == 3
        assert data["clusters"]["0"]["frequency"] == 5
        assert not list(tmp_path.rglob(".*tmp*")), "temp files must not remain"
