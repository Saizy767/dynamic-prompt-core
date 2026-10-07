"""Tests for state dump/load and in_prompt flag updates."""

from __future__ import annotations

import json

from dynamic_prompt_core.application.use_cases.run_cycle.state import (
    CycleState,
    dump_state,
    load_state,
    update_in_prompt_flags,
)
from dynamic_prompt_core.domain.prompts.base import PromptArtifact


def test_dump_state_produces_valid_json(tmp_path, mock_run_repository):
    """dump_state produces a valid JSON file with run_id and timestamp."""
    state = CycleState(round_counter=2, run_id="test-run")
    state.thesis_bank_path = "thesis.json"
    state.candidate_queue = [{"cluster_id": 1}]
    state.rollback_counter = 1

    path = dump_state(state, mock_run_repository, str(tmp_path))
    assert path.endswith(".json")
    assert "test-run" in path
    data = json.loads(open(path).read())
    if isinstance(data, list):
        data = data[0]
    assert data["round_counter"] == 2
    assert data["rollback_counter"] == 1


def test_load_state_reproduces_all_fields(tmp_path, mock_run_repository):
    """load_state reproduces round_counter, candidate_queue, rollback_counter."""
    state = CycleState(
        round_counter=3,
        run_id="test-run",
        thesis_bank_path="thesis.json",
        rollback_counter=2,
    )
    state.candidate_queue = [{"cluster_id": 5}, {"cluster_id": 6}]
    state.accepted_history = ["v0", "v1"]

    path = dump_state(state, mock_run_repository, str(tmp_path))
    loaded = load_state(path)

    assert loaded.round_counter == 3
    assert loaded.rollback_counter == 2
    assert len(loaded.candidate_queue) == 2
    assert loaded.candidate_queue[0]["cluster_id"] == 5
    assert loaded.accepted_history == ["v0", "v1"]
    assert loaded.thesis_bank_path == "thesis.json"


def test_load_state_preserves_active_version(tmp_path, mock_run_repository):
    """load_state preserves the active version."""
    custom_prompt = PromptArtifact(version="v3", layers=None, text="custom")
    state = CycleState(round_counter=1, active_version=custom_prompt, run_id="r")
    path = dump_state(state, mock_run_repository, str(tmp_path))
    loaded = load_state(path)
    assert loaded.active_version.version == "v3"
    assert loaded.active_version.text == "custom"


def test_update_in_prompt_flags_sets_correct_values(tmp_path):
    """update_in_prompt_flags sets in_prompt=true for accepted clusters."""
    bank = {
        "theses": {
            "t1": {"cluster_id": 1, "in_prompt": False, "text_raw": "t1"},
            "t2": {"cluster_id": 2, "in_prompt": True, "text_raw": "t2"},
            "t3": {"cluster_id": 3, "in_prompt": False, "text_raw": "t3"},
        },
        "clusters": {},
    }
    bank_path = str(tmp_path / "thesis_bank.json")
    with open(bank_path, "w") as f:
        json.dump(bank, f)

    update_in_prompt_flags(bank_path, {"1", "3"})

    with open(bank_path) as f:
        updated = json.load(f)
    assert updated["theses"]["t1"]["in_prompt"] is True
    assert updated["theses"]["t2"]["in_prompt"] is False
    assert updated["theses"]["t3"]["in_prompt"] is True


def test_resume_from_dumped_state_preserves_fields(tmp_path, mock_run_repository):
    """Resume from a dumped state at round 3 preserves all fields."""
    state = CycleState(
        round_counter=3,
        run_id="resume-test",
        rollback_counter=1,
        thesis_bank_path="bank.json",
        clusters_path="clusters.json",
    )
    state.candidate_queue = [{"cluster_id": 7}]

    path = dump_state(state, mock_run_repository, str(tmp_path))
    loaded = load_state(path)

    assert loaded.round_counter == 3
    assert loaded.rollback_counter == 1
    assert loaded.thesis_bank_path == "bank.json"
    assert loaded.clusters_path == "clusters.json"
    assert loaded.candidate_queue == [{"cluster_id": 7}]
