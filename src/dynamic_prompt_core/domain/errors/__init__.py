"""Public API for domain errors."""
from __future__ import annotations

from dynamic_prompt_core.domain.errors.dataset import DatasetError
from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError

__all__ = ["CandidateScoringError", "DatasetError"]
