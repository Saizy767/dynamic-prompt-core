"""ClassifyInput use case: orchestrate candidate scoring and classification."""
from __future__ import annotations

from dynamic_prompt_core.application.use_cases.classify_input.classify_input import (
    classify_input,
)
from dynamic_prompt_core.application.use_cases.classify_input.deps import (
    ClassifyInputDeps,
)

__all__ = ["ClassifyInputDeps", "classify_input"]
