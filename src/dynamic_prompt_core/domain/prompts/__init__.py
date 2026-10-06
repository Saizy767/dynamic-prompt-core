"""Public API for the domain prompts package."""
from __future__ import annotations

from dynamic_prompt_core.domain.prompts.base import (
    PromptArtifact,
    PromptLayer,
    build_classification_prompt,
    render,
)
from dynamic_prompt_core.domain.prompts.fixed import (
    CLASSIFICATION_PROMPT_V0,
    EXTRACTION_PROMPT,
)

__all__ = [
    "CLASSIFICATION_PROMPT_V0",
    "EXTRACTION_PROMPT",
    "PromptArtifact",
    "PromptLayer",
    "build_classification_prompt",
    "render",
]
