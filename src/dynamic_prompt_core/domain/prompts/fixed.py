"""Fixed (non-optimized) prompt: classification v0."""

from __future__ import annotations

from dynamic_prompt_core.domain.prompts.base import (
    PromptArtifact,
    PromptLayer,
    render,
)

_V0_LAYERS = PromptLayer(
    role=(
        "You are a binary text classifier. You read a short text and decide "
        "whether it belongs to class 1 or class 0."
    ),
    task=(
        "Classify the user text into one of two classes: 1 (the text matches "
        "the target category described by the rules) or 0 (the text does not "
        "match, e.g. it is unrelated, metaphorical, or off-topic)."
    ),
    rules=[],
    output_contract=(
        "Reply with a single JSON object with two fields: "
        '"decision" (integer 0 or 1) and "confidence" (integer 0-100). '
        "No markdown, no extra text."
    ),
    fallback=("If you cannot decide, return decision 0 with confidence 0."),
)

CLASSIFICATION_PROMPT_V0: PromptArtifact = PromptArtifact(
    version="classify-v0",
    layers=_V0_LAYERS,
    text=render(_V0_LAYERS),
)
