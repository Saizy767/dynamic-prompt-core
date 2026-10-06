"""Fixed (non-optimized) prompts: classification v0 and extraction v0."""
from __future__ import annotations

from dynamic_prompt_core.domain.prompts.base import (
    PromptArtifact,
    PromptLayer,
    build_classification_prompt,
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
    rules=[
        "Decide based only on the content of the text, not on hashtags, "
        "mentions, or URLs alone.",
        "If the text uses target-category words metaphorically or in jest, "
        "classify as 0.",
        "Factual statements that clearly match the target category are 1.",
        "If the evidence is genuinely ambiguous, lean toward 0 and set a "
        "low confidence.",
    ],
    output_contract=(
        "Reply with a single JSON object with two fields: "
        '"decision" (integer 0 or 1) and "confidence" (integer 0-100). '
        "No markdown, no extra text."
    ),
    fallback=(
        "If you cannot decide, return decision 0 with confidence 0."
    ),
)

CLASSIFICATION_PROMPT_V0: PromptArtifact = build_classification_prompt(
    "classify-v0", _V0_LAYERS
)


_EXTRACTION_TEXT = (
    "## Role\n"
    "You are a thesis extraction assistant. You read a short text and extract "
    "its key claims.\n\n"
    "## Task\n"
    "Extract 3 to 5 concise theses from the user text. Each thesis must be a "
    "short phrase of 2 to 6 words capturing a distinct claim or fact.\n\n"
    "## Output contract\n"
    'Reply with a single JSON object: {"theses": ["...", "...", "..."]}. '
    "No markdown, no extra text.\n\n"
    "## Fallback\n"
    'If the text is too short or unclear, return {"theses": ["text too short", '
    '"unable to extract", "no clear claims"]}.'
)

EXTRACTION_PROMPT: PromptArtifact = PromptArtifact(
    version="extract-v0",
    layers=None,
    text=_EXTRACTION_TEXT,
)
