"""
Prompt store: baseline classification prompt v0 and fixed extraction prompt.

Both are exposed as PromptArtifact dataclasses carrying a version, the layered
content, the rendered text, and a sha256 hash (first 16 hex chars, matching
the transport layer's system_prompt_hash style).

Rendering uses plain string joins — no external templating dependency.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PromptLayer:
    """The five semantic layers of a classification prompt."""
    role: str
    task: str
    rules: List[str]
    output_contract: str
    fallback: str


def render(layers: PromptLayer) -> str:
    """Join the five layers into a single system-prompt string."""
    rules_text = "\n".join(f"- {r}" for r in layers.rules)
    return (
        f"## Role\n{layers.role}\n\n"
        f"## Task\n{layers.task}\n\n"
        f"## Rules\n{rules_text}\n\n"
        f"## Output contract\n{layers.output_contract}\n\n"
        f"## Fallback\n{layers.fallback}"
    )


@dataclass(frozen=True)
class PromptArtifact:
    """A versioned, hashed prompt artifact ready to pass to the client."""
    version: str
    layers: Optional[PromptLayer]
    text: str
    sha256: str = field(default="")

    def __post_init__(self) -> None:
        if not self.sha256:
            object.__setattr__(
                self, "sha256",
                hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16],
            )


def build_classification_prompt(
    version: str, layers: PromptLayer
) -> PromptArtifact:
    """Assemble a classification prompt artifact, enforcing 3-5 rules."""
    if not 3 <= len(layers.rules) <= 5:
        raise ValueError(
            f"classification prompt must have 3-5 rules, "
            f"got {len(layers.rules)}"
        )
    text = render(layers)
    return PromptArtifact(version=version, layers=layers, text=text)


# --------------------------------------------------------------------------- #
#  Classification prompt v0
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Fixed extraction prompt (not optimized on Stage 1)
# --------------------------------------------------------------------------- #
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
