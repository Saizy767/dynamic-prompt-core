"""Prompt construction for candidate scoring.

The prompt builder owns the model-facing scoring prompt format.  It exposes
the prefix and candidate portions separately so the scorer can determine the
tokenization boundary without substring search.

The active classification rules are supplied per-call as semantic criteria
strings (from ``PromptArtifact.layers.rules``).  The builder incorporates them
into the prompt between the fixed candidate-evaluation instructions and the
input text.  The builder does NOT duplicate or reference the default
classification rules from ``CLASSIFICATION_PROMPT_V0``.
"""
from __future__ import annotations

from collections.abc import Sequence

from dynamic_prompt_core.domain.models.candidate import Candidate

_CANDIDATE_LABEL = "Candidate: "

_FIXED_INSTRUCTIONS = (
    "You are a binary text classifier. You read a short text and evaluate "
    "how well a candidate label matches the text.\n\n"
    "Evaluate how well the candidate describes or matches the input text "
    "according to the classification rules below.\n\n"
)


def _format_rules(rules: Sequence[str]) -> str:
    if not rules:
        return ""
    lines = "\n".join(f"- {r}" for r in rules)
    return f"Classification rules:\n{lines}\n\n"


class ScoringPromptBuilder:
    """Construct model-facing scoring prompts from semantic data.

    The prompt format is:

        {fixed_instructions}Classification rules:\\n- {rule 1}\\n...
        \\n\\nInput text:\\n{text}\\nCandidate: {candidate}

    The prefix (everything up to and including ``Candidate: ``) and the
    candidate (``candidate.value`` verbatim) are exposed separately so the
    scorer can encode them independently and determine the tokenization
    boundary structurally.

    The candidate continuation is the final textual component of the prompt:
    no semantic instructions or other generated text appear between the
    candidate label and the candidate value.
    """

    def build_prefix(self, text: str, rules: Sequence[str]) -> str:
        """Return the prompt text preceding the candidate value."""
        return f"{_FIXED_INSTRUCTIONS}{_format_rules(rules)}Input text:\n{text}\n{_CANDIDATE_LABEL}"

    def build_candidate(self, candidate: Candidate) -> str:
        """Return the candidate value verbatim, with no normalization."""
        return candidate.value

    def build(self, text: str, candidate: Candidate, rules: Sequence[str]) -> str:
        """Return the full scoring prompt."""
        return self.build_prefix(text, rules) + self.build_candidate(candidate)
