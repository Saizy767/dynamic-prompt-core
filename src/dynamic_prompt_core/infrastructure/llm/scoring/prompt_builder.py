"""Prompt construction for candidate scoring.

The prompt builder owns the model-facing scoring prompt format.  It exposes
the prefix and candidate portions separately so the scorer can determine the
tokenization boundary without substring search.
"""
from __future__ import annotations

from dynamic_prompt_core.domain.models.candidate import Candidate

_CANDIDATE_LABEL = "Candidate: "


class ScoringPromptBuilder:
    """Construct model-facing scoring prompts from semantic data.

    The prompt format is:

        {input}\\nCandidate: {candidate}

    The prefix (``{input}\\nCandidate: ``) and the candidate (``candidate.value``
    verbatim) are exposed separately so the scorer can encode them independently
    and determine the tokenization boundary structurally.
    """

    def build_prefix(self, text: str) -> str:
        """Return the prompt text preceding the candidate value."""
        return f"{text}\n{_CANDIDATE_LABEL}"

    def build_candidate(self, candidate: Candidate) -> str:
        """Return the candidate value verbatim, with no normalization."""
        return candidate.value

    def build(self, text: str, candidate: Candidate) -> str:
        """Return the full scoring prompt."""
        return self.build_prefix(text) + self.build_candidate(candidate)
