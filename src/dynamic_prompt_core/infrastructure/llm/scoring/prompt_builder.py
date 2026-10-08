"""Prompt construction for candidate scoring.

The prompt builder owns the model-facing scoring prompt format.  It exposes
the prefix and candidate portions separately so the scorer can determine the
tokenization boundary without substring search.

The semantic prefix intentionally duplicates the relevant semantic content of
``CLASSIFICATION_PROMPT_V0``.  The two prompts MUST NOT be treated as
automatically synchronized: changes to the generative prompt do not
automatically modify the judgment prompt, and changes to the judgment prompt
require explicit review.
"""
from __future__ import annotations

from dynamic_prompt_core.domain.models.candidate import Candidate

_CANDIDATE_LABEL = "Candidate: "

_SEMANTIC_PREFIX = (
    "You are a binary text classifier. You read a short text and evaluate "
    "how well a candidate label matches the text.\n\n"
    "Evaluate how well the candidate describes or matches the input text "
    "according to the following criteria:\n"
    "- The candidate is relevant if the text matches the target category "
    "described by the candidate.\n"
    "- The candidate is irrelevant if the text does not match, e.g. it is "
    "unrelated, metaphorical, or off-topic.\n"
    "- Base the evaluation only on the content of the text, not on hashtags, "
    "mentions, or URLs alone.\n"
    "- If the text uses target-category words metaphorically or in jest, the "
    "candidate is not a good match.\n"
    "- Factual statements that clearly match the target category indicate a "
    "good match.\n"
    "- If the evidence is genuinely ambiguous, the candidate is a weak match.\n\n"
)


class ScoringPromptBuilder:
    """Construct model-facing scoring prompts from semantic data.

    The prompt format is:

        {semantic_prefix}{input}\\nCandidate: {candidate}

    The prefix (``{semantic_prefix}{input}\\nCandidate: ``) and the candidate
    (``candidate.value`` verbatim) are exposed separately so the scorer can
    encode them independently and determine the tokenization boundary
    structurally.

    The candidate continuation is the final textual component of the prompt:
    no semantic instructions or other generated text appear between the
    candidate label and the candidate value.
    """

    def build_prefix(self, text: str) -> str:
        """Return the prompt text preceding the candidate value."""
        return f"{_SEMANTIC_PREFIX}{text}\n{_CANDIDATE_LABEL}"

    def build_candidate(self, candidate: Candidate) -> str:
        """Return the candidate value verbatim, with no normalization."""
        return candidate.value

    def build(self, text: str, candidate: Candidate) -> str:
        """Return the full scoring prompt."""
        return self.build_prefix(text) + self.build_candidate(candidate)
