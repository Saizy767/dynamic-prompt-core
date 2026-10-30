"""Prompt construction for classifier scoring.

Adapted from ``ScoringPromptBuilder`` to accept a loaded system prompt as the
initial prefix instead of a hardcoded constant.  The rules formatting,
candidate building, and prefix structure are preserved.
"""

from __future__ import annotations

from collections.abc import Sequence

from dynamic_prompt_core.domain.models.candidate import Candidate

_CANDIDATE_LABEL = "Candidate: "


def _format_rules(rules: Sequence[str]) -> str:
    if not rules:
        return ""
    lines = "\n".join(f"- {r}" for r in rules)
    return f"Classification rules:\n{lines}\n\n"


class ClassifierPromptBuilder:
    """Construct model-facing scoring prompts from semantic data.

    The prompt format is:

        {system_prompt}Classification rules:\\n- {rule 1}\\n...
        \\n\\nInput text:\\n{text}\\nCandidate: {candidate}
    """

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    def build_prefix(self, text: str, rules: Sequence[str]) -> str:
        return f"{self._system_prompt}{_format_rules(rules)}Input text:\n{text}\n{_CANDIDATE_LABEL}"

    def build_candidate(self, candidate: Candidate) -> str:
        return candidate.value

    def build(self, text: str, candidate: Candidate, rules: Sequence[str]) -> str:
        return self.build_prefix(text, rules) + self.build_candidate(candidate)
