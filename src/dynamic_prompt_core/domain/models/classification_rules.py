"""Immutable domain model for the normalized set of classification rules."""
from __future__ import annotations

from dataclasses import dataclass

from dynamic_prompt_core.domain.prompts.base import PromptArtifact


@dataclass(frozen=True)
class ClassificationRules:
    """An immutable, ordered set of classification rules.

    An empty ``ClassificationRules`` (zero rules) is a valid representation
    of the *absence* of rules.  It is distinct from invalid (malformed) rules,
    which raise at conversion time.
    """

    rules: tuple[str, ...]


def classification_rules_from_artifact(artifact: PromptArtifact) -> ClassificationRules:
    """Convert a ``PromptArtifact`` into ``ClassificationRules``.

    - ``layers is None`` or ``layers.rules == []`` → empty ``ClassificationRules``.
    - Non-string elements in ``layers.rules`` → ``TypeError``.
    - Rule order and text are preserved verbatim.
    """
    layers = artifact.layers
    if layers is None:
        return ClassificationRules(rules=())
    for r in layers.rules:
        if not isinstance(r, str):
            raise TypeError(
                f"rule {r!r} in prompt {artifact.version!r} is not a string"
            )
    return ClassificationRules(rules=tuple(layers.rules))
