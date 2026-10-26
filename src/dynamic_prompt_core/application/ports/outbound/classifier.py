"""Outbound port: contract for classification."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dynamic_prompt_core.domain.models.classification import Classification
from dynamic_prompt_core.domain.models.classification_rules import ClassificationRules


@runtime_checkable
class Classifier(Protocol):
    """Outbound port: classify input text against a set of rules.

    Encapsulates the full classification process: rule validation, candidate
    handling, logit-based scoring, and final decision.  Returns a
    ``Classification`` directly — the application does not orchestrate
    scoring and decision separately.

    The port exposes no logits, token IDs, tokenizer APIs, model objects, or
    provider-specific types.
    """

    async def classify(
        self,
        text: str,
        rules: ClassificationRules,
    ) -> Classification:
        """Classify ``text`` using the active ``ClassificationRules``.

        Returns a ``Classification`` with the selected candidate and full
        judgment set.  Raises an exception if classification fails.
        """
        ...
