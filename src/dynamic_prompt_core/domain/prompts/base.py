"""Base prompt layers and artifacts for the classification pipeline."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptLayer:
    """The five semantic layers of a classification prompt."""
    role: str
    task: str
    rules: list[str]
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
    layers: PromptLayer | None
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
