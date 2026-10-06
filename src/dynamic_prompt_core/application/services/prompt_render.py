"""Prompt rendering service for the application layer."""
from __future__ import annotations

from dynamic_prompt_core.domain.prompts.base import PromptLayer, render


def render_prompt(layers: PromptLayer) -> str:
    """Render a prompt's layers into a single system-prompt string."""
    return render(layers)
