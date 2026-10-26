"""Resource loading for file-based system prompts."""
from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    """Load a system prompt text file by name (without ``.txt`` extension).

    Raises ``FileNotFoundError`` with a clear message if the file is missing.
    """
    try:
        return (files(__package__) / f"{name}.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"system prompt resource '{name}.txt' not found in {__package__}"
        ) from None
