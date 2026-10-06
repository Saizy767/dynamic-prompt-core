"""Domain errors for dataset operations."""
from __future__ import annotations


class DatasetError(ValueError):
    """Raised when a dataset file is invalid or cannot be loaded."""
