"""Backend and structured-output mode enums for the LLM transport layer."""
from __future__ import annotations

from enum import StrEnum


class Backend(StrEnum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OPENAI = "openai"


class StructuredMode(StrEnum):
    JSON_SCHEMA = "json_schema"   # native constrained decoding
    JSON_OBJECT = "json_object"   # valid JSON only; schema injected into prompt
    NONE = "none"                 # no constraints at all
