"""Backend adapters for building OpenAI-compatible chat payloads."""
from __future__ import annotations

import json
from functools import cache
from typing import Any

from pydantic import BaseModel

from dynamic_prompt_core.infrastructure.llm.enums import Backend, StructuredMode


def _strip_chat_suffix(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    suffix = "/chat/completions"
    if endpoint.endswith(suffix):
        endpoint = endpoint[: -len(suffix)]
    return endpoint


class _BaseAdapter:
    structured_mode: StructuredMode

    def chat_url(self, endpoint: str) -> str:
        return _strip_chat_suffix(endpoint) + "/chat/completions"

    def build_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        json_schema_obj: dict[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _schema_into_prompt(system_prompt: str, schema: dict[str, Any]) -> str:
        return (
            f"{system_prompt}\n\n"
            "Reply with a single JSON object that strictly matches this JSON "
            "Schema (no markdown fences, no commentary):\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )


class _VLLMAdapter(_BaseAdapter):
    def __init__(self, mode: StructuredMode) -> None:
        self.structured_mode = mode

    def build_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        json_schema_obj: dict[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if self.structured_mode is StructuredMode.JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema_obj,
            }
        elif self.structured_mode is StructuredMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
            messages[0]["content"] = self._schema_into_prompt(
                system_prompt, json_schema_obj["schema"]
            )
        return payload


class _OpenAIAdapter(_VLLMAdapter):
    """Generic OpenAI-compatible: like vLLM, but no chat_template_kwargs."""
    def build_payload(self, **kw: Any) -> dict[str, Any]:
        payload = super().build_payload(**kw)
        payload.pop("chat_template_kwargs", None)
        return payload


class _LlamaCppAdapter(_BaseAdapter):
    def __init__(self, mode: StructuredMode) -> None:
        self.structured_mode = mode

    def build_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        json_schema_obj: dict[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        # llama-server does not understand chat_template_kwargs and, on older
        # builds, not json_schema either. Default to json_object + prompt.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.structured_mode is StructuredMode.JSON_SCHEMA:
            # Supported by recent llama.cpp builds; older ones return 400.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema_obj,
            }
        elif self.structured_mode is StructuredMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
            messages[0]["content"] = self._schema_into_prompt(
                system_prompt, json_schema_obj["schema"]
            )
        return payload


_ADAPTERS = {
    Backend.VLLM: _VLLMAdapter,
    Backend.OPENAI: _OpenAIAdapter,
    Backend.LLAMACPP: _LlamaCppAdapter,
}

_DEFAULT_MODE = {
    Backend.VLLM: StructuredMode.JSON_SCHEMA,
    Backend.OPENAI: StructuredMode.JSON_SCHEMA,
    Backend.LLAMACPP: StructuredMode.JSON_OBJECT,
}


@cache
def _schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    return {
        "name": "request_analysis",
        "description": "Analysis of requests",
        "schema": model_cls.model_json_schema(),
        "strict": True,
    }
