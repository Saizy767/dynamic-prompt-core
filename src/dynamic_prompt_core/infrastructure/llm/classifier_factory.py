"""Composition-root factory for the classifier.

Constructs an ``LLMLogitClassifier`` with all infrastructure dependencies
from a ``ScorerBackendConfig``.  The result structurally satisfies the
``Classifier`` port; callers consume it through the port.
"""

from __future__ import annotations

import logging

from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.classifier import LLMLogitClassifier
from dynamic_prompt_core.infrastructure.llm.scoring.config import ScorerBackendConfig

log = logging.getLogger(__name__)


def build_classifier(
    config: ScorerBackendConfig,
    candidates: list[Candidate],
) -> LLMLogitClassifier:
    """Construct a ``Classifier`` backed by LLM logits.

    Dispatches on ``config.backend``: the sole backend is ``"llama_server"``
    (HTTP scoring against a running llama-server).  Returns an
    ``LLMLogitClassifier`` with the shared ``ClassifierPromptBuilder`` (loaded
    from the classifier system-prompt resource) and ``LogitScorer``.
    """
    from dynamic_prompt_core.infrastructure.llm.prompts import load_prompt
    from dynamic_prompt_core.infrastructure.llm.scoring.classifier_prompt_builder import (
        ClassifierPromptBuilder,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
        LogitScorer,
    )

    system_prompt = load_prompt("classifier")
    prompt_builder = ClassifierPromptBuilder(system_prompt=system_prompt)
    logit_scorer = LogitScorer()

    assert config.llama_server is not None
    log.info(
        "Building classifier: backend=%s, endpoint=%s",
        config.backend,
        config.llama_server.endpoint,
    )
    if config.backend == "llama_server":
        tokenizer, model = _build_llama_server_adapters(config)
    else:
        raise ValueError(
            f"unsupported scorer backend {config.backend!r}; "
            f"ScorerBackendConfig validation should have rejected it"
        )

    return LLMLogitClassifier(
        prompt_builder=prompt_builder,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        logit_scorer=logit_scorer,
        candidates=candidates,
    )


def _build_llama_server_adapters(config: ScorerBackendConfig) -> tuple[object, object]:
    from dynamic_prompt_core.infrastructure.llm.scoring.llama_server_model_adapter import (
        LlamaServerCausalLMAdapter,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.llama_server_tokenizer_adapter import (
        LlamaServerBatchTokenizerAdapter,
    )

    assert config.llama_server is not None
    ls = config.llama_server
    model_adapter = LlamaServerCausalLMAdapter(
        endpoint=ls.endpoint,
        timeout=ls.timeout,
        served_model_name=None,
        gguf_path=ls.gguf_path,
        max_concurrency=ls.max_concurrency,
    )
    tokenizer_adapter = LlamaServerBatchTokenizerAdapter(
        endpoint=ls.endpoint,
        timeout=ls.timeout,
    )
    return tokenizer_adapter, model_adapter
