"""Composition-root factory for the classifier.

Constructs an ``LLMLogitClassifier`` with all infrastructure dependencies
from a ``ScorerBackendConfig``.  Callers receive the result through the
``Classifier`` port.
"""
from __future__ import annotations

import logging

from dynamic_prompt_core.application.ports.outbound.classifier import Classifier
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.infrastructure.llm.scoring.config import ScorerBackendConfig

log = logging.getLogger(__name__)


def build_classifier(
    config: ScorerBackendConfig,
    candidates: list[Candidate],
) -> Classifier:
    """Construct a ``Classifier`` backed by LLM logits.

    Dispatches on ``config.backend``: ``"huggingface"``, ``"gguf"``, or
    ``"llama_server"``.  All return an ``LLMLogitClassifier`` with the
    shared ``ClassifierPromptBuilder`` (loaded from the classifier system
    prompt resource) and ``LogitScorer``.
    """
    from dynamic_prompt_core.infrastructure.llm.classifier import LLMLogitClassifier
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

    log.info(
        "Building classifier: backend=%s, model_path=%s",
        config.backend, config.model_path,
    )
    if config.backend == "gguf":
        tokenizer, model = _build_gguf_adapters(config)
    elif config.backend == "llama_server":
        tokenizer, model = _build_llama_server_adapters(config)
    else:
        tokenizer, model = _build_huggingface_adapters(config)

    return LLMLogitClassifier(
        prompt_builder=prompt_builder,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        logit_scorer=logit_scorer,
        candidates=candidates,
    )


def _build_huggingface_adapters(
    config: ScorerBackendConfig,
) -> tuple[object, object]:
    from transformers import AutoModelForCausalLM

    from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
        TorchBatchedCausalLMAdapter,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
        HuggingFaceBatchTokenizerAdapter,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(config.model_path)
    return (
        HuggingFaceBatchTokenizerAdapter(config.model_path),
        TorchBatchedCausalLMAdapter(hf_model, model_path=config.model_path),
    )


def _build_gguf_adapters(config: ScorerBackendConfig) -> tuple[object, object]:
    from dynamic_prompt_core.infrastructure.llm.scoring.llama_cpp_adapter import (
        LlamaCppBatchedCausalLMAdapter,
        LlamaCppBatchTokenizerAdapter,
    )

    assert config.gguf is not None
    model_adapter = LlamaCppBatchedCausalLMAdapter(
        model_path=config.model_path,
        n_ctx=config.gguf.n_ctx,
        n_batch=config.gguf.n_batch,
        n_threads=config.gguf.n_threads,
        n_gpu_layers=config.gguf.n_gpu_layers,
        seed=config.gguf.seed,
        verbose=config.gguf.verbose,
        logits_all=config.gguf.logits_all,
    )
    tokenizer_adapter = LlamaCppBatchTokenizerAdapter(model_adapter.llama)
    return tokenizer_adapter, model_adapter


def _build_llama_server_adapters(
    config: ScorerBackendConfig,
) -> tuple[object, object]:
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
    )
    tokenizer_adapter = LlamaServerBatchTokenizerAdapter(
        endpoint=ls.endpoint,
        timeout=ls.timeout,
    )
    return tokenizer_adapter, model_adapter
