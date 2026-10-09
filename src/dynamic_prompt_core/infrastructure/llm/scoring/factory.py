"""Composition-root factory for the candidate scorer.

Constructs a ``LLMLogitCandidateScorer`` with all infrastructure dependencies
(prompt builder, tokenizer, model adapter, logit scorer) from a
``ScorerBackendConfig``.  Callers receive the result through the
``CandidateScorer`` port; this module may import ``transformers`` and
infrastructure scoring modules.
"""
from __future__ import annotations

import logging

from dynamic_prompt_core.application.ports.outbound.candidate_scorer import (
    CandidateScorer,
)
from dynamic_prompt_core.infrastructure.llm.scoring.config import ScorerBackendConfig

log = logging.getLogger(__name__)


def build_candidate_scorer(config: ScorerBackendConfig) -> CandidateScorer:
    """Construct a ``CandidateScorer`` backed by LLM logits.

    Dispatches on ``config.backend``: ``"huggingface"`` loads the model via
    ``AutoModelForCausalLM.from_pretrained`` and wires the Hugging Face
    adapters; ``"gguf"`` lazy-imports the GGUF adapters and loads the model
    via ``llama-cpp-python``.  Both return an ``LLMLogitCandidateScorer`` with
    the shared ``ScoringPromptBuilder`` and ``LogitScorer``.  The result
    satisfies the ``CandidateScorer`` port contract.
    """
    from dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer import (
        LLMLogitCandidateScorer,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
        LogitScorer,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
        ScoringPromptBuilder,
    )

    prompt_builder = ScoringPromptBuilder()
    logit_scorer = LogitScorer()

    log.info(
        "Building candidate scorer: backend=%s, model_path=%s",
        config.backend, config.model_path,
    )
    if config.backend == "gguf":
        tokenizer, model = _build_gguf_adapters(config)
    else:
        tokenizer, model = _build_huggingface_adapters(config)

    return LLMLogitCandidateScorer(
        prompt_builder=prompt_builder,
        tokenizer=tokenizer,
        model=model,
        logit_scorer=logit_scorer,
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
