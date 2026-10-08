"""Composition-root factory for the candidate scorer.

Constructs a ``LLMLogitCandidateScorer`` with all infrastructure dependencies
(prompt builder, tokenizer, model adapter, logit scorer) from a model path.
Callers receive the result through the ``CandidateScorer`` port; this module
may import ``transformers`` and infrastructure scoring modules.
"""
from __future__ import annotations

from dynamic_prompt_core.application.ports.outbound.candidate_scorer import (
    CandidateScorer,
)


def build_candidate_scorer(model_path: str) -> CandidateScorer:
    """Construct a ``CandidateScorer`` backed by LLM logits.

    Loads the model via ``AutoModelForCausalLM.from_pretrained`` and wires
    the scoring infrastructure components.  The result satisfies the
    ``CandidateScorer`` port contract.
    """
    from transformers import AutoModelForCausalLM

    from dynamic_prompt_core.infrastructure.llm.scoring.candidate_scorer import (
        LLMLogitCandidateScorer,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.logit_scorer import (
        LogitScorer,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.model_adapter import (
        TorchBatchedCausalLMAdapter,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.prompt_builder import (
        ScoringPromptBuilder,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.tokenizer_adapter import (
        HuggingFaceBatchTokenizerAdapter,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    return LLMLogitCandidateScorer(
        prompt_builder=ScoringPromptBuilder(),
        tokenizer=HuggingFaceBatchTokenizerAdapter(model_path),
        model=TorchBatchedCausalLMAdapter(hf_model),
        logit_scorer=LogitScorer(),
    )
