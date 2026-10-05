"""
Stage 3 prompt composer.

Loads a rule-candidate artifact, formulates each candidate into a single-sentence
rule via an LLM, verifies semantic fidelity against the source cluster centroid,
assembles a new classification prompt version from the unchanged base layers plus
accepted rules, validates hard limits, and persists a reloadable prompt-version
artifact with full lineage back to source clusters.

Usage:
    python prompt_composer.py --artifact data/results/rule_candidates_*.json
    python prompt_composer.py --artifact data/results/rule_candidates_*.json \
        --thesis-artifact data/results/thesis_bank_*.json \
        --config config.toml --endpoint http://127.0.0.1:8080/v1
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prompts import PromptArtifact, PromptLayer, build_classification_prompt, render
from schemas import RuleFormulation
from thesis_analyzer import cosine_similarity, compute_embedding

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_MAX_RULES = 5
DEFAULT_MAX_PROMPT_TOKENS = 300
DEFAULT_MAX_RULE_WORDS = 15
DEFAULT_DISTORTION_THRESHOLD = 0.75
DEFAULT_ALLOW_REFORMULATION = True
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_FORMULATION_MAX_TOKENS = 64
DEFAULT_FORMULATION_TEMPERATURE = 0.0
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"

REQUIRED_CANDIDATE_FIELDS = (
    "cluster_id",
    "representative_theses",
    "precision",
    "frequency",
    "positive_hits",
    "negative_hits",
    "rank",
)

REQUIRED_BASE_LAYERS = ("role", "task", "output_contract", "fallback")


class PromptComposerError(ValueError):
    """Raised when an artifact is invalid or composition fails."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class PromptComposerConfig:
    max_rules: int = DEFAULT_MAX_RULES
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    max_rule_words: int = DEFAULT_MAX_RULE_WORDS
    distortion_threshold: float = DEFAULT_DISTORTION_THRESHOLD
    allow_reformulation: bool = DEFAULT_ALLOW_REFORMULATION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    formulation_max_tokens: int = DEFAULT_FORMULATION_MAX_TOKENS
    formulation_temperature: float = DEFAULT_FORMULATION_TEMPERATURE
    output_dir: str = DEFAULT_OUTPUT_DIR

    @classmethod
    def from_config(
        cls, config_path: str = DEFAULT_CONFIG_PATH
    ) -> "PromptComposerConfig":
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("prompt_composer", {})
        return cls(
            max_rules=int(s.get("max_rules", DEFAULT_MAX_RULES)),
            max_prompt_tokens=int(
                s.get("max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)
            ),
            max_rule_words=int(s.get("max_rule_words", DEFAULT_MAX_RULE_WORDS)),
            distortion_threshold=float(
                s.get("distortion_threshold", DEFAULT_DISTORTION_THRESHOLD)
            ),
            allow_reformulation=bool(
                s.get("allow_reformulation", DEFAULT_ALLOW_REFORMULATION)
            ),
            embedding_model=s.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            formulation_max_tokens=int(
                s.get("formulation_max_tokens", DEFAULT_FORMULATION_MAX_TOKENS)
            ),
            formulation_temperature=float(
                s.get("formulation_temperature", DEFAULT_FORMULATION_TEMPERATURE)
            ),
            output_dir=s.get("output_dir", DEFAULT_OUTPUT_DIR),
        )


# --------------------------------------------------------------------------- #
#  Artifact loading and validation
# --------------------------------------------------------------------------- #
def _validate_candidate(candidate: Dict[str, Any]) -> None:
    cid = candidate.get("cluster_id")
    for fname in REQUIRED_CANDIDATE_FIELDS:
        if fname not in candidate or candidate[fname] is None:
            raise PromptComposerError(
                f"candidate cluster_id={cid!r} is missing required field '{fname}'"
            )


def load_candidates(path: str) -> Dict[str, Any]:
    """Load and schema-validate a rule-candidate artifact (JSON with candidates).

    Returns a dict with 'metadata' and 'candidates' keys.
    Raises PromptComposerError naming the first offending candidate's cluster_id
    and the missing field when a candidate does not conform.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "candidates" not in data:
        raise PromptComposerError(
            "artifact is missing top-level 'candidates' key"
        )

    for candidate in data["candidates"]:
        _validate_candidate(candidate)

    return data


def load_base_layers(
    prompt_artifact: PromptArtifact,
) -> Dict[str, str]:
    """Extract the four base layers from a PromptArtifact.

    Returns a dict with 'role', 'task', 'output_contract', 'fallback'.
    Raises PromptComposerError naming the missing layer when layers is None
    or any base layer is an empty string.
    """
    layers = prompt_artifact.layers
    if layers is None:
        raise PromptComposerError(
            "active prompt version has no layers (layers is None)"
        )

    base: Dict[str, str] = {
        "role": layers.role,
        "task": layers.task,
        "output_contract": layers.output_contract,
        "fallback": layers.fallback,
    }
    for name, value in base.items():
        if not isinstance(value, str) or value == "":
            raise PromptComposerError(
                f"base layer '{name}' is missing or empty"
            )
    return base


def load_centroids(
    thesis_artifact_path: Optional[str],
) -> Dict[int, np.ndarray]:
    """Load cluster centroids from a thesis-bank dump artifact.

    Returns a {cluster_id: centroid_ndarray} map. Returns an empty map when
    the path is None.
    """
    if thesis_artifact_path is None:
        return {}

    with open(thesis_artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "clusters" not in data:
        raise PromptComposerError(
            "thesis artifact is missing top-level 'clusters' key"
        )

    centroids: Dict[int, np.ndarray] = {}
    for cid_str, cluster in data["clusters"].items():
        centroid = cluster.get("centroid")
        if centroid is not None:
            centroids[int(cid_str)] = np.array(centroid, dtype=np.float32)
    return centroids


# --------------------------------------------------------------------------- #
#  Rule formulation via LLM
# --------------------------------------------------------------------------- #
_FORMULATION_PROMPT = (
    "You are a prompt-engineering assistant. You read representative theses for "
    "a semantic cluster of texts, along with the cluster's precision and "
    "frequency, and formulate a single concise classification rule that tells a "
    "binary classifier when to assign class 1 based on that cluster.\n\n"
    "The rule must be a single sentence, at most 15 words, in the same language "
    "as the theses. Do not add explanations, examples, or numbering.\n\n"
    "Reply with a single JSON object: {\"rule\": \"...\"}."
)

_REFORMULATION_INSTRUCTION = (
    "\n\nThe previous rule drifted from the source cluster. Reformulate it so it "
    "stays closer to the meaning of the representative theses above, while "
    "remaining a single sentence of at most 15 words."
)


def _build_user_message(candidate: Dict[str, Any]) -> str:
    """Build the LLM user message from a candidate's theses and stats."""
    theses = candidate.get("representative_theses", [])
    thesis_lines: List[str] = []
    for t in theses:
        text = t.get("text_raw", t.get("text_norm", ""))
        if text:
            thesis_lines.append(f"- {text}")
    theses_text = "\n".join(thesis_lines) if thesis_lines else "(none)"
    return (
        f"Representative theses:\n{theses_text}\n\n"
        f"precision={candidate['precision']:.4f}\n"
        f"frequency={candidate['frequency']}\n\n"
        f"Formulate a single-sentence classification rule for class 1."
    )


async def formulate_rule(
    candidate: Dict[str, Any],
    config: PromptComposerConfig,
    async_task: Any,
) -> Optional[str]:
    """Formulate a single-sentence rule from a candidate via an LLM call.

    Returns the rule string, or None when the LLM call fails or returns no
    parseable result.
    """
    import aiohttp

    user_message = _build_user_message(candidate)
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        result = await async_task.analyze(
            session,
            user_message,
            RuleFormulation,
            call_type="formulate_rule",
            system_prompt=_FORMULATION_PROMPT,
            max_tokens=config.formulation_max_tokens,
        )
    if result is None:
        return None
    return result.rule


async def reformulate_rule(
    candidate: Dict[str, Any],
    previous_rule: str,
    config: PromptComposerConfig,
    async_task: Any,
) -> Optional[str]:
    """Reformulate a rule that failed the distortion check.

    Appends a faithfulness instruction referencing the previous rule and makes
    a second LLM call. Returns the new rule string, or None on failure.
    """
    import aiohttp

    user_message = (
        _build_user_message(candidate)
        + f"\n\nPrevious rule: \"{previous_rule}\""
        + _REFORMULATION_INSTRUCTION
    )
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        result = await async_task.analyze(
            session,
            user_message,
            RuleFormulation,
            call_type="reformulate_rule",
            system_prompt=_FORMULATION_PROMPT,
            max_tokens=config.formulation_max_tokens,
        )
    if result is None:
        return None
    return result.rule


def enforce_rule_length(
    rule: str, max_words: int
) -> Tuple[str, bool]:
    """Truncate a rule to max_words words.

    Returns (truncated_rule, was_truncated).
    """
    words = rule.split()
    if len(words) <= max_words:
        return rule, False
    truncated = " ".join(words[:max_words])
    log.warning(
        "rule truncated from %d to %d words: %r -> %r",
        len(words), max_words, rule, truncated,
    )
    return truncated, True


# --------------------------------------------------------------------------- #
#  Semantic distortion check
# --------------------------------------------------------------------------- #
def compute_rule_embedding(
    rule: str, model_name: str
) -> np.ndarray:
    """Compute a float32 embedding vector for a formulated rule.

    Reuses the lazy-loaded sentence-transformers model from thesis_analyzer.py.
    """
    return compute_embedding(rule, model_name)


def check_distortion(
    rule: str,
    centroid: Optional[np.ndarray],
    config: PromptComposerConfig,
    model_name: str,
) -> Tuple[bool, Optional[float]]:
    """Check a formulated rule for semantic distortion.

    Compares the rule embedding against the source cluster centroid using cosine
    similarity. Returns (passed, similarity).

    When centroid is None, returns (True, None) and logs a WARNING that the
    safety net was bypassed.
    """
    if centroid is None:
        log.warning(
            "distortion check skipped (no centroid available) for rule: %r",
            rule,
        )
        return True, None

    rule_embedding = compute_rule_embedding(rule, model_name)
    similarity = cosine_similarity(rule_embedding, centroid)
    passed = similarity >= config.distortion_threshold
    if not passed:
        log.info(
            "distortion check failed: similarity=%.4f < threshold=%.4f for rule: %r",
            similarity, config.distortion_threshold, rule,
        )
    return passed, similarity


# --------------------------------------------------------------------------- #
#  Prompt assembly and validation
# --------------------------------------------------------------------------- #
_VERSION_RE = re.compile(r"^(.+)-v(\d+)$")


def increment_version(base_version: str) -> str:
    """Increment the trailing integer of a classify-vN version.

    classify-v0 -> classify-v1, classify-v1 -> classify-v2.
    When the pattern does not match, appends '-next'.
    """
    match = _VERSION_RE.match(base_version)
    if match:
        prefix = match.group(1)
        n = int(match.group(2))
        return f"{prefix}-v{n + 1}"
    return f"{base_version}-next"


def assemble_prompt(
    base_layers: Dict[str, str],
    accepted_rules: List[str],
    base_version: str,
) -> PromptArtifact:
    """Assemble a new prompt version from base layers plus accepted rules.

    The rules section is replaced entirely by accepted_rules. The layer order
    is role, task, rules, output contract, fallback. The base layers are carried
    over unchanged.
    """
    layers = PromptLayer(
        role=base_layers["role"],
        task=base_layers["task"],
        rules=list(accepted_rules),
        output_contract=base_layers["output_contract"],
        fallback=base_layers["fallback"],
    )
    new_version = increment_version(base_version)
    text = render(layers)
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return PromptArtifact(
        version=new_version,
        layers=layers,
        text=text,
        sha256=sha256,
    )


def _count_tokens(text: str, tokenizer: Any) -> Tuple[int, bool]:
    """Count tokens in text. Returns (count, used_word_fallback)."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text)), False
        except Exception:
            pass
    log.warning(
        "tokenizer unavailable; falling back to word count for token limit check"
    )
    return len(text.split()), True


def validate_limits(
    prompt_artifact: PromptArtifact,
    config: PromptComposerConfig,
    tokenizer: Any = None,
) -> None:
    """Validate the assembled prompt against hard limits.

    Raises PromptComposerError with a message naming the violated limit.
    """
    layers = prompt_artifact.layers
    if layers is None:
        raise PromptComposerError("prompt has no layers")

    rule_count = len(layers.rules)
    if rule_count > config.max_rules:
        raise PromptComposerError(
            f"too many rules: {rule_count} > {config.max_rules}"
        )

    token_count, _ = _count_tokens(prompt_artifact.text, tokenizer)
    if token_count > config.max_prompt_tokens:
        raise PromptComposerError(
            f"prompt too long: {token_count} > {config.max_prompt_tokens}"
        )

    if not layers.output_contract:
        raise PromptComposerError("output contract is empty")
    if not layers.fallback:
        raise PromptComposerError("fallback is empty")


# --------------------------------------------------------------------------- #
#  Composition orchestration
# --------------------------------------------------------------------------- #
async def compose(
    candidates: List[Dict[str, Any]],
    base_layers: Dict[str, str],
    centroids: Dict[int, np.ndarray],
    config: PromptComposerConfig,
    async_task: Any,
    base_version: str,
) -> Tuple[PromptArtifact, List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """Run the full composition pipeline over all candidates.

    Chains formulation, length enforcement, distortion check, optional
    reformulation, and assembly. Returns (prompt_artifact, rules_with_lineage,
    counters, rejected_rules) where each rule carries cluster_id and text,
    counters has candidates_in, rules_formulated, rejected_distortion,
    rejected_limits, final_rules_count, and rejected_rules is a list of
    {cluster_id, text, cosine, threshold, reason} for each candidate that did
    not become an accepted rule (reason is one of distortion,
    empty_formulation).
    """
    candidates_in = len(candidates)
    rules_formulated = 0
    rejected_distortion = 0

    accepted_rules: List[str] = []
    rules_with_lineage: List[Dict[str, Any]] = []
    rejected_rules: List[Dict[str, Any]] = []

    for candidate in candidates:
        cluster_id = candidate["cluster_id"]

        rule = await formulate_rule(candidate, config, async_task)
        if rule is None or rule.strip() == "":
            log.warning(
                "formulation returned empty rule for cluster_id=%s; skipping",
                cluster_id,
            )
            rejected_distortion += 1
            rejected_rules.append({
                "cluster_id": cluster_id,
                "text": None,
                "cosine": None,
                "threshold": config.distortion_threshold,
                "reason": "empty_formulation",
            })
            continue
        rules_formulated += 1

        rule, was_truncated = enforce_rule_length(rule, config.max_rule_words)

        centroid = centroids.get(cluster_id)
        passed, similarity = check_distortion(
            rule, centroid, config, config.embedding_model
        )

        if not passed and config.allow_reformulation:
            log.info(
                "attempting reformulation for cluster_id=%s (similarity=%.4f)",
                cluster_id,
                similarity if similarity is not None else 0.0,
            )
            new_rule = await reformulate_rule(
                candidate, rule, config, async_task
            )
            if new_rule is not None and new_rule.strip() != "":
                new_rule, _ = enforce_rule_length(
                    new_rule, config.max_rule_words
                )
                passed, similarity = check_distortion(
                    new_rule, centroid, config, config.embedding_model
                )
                if passed:
                    rule = new_rule

        if not passed:
            log.warning(
                "rule rejected for distortion: cluster_id=%s similarity=%.4f",
                cluster_id,
                similarity if similarity is not None else 0.0,
            )
            rejected_distortion += 1
            rejected_rules.append({
                "cluster_id": cluster_id,
                "text": rule,
                "cosine": similarity,
                "threshold": config.distortion_threshold,
                "reason": "distortion",
            })
            continue

        accepted_rules.append(rule)
        rules_with_lineage.append({"cluster_id": cluster_id, "text": rule})

    prompt_artifact = assemble_prompt(
        base_layers, accepted_rules, base_version
    )

    rejected_limits = 0
    final_rules_count = len(accepted_rules)

    counters = {
        "candidates_in": candidates_in,
        "rules_formulated": rules_formulated,
        "rejected_distortion": rejected_distortion,
        "rejected_limits": rejected_limits,
        "final_rules_count": final_rules_count,
    }

    return prompt_artifact, rules_with_lineage, counters, rejected_rules


# --------------------------------------------------------------------------- #
#  Prompt version artifact persistence
# --------------------------------------------------------------------------- #
def write_prompt_version(
    prompt_artifact: PromptArtifact,
    rules_with_lineage: List[Dict[str, Any]],
    source_candidates: List[int],
    base_version: str,
    run_id: str,
    counters: Dict[str, int],
    config: PromptComposerConfig,
    output_dir: str,
    rejected_rules: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Serialize the new prompt version to a JSON artifact.

    Filename: prompt_v{version}_{run_id}_{timestamp}.json
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = prompt_artifact.version
    name = f"prompt_v{version}_{run_id}_{timestamp}.json"
    path = os.path.join(output_dir, name)

    dump = {
        "version": version,
        "text": prompt_artifact.text,
        "hash": prompt_artifact.sha256,
        "rules": rules_with_lineage,
        "source_candidates": source_candidates,
        "base_version": base_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "run_id": run_id,
            "timestamp": timestamp,
            "config": {
                "max_rules": config.max_rules,
                "max_prompt_tokens": config.max_prompt_tokens,
                "max_rule_words": config.max_rule_words,
                "distortion_threshold": config.distortion_threshold,
                "allow_reformulation": config.allow_reformulation,
                "embedding_model": config.embedding_model,
            },
            "counters": counters,
            "rejected_rules": rejected_rules if rejected_rules is not None else [],
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    return path


def load_prompt_version(path: str) -> Dict[str, Any]:
    """Restore a prompt version from a prompt_v*_*_*.json artifact.

    Returns a dict with version, text, hash, rules, source_candidates,
    base_version, created_at, and metadata. No composition is re-run. The
    metadata.rejected_rules field defaults to an empty list when absent (older
    artifacts).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    if "rejected_rules" not in meta:
        meta["rejected_rules"] = []
        data["metadata"] = meta
    return data


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _print_summary(
    prompt_artifact: PromptArtifact,
    counters: Dict[str, int],
    run_id: str,
    base_version: str,
) -> None:
    print("\n" + "=" * 60)
    print(
        f"Prompt version  (version={prompt_artifact.version}, "
        f"base={base_version}, run_id={run_id})"
    )
    print("=" * 60)
    print(
        f"  candidates_in      : {counters['candidates_in']}\n"
        f"  rules_formulated   : {counters['rules_formulated']}\n"
        f"  rejected_distortion : {counters['rejected_distortion']}\n"
        f"  rejected_limits    : {counters['rejected_limits']}\n"
        f"  final_rules_count  : {counters['final_rules_count']}"
    )
    print("-" * 60)
    print(f"  hash: {prompt_artifact.sha256}")
    print(f"  tokens (words): {len(prompt_artifact.text.split())}")
    print("=" * 60)


async def _run_composition(
    candidates: List[Dict[str, Any]],
    base_layers: Dict[str, str],
    centroids: Dict[int, np.ndarray],
    config: PromptComposerConfig,
    base_version: str,
    endpoint: str,
    config_path: str,
) -> Tuple[PromptArtifact, List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """Create an AsyncTask and run the composition pipeline."""
    from asyncTask import AsyncTask

    async_task = AsyncTask(
        config_path=config_path,
        endpoint=endpoint,
        max_tokens=config.formulation_max_tokens,
        temperature=config.formulation_temperature,
    )
    return await compose(
        candidates, base_layers, centroids, config, async_task, base_version
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 prompt composer"
    )
    parser.add_argument(
        "--artifact", required=True,
        help="Path to rule-candidate artifact (JSON with candidates)",
    )
    parser.add_argument(
        "--thesis-artifact", default=None,
        help="Path to thesis-bank dump (JSON with cluster centroids)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from prompts import CLASSIFICATION_PROMPT_V0

    config = PromptComposerConfig.from_config(args.config)
    candidate_artifact = load_candidates(args.artifact)
    run_id = candidate_artifact.get("metadata", {}).get("run_id", "unknown")

    base_layers = load_base_layers(CLASSIFICATION_PROMPT_V0)
    base_version = CLASSIFICATION_PROMPT_V0.version

    centroids = load_centroids(args.thesis_artifact)

    candidates = candidate_artifact["candidates"]

    prompt_artifact, rules_with_lineage, counters, rejected_rules = asyncio.run(
        _run_composition(
            candidates,
            base_layers,
            centroids,
            config,
            base_version,
            args.endpoint,
            args.config,
        )
    )

    tokenizer = None
    try:
        from asyncTask import AsyncTask
        async_task = AsyncTask(
            config_path=args.config, endpoint=args.endpoint
        )
        tokenizer = async_task._tokenizer
    except Exception:
        pass

    try:
        validate_limits(prompt_artifact, config, tokenizer)
    except PromptComposerError as exc:
        counters["rejected_limits"] = 1
        log.error("limit validation failed: %s", exc)
        raise

    source_candidates = [r["cluster_id"] for r in rules_with_lineage]

    log.info(
        "composition: candidates_in=%d rules_formulated=%d "
        "rejected_distortion=%d rejected_limits=%d final_rules_count=%d",
        counters["candidates_in"],
        counters["rules_formulated"],
        counters["rejected_distortion"],
        counters["rejected_limits"],
        counters["final_rules_count"],
    )

    artifact_path = write_prompt_version(
        prompt_artifact,
        rules_with_lineage,
        source_candidates,
        base_version,
        run_id,
        counters,
        config,
        config.output_dir,
        rejected_rules,
    )
    _print_summary(prompt_artifact, counters, run_id, base_version)
    print(f"Prompt version: {artifact_path}")


if __name__ == "__main__":
    main()
