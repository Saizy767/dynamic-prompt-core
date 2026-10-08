"""
Stage 1 baseline runner.

Runs the prepared dataset (dev or holdout) through the model with a fixed
extraction prompt and the current classification prompt version, collects
results with full schema, normalizes theses, checkpoints progress, and
persists a jsonl results artifact for the metrics and thesis-analysis stages.

Usage:
    python runner.py --split dev
    python runner.py --split holdout --run-id myrun
    python runner.py --split dev --duration 1000 --cycles 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp

from dynamic_prompt_core.application.ports.outbound.candidate_scorer import CandidateScorer
from dynamic_prompt_core.application.schemas import ThesisExtraction
from dynamic_prompt_core.application.services.classification_policy import ClassificationPolicy
from dynamic_prompt_core.application.services.metrics import metrics as metrics_module
from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.dataset import Record
from dynamic_prompt_core.domain.prompts import (
    CLASSIFICATION_PROMPT_V0,
    EXTRACTION_PROMPT,
    PromptArtifact,
)
from dynamic_prompt_core.infrastructure.data.writer import load_artifact
from dynamic_prompt_core.infrastructure.llm import (
    AsyncTask,
    CallResult,
    ResponseStatus,
)
from dynamic_prompt_core.infrastructure.nlp import normalize_theses

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_CONCURRENCY = 4
DEFAULT_CHECKPOINT_INTERVAL = 1
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_LANGUAGE = "ru"
DEFAULT_PROMPT_VERSION = "classify-v0"
DEFAULT_CANDIDATES: list[str] = ["0", "1"]

_RUNNER_STATUS_OK = "ok"
_RUNNER_STATUS_REPAIRED = "repaired"
_RUNNER_STATUS_FAILED = "failed"
_RUNNER_STATUS_NOT_ATTEMPTED = "not_attempted"


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class RunnerConfig:
    concurrency: int = DEFAULT_CONCURRENCY
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    output_dir: str = DEFAULT_OUTPUT_DIR
    language: str = DEFAULT_LANGUAGE
    prompt_version: str = DEFAULT_PROMPT_VERSION
    dataset_artifact: str = ""
    candidates: list[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATES))

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_CONFIG_PATH) -> RunnerConfig:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        r = config.get("runner", {})
        c = config.get("classification", {})
        return cls(
            concurrency=int(r.get("concurrency", DEFAULT_CONCURRENCY)),
            checkpoint_interval=int(r.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)),
            output_dir=r.get("output_dir", DEFAULT_OUTPUT_DIR),
            language=r.get("language", DEFAULT_LANGUAGE),
            prompt_version=r.get("prompt_version", DEFAULT_PROMPT_VERSION),
            dataset_artifact=r.get("dataset_artifact", ""),
            candidates=list(c.get("candidates", DEFAULT_CANDIDATES)),
        )


# --------------------------------------------------------------------------- #
#  Result row
# --------------------------------------------------------------------------- #
@dataclass
class ResultRow:
    id: Any
    text: str
    true_label: int
    predicted_decision: int | None = None
    selected_candidate: str | None = None
    judgment_scores: dict[str, float] = field(default_factory=dict)
    theses_raw: list[str] = field(default_factory=list)
    theses_norm: list[str] = field(default_factory=list)
    extract_status: str = _RUNNER_STATUS_NOT_ATTEMPTED
    classify_status: str = _RUNNER_STATUS_NOT_ATTEMPTED
    extract_latency_ms: float = 0.0
    classify_latency_ms: float = 0.0
    raw_extract: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "true_label": self.true_label,
            "predicted_decision": self.predicted_decision,
            "selected_candidate": self.selected_candidate,
            "judgment_scores": self.judgment_scores,
            "theses_raw": self.theses_raw,
            "theses_norm": self.theses_norm,
            "extract_status": self.extract_status,
            "classify_status": self.classify_status,
            "extract_latency_ms": self.extract_latency_ms,
            "classify_latency_ms": self.classify_latency_ms,
            "raw_extract": self.raw_extract,
        }


# --------------------------------------------------------------------------- #
#  Status mapping + repair
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _try_repair(raw_content: str | None, model_cls: type) -> Any | None:
    """Attempt to repair a malformed response by stripping markdown fences."""
    if not raw_content:
        return None
    stripped = raw_content.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        candidate = m.group(1).strip()
        try:
            return model_cls.model_validate_json(candidate)  # type: ignore[attr-defined]
        except Exception:
            pass
    return None


def _map_call_status(call_result: CallResult, model_cls: type) -> tuple[str, Any | None]:
    """Map a CallResult to a runner status and parsed model (with repair)."""
    if call_result.parsed is not None:
        return _RUNNER_STATUS_OK, call_result.parsed
    repaired = _try_repair(call_result.raw_content, model_cls)
    if repaired is not None:
        return _RUNNER_STATUS_REPAIRED, repaired
    return _RUNNER_STATUS_FAILED, None


# --------------------------------------------------------------------------- #
#  BaselineRunner
# --------------------------------------------------------------------------- #
class BaselineRunner:
    def __init__(
        self,
        task: AsyncTask,
        config: RunnerConfig,
        split: str,
        scorer: CandidateScorer,
        policy: ClassificationPolicy,
        candidates: list[Candidate],
        run_id: str | None = None,
        dataset_artifact: str | None = None,
        duration: int | None = None,
        classify_prompt: PromptArtifact | None = None,
    ) -> None:
        self._task = task
        self._scorer = scorer
        self._policy = policy
        self._candidates = candidates
        self._config = config
        self._split = split
        self._run_id = run_id or f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        self._language = config.language
        self._output_dir = config.output_dir
        self._concurrency = config.concurrency
        self._checkpoint_interval = config.checkpoint_interval
        self._timestamp: str | None = None

        if classify_prompt is not None:
            self._classify_prompt = classify_prompt
            self._prompt_version = classify_prompt.version
        else:
            self._classify_prompt = CLASSIFICATION_PROMPT_V0
            self._prompt_version = config.prompt_version
        self._extract_prompt = EXTRACTION_PROMPT

        self._dataset_artifact = dataset_artifact or config.dataset_artifact
        self._duration = duration

        self._semaphore: asyncio.Semaphore | None = None
        self._ckpt_lock = asyncio.Lock()
        self._ckpt_count = 0
        self._stats: dict[str, int] = {
            "ok": 0,
            "repaired": 0,
            "parse_failed": 0,
            "network_error": 0,
            "timeout": 0,
        }

    # -- paths -------------------------------------------------------------- #
    def _find_existing_checkpoint(self) -> str | None:
        """Glob for a .tmp checkpoint matching run_id + prompt_version + split."""
        import glob

        pattern = f"results_{self._run_id}_{self._prompt_version}_{self._split}_*.jsonl.tmp"
        matches = glob.glob(os.path.join(self._output_dir, pattern))
        return matches[0] if matches else None

    def _init_timestamp(self) -> None:
        """Set timestamp from an existing checkpoint, or generate a new one."""
        existing = self._find_existing_checkpoint()
        if existing:
            basename = os.path.basename(existing)
            prefix = f"results_{self._run_id}_{self._prompt_version}_{self._split}_"
            suffix = ".jsonl.tmp"
            if basename.startswith(prefix) and basename.endswith(suffix):
                self._timestamp = basename[len(prefix) : -len(suffix)]
            else:
                self._timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        else:
            self._timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    def _checkpoint_path(self) -> str:
        name = (
            f"results_{self._run_id}_{self._prompt_version}_"
            f"{self._split}_{self._timestamp}.jsonl.tmp"
        )
        return os.path.join(self._output_dir, name)

    def _artifact_path(self) -> str:
        name = (
            f"results_{self._run_id}_{self._prompt_version}_{self._split}_{self._timestamp}.jsonl"
        )
        return os.path.join(self._output_dir, name)

    def _request_log_path(self) -> str:
        name = f"requests_{self._run_id}_{self._split}.jsonl"
        return os.path.join(self._output_dir, name)

    # -- checkpoint --------------------------------------------------------- #
    def _load_checkpoint_ids(self) -> set[Any]:
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return set()
        ids: set[Any] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    ids.add(row["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids

    async def _checkpoint_write(self, row: ResultRow) -> None:
        line = json.dumps(row.to_dict(), ensure_ascii=False) + "\n"
        async with self._ckpt_lock:
            with open(self._checkpoint_path(), "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                self._ckpt_count += 1
                if self._ckpt_count % self._checkpoint_interval == 0:
                    os.fsync(f.fileno())

    def _finalize_artifact(self) -> str:
        tmp = self._checkpoint_path()
        final = self._artifact_path()
        if os.path.exists(tmp):
            os.replace(tmp, final)
        return final

    # -- stats -------------------------------------------------------------- #
    def _update_stats(self, call_result: CallResult, runner_status: str) -> None:
        if runner_status == _RUNNER_STATUS_OK:
            self._stats["ok"] += 1
        elif runner_status == _RUNNER_STATUS_REPAIRED:
            self._stats["repaired"] += 1
        elif call_result.status in (
            ResponseStatus.NETWORK_ERROR,
            ResponseStatus.HTTP_ERROR,
        ):
            self._stats["network_error"] += 1
        elif call_result.status == ResponseStatus.TIMEOUT:
            self._stats["timeout"] += 1
        else:
            self._stats["parse_failed"] += 1

    # -- per-example -------------------------------------------------------- #
    async def _process_example(
        self,
        session: aiohttp.ClientSession,
        example: Record,
        index: int,
        results: list[ResultRow | None],
    ) -> None:
        assert self._semaphore is not None
        async with self._semaphore:
            try:
                extract_cr = await self._task.extract_theses_detailed(
                    session,
                    example.text,
                    ThesisExtraction,
                    system_prompt=self._extract_prompt.text,
                    true_val=example.label,
                )
            except Exception as exc:
                log.exception("extract_theses crashed for id=%s: %s", example.id, exc)
                extract_cr = CallResult(
                    parsed=None,
                    raw_content=None,
                    latency_ms=0.0,
                    parse_status=None,
                    status=ResponseStatus.UNEXPECTED_SHAPE,
                    error=str(exc),
                )

            extract_status, extract_parsed = _map_call_status(extract_cr, ThesisExtraction)

            theses_raw = list(extract_parsed.theses) if extract_parsed else []
            theses_norm = normalize_theses(theses_raw, self._language) if theses_raw else []

            classify_start = time.monotonic()
            try:
                judgments = await self._scorer.score(example.text, self._candidates)
                classification = self._policy.classify(judgments)
                classify_status = _RUNNER_STATUS_OK
                selected_candidate = classification.selected.value
                try:
                    predicted_decision = int(classification.selected.value)
                except (ValueError, TypeError):
                    predicted_decision = None
                judgment_scores = {j.candidate.value: j.score for j in classification.judgments}
            except CandidateScoringError as exc:
                log.exception("candidate scoring failed for id=%s: %s", example.id, exc)
                classify_status = _RUNNER_STATUS_FAILED
                predicted_decision = None
                selected_candidate = None
                judgment_scores = {}
            except Exception as exc:
                log.exception("classify crashed for id=%s: %s", example.id, exc)
                classify_status = _RUNNER_STATUS_FAILED
                predicted_decision = None
                selected_candidate = None
                judgment_scores = {}
            classify_latency_ms = (time.monotonic() - classify_start) * 1000

            if classify_status == _RUNNER_STATUS_OK:
                self._stats["ok"] += 1
            else:
                self._stats["parse_failed"] += 1

            row = ResultRow(
                id=example.id,
                text=example.text,
                true_label=example.label,
                predicted_decision=predicted_decision,
                selected_candidate=selected_candidate,
                judgment_scores=judgment_scores,
                theses_raw=theses_raw,
                theses_norm=theses_norm,
                extract_status=extract_status,
                classify_status=classify_status,
                extract_latency_ms=extract_cr.latency_ms,
                classify_latency_ms=classify_latency_ms,
                raw_extract=extract_cr.raw_content,
            )

            self._update_stats(extract_cr, extract_status)

            results[index] = row
            await self._checkpoint_write(row)

    # -- summary ------------------------------------------------------------ #
    def _print_summary(self, total: int) -> None:
        s = self._stats
        print("\n" + "=" * 60)
        print(f"Run summary  (run_id={self._run_id}, split={self._split})")
        print("=" * 60)
        print(f"  Total examples : {total}")
        print(f"  ok             : {s['ok']}")
        print(f"  repaired       : {s['repaired']}")
        print(f"  parse_failed   : {s['parse_failed']}")
        print(f"  network_error  : {s['network_error']}")
        print(f"  timeout        : {s['timeout']}")
        print("=" * 60)

    # -- main run ----------------------------------------------------------- #
    async def run(self) -> tuple[list[ResultRow], str]:
        os.makedirs(self._output_dir, exist_ok=True)
        self._init_timestamp()

        if not self._dataset_artifact:
            raise ValueError(
                "dataset_artifact path is required (config [runner].dataset_artifact "
                "or --dataset-artifact)"
            )

        dataset = load_artifact(self._dataset_artifact)
        examples: list[Record] = getattr(dataset, self._split)
        if not examples:
            raise ValueError(f"split '{self._split}' has no examples")

        if self._duration is not None:
            if self._duration <= 0:
                raise ValueError(f"--duration must be a positive integer, got {self._duration}")
            if self._duration > len(examples):
                raise ValueError(
                    f"--duration ({self._duration}) exceeds split '{self._split}' "
                    f"size ({len(examples)})"
                )
            examples = examples[: self._duration]
            print(
                f"Duration: processing {len(examples)} of "
                f"{len(getattr(dataset, self._split))} examples"
            )

        processed_ids = self._load_checkpoint_ids()
        if processed_ids:
            print(
                f"Resuming: {len(processed_ids)} examples already in checkpoint, "
                f"{len(examples) - len([e for e in examples if e.id in processed_ids])} remaining."
            )

        to_run = [(i, e) for i, e in enumerate(examples) if e.id not in processed_ids]
        if not to_run:
            print("All examples already processed. Finalizing artifact.")

        results: list[ResultRow | None] = [None] * len(examples)
        self._semaphore = asyncio.Semaphore(self._concurrency)

        connector = aiohttp.TCPConnector(limit=self._concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            await asyncio.gather(
                *(self._process_example(session, e, i, results) for i, e in to_run)
            )

        artifact_path = self._finalize_artifact()
        self._print_summary(len(examples))
        print(f"Artifact: {artifact_path}")
        print(f"Request log: {self._request_log_path()}")

        return [r for r in results if r is not None], artifact_path


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
async def _main_async(args: argparse.Namespace) -> None:
    config = RunnerConfig.from_config(args.config)

    pass  # logging configured by composition root

    run_id = args.run_id
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not run_id:
        run_id = f"run-{timestamp}"

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config, "rb") as f:
        llm_config = tomllib.load(f).get("llm", {})
    model_path = llm_config.get("model_path", "model")

    from dynamic_prompt_core.application.services.classification_policy import (
        ArgmaxClassificationPolicy,
    )
    from dynamic_prompt_core.infrastructure.llm.scoring.factory import (
        build_candidate_scorer,
    )

    scorer = build_candidate_scorer(model_path)
    policy = ArgmaxClassificationPolicy()
    candidates = [Candidate(v) for v in config.candidates]

    cycles = args.cycles
    for cycle in range(1, cycles + 1):
        cycle_run_id = run_id if cycles == 1 else f"{run_id}-c{cycle}"
        if cycles > 1:
            print(f"\n{'=' * 60}")
            print(f"Cycle {cycle}/{cycles}  (run_id={cycle_run_id})")
            print(f"{'=' * 60}")

        log_path = os.path.join(output_dir, f"requests_{cycle_run_id}_{args.split}.jsonl")

        task = AsyncTask(
            config_path=args.config,
            endpoint=args.endpoint,
            log_path=log_path,
            run_id=cycle_run_id,
        )

        runner = BaselineRunner(
            task=task,
            config=config,
            split=args.split,
            scorer=scorer,
            policy=policy,
            candidates=candidates,
            run_id=cycle_run_id,
            dataset_artifact=args.dataset_artifact,
            duration=args.duration,
        )

        _, artifact_path = await runner.run()

        # Auto-compute metrics from the cycle's results artifact
        try:
            rows = metrics_module.load_results(artifact_path)
            m_config = metrics_module.MetricsConfig.from_config(args.config)
            m = metrics_module.compute_metrics(rows, m_config, artifact_path)
            metrics_path = metrics_module.write_metrics(m, m_config.output_dir)
            print(f"\nMetrics artifact: {metrics_path}")
            print(
                f"  accuracy={m['accuracy']:.4f}  macro-F1={m['f1']['macro']:.4f}"
                f"  minority-F1={m['f1']['minority']:.4f}  biased={m['bias']['biased']}"
            )
        except Exception as exc:
            log.warning("Metrics computation failed for %s: %s", artifact_path, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 baseline runner")
    parser.add_argument("--split", choices=["dev", "holdout"], required=True)
    parser.add_argument("--run-id", default=None, help="Run identifier (for resume)")
    parser.add_argument(
        "--dataset-artifact",
        default=None,
        help="Path to the prepared dataset artifact (jsonl/parquet)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080/v1",
        help="LLM server endpoint",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Number of examples from the split to process (must be <= split size)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of passes over the selected examples (each cycle gets its own artifact)",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
