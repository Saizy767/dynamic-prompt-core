"""
Baseline runner for the optimization cycle.

Runs the prepared dataset (dev or holdout) through the model with a fixed
extraction prompt and the current classification prompt version, collects
results with full schema, normalizes theses, checkpoints progress, and
persists a jsonl results artifact for the metrics and thesis-analysis
stages. Driven by the cycle via ``BaselineRunner.run``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.application.ports.outbound.classifier import Classifier
from dynamic_prompt_core.application.ports.outbound.dataset_repository import (
    DatasetRepository,
)
from dynamic_prompt_core.application.ports.outbound.extractor import Extractor
from dynamic_prompt_core.application.use_cases.run_baseline.run_metadata import (
    build_run_metadata,
    load_manifest,
    verify_resume_identity,
    write_manifest,
)
from dynamic_prompt_core.domain.errors.scoring import CandidateScoringError
from dynamic_prompt_core.domain.models.candidate import Candidate
from dynamic_prompt_core.domain.models.classification_rules import (
    classification_rules_from_artifact,
)
from dynamic_prompt_core.domain.models.dataset import Record
from dynamic_prompt_core.domain.prompts import (
    CLASSIFICATION_PROMPT_V0,
    PromptArtifact,
)

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
#  BaselineRunner
# --------------------------------------------------------------------------- #
class BaselineRunner:
    def __init__(
        self,
        classifier: Classifier,
        extractor: Extractor,
        dataset_repository: DatasetRepository,
        config: RunnerConfig,
        split: str,
        candidates: list[Candidate],
        run_id: str | None = None,
        dataset_artifact: str | None = None,
        duration: int | None = None,
        classify_prompt: PromptArtifact | None = None,
        backend_metadata: dict[str, Any] | None = None,
        skip_extraction: bool = False,
    ) -> None:
        self._classifier = classifier
        self._extractor = extractor
        self._dataset_repository = dataset_repository
        self._candidates = candidates
        self._config = config
        self._split = split
        self._run_id = run_id or f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        self._output_dir = config.output_dir
        self._concurrency = config.concurrency
        self._checkpoint_interval = config.checkpoint_interval
        self._timestamp: str | None = None
        self._backend_metadata = backend_metadata
        self._skip_extraction = skip_extraction

        if classify_prompt is not None:
            self._classify_prompt = classify_prompt
            self._prompt_version = classify_prompt.version
        else:
            self._classify_prompt = CLASSIFICATION_PROMPT_V0
            self._prompt_version = config.prompt_version

        self._rules = classification_rules_from_artifact(self._classify_prompt)

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
            "classify_ok": 0,
            "classify_failed": 0,
        }

    # -- paths -------------------------------------------------------------- #
    def _results_dir(self) -> str:
        return os.path.join(self._output_dir, "result_cycle")

    def _log_dir(self) -> str:
        return os.path.join(self._output_dir, "log_cycle")

    def _manifest_dir(self) -> str:
        return os.path.join(self._output_dir, "manifest_cycle")

    def _find_existing_checkpoint(self) -> str | None:
        """Glob for a .tmp checkpoint matching run_id + prompt_version + split."""
        import glob

        pattern = f"results_{self._run_id}_{self._prompt_version}_{self._split}_*.jsonl.tmp"
        matches = glob.glob(os.path.join(self._results_dir(), pattern))
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
        return os.path.join(self._results_dir(), name)

    def _artifact_path(self) -> str:
        name = (
            f"results_{self._run_id}_{self._prompt_version}_{self._split}_{self._timestamp}.jsonl"
        )
        return os.path.join(self._results_dir(), name)

    def _manifest_path(self) -> str:
        name = f"manifest_{self._run_id}_{self._prompt_version}_{self._split}.json"
        return os.path.join(self._manifest_dir(), name)

    def _resolve_backend_metadata(self) -> dict[str, Any] | None:
        backend_metadata = self._backend_metadata
        if backend_metadata is not None:
            return backend_metadata
        describe_fn = getattr(self._classifier, "describe", None)
        if describe_fn is not None:
            result = describe_fn()
            if isinstance(result, dict):
                return dict(result)
        return None

    def _dataset_fingerprint(self) -> str:
        with open(self._dataset_artifact, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    def _write_manifest(self, dataset_fingerprint: str | None) -> None:
        backend_metadata = self._resolve_backend_metadata()
        if backend_metadata is None:
            return
        prompt_sha256 = getattr(self._classify_prompt, "sha256", None)
        meta = build_run_metadata(
            run_id=self._run_id,
            prompt_version=self._prompt_version,
            backend_metadata=backend_metadata,
            prompt_sha256=prompt_sha256,
            split=self._split,
            dataset_path=self._dataset_artifact,
            dataset_fingerprint=dataset_fingerprint,
            timestamp=self._timestamp,
        )
        os.makedirs(self._manifest_dir(), exist_ok=True)
        write_manifest(self._manifest_path(), meta)

    def _verify_resume_identity(self, dataset_fingerprint: str | None = None) -> None:
        manifest_path = self._manifest_path()
        if not os.path.exists(manifest_path):
            return
        backend_metadata = self._resolve_backend_metadata()
        if backend_metadata is None:
            return
        recorded = load_manifest(manifest_path)
        prompt_sha256 = getattr(self._classify_prompt, "sha256", None)
        current = build_run_metadata(
            run_id=self._run_id,
            prompt_version=self._prompt_version,
            backend_metadata=backend_metadata,
            prompt_sha256=prompt_sha256,
            split=self._split,
            dataset_path=self._dataset_artifact,
            dataset_fingerprint=dataset_fingerprint,
            timestamp=self._timestamp,
        )
        result = verify_resume_identity(recorded, current)
        if result.should_reject:
            raise ValueError(
                f"resume identity mismatch in semantic fields "
                f"{result.mismatched_semantic_fields}; the recorded run used a "
                f"different model, dataset, or scoring configuration — reject "
                f"or start a new experiment"
            )

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
    def _update_stats(self, call_status: str | None, runner_status: str) -> None:
        if runner_status == _RUNNER_STATUS_OK:
            self._stats["ok"] += 1
        elif runner_status == _RUNNER_STATUS_REPAIRED:
            self._stats["repaired"] += 1
        elif call_status in ("network_error", "http_error"):
            self._stats["network_error"] += 1
        elif call_status == "timeout":
            self._stats["timeout"] += 1
        else:
            self._stats["parse_failed"] += 1

    # -- per-example -------------------------------------------------------- #
    async def _process_example(
        self,
        example: Record,
        index: int,
        results: list[ResultRow | None],
    ) -> None:
        assert self._semaphore is not None
        async with self._semaphore:
            # -- Classification (runs first) -- #
            classify_start = time.monotonic()
            if self._classify_prompt.layers is None:
                # shortcut: malformed prompt only; empty rules go to the
                # classifier as an honest zero-shot baseline.
                log.info(
                    "BaselineRunner: malformed prompt (layers is None) for id=%s, "
                    "setting predicted_decision=0 without classifier call",
                    example.id,
                )
                predicted_decision = 0
                selected_candidate = "0"
                classify_status = _RUNNER_STATUS_OK
                judgment_scores: dict[str, float] = {}
            else:
                try:
                    classification = await self._classifier.classify(
                        example.text,
                        self._rules,
                    )
                    classify_status = _RUNNER_STATUS_OK
                    selected_candidate = classification.selected.value
                    try:
                        predicted_decision = int(classification.selected.value)
                    except (ValueError, TypeError):
                        predicted_decision = None
                    judgment_scores = {j.candidate.value: j.score for j in classification.judgments}
                except CandidateScoringError as exc:
                    log.exception("classification failed for id=%s: %s", example.id, exc)
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
                self._stats["classify_ok"] += 1
            else:
                self._stats["classify_failed"] += 1

            # -- Extraction (runs after classification) -- #
            if self._skip_extraction:
                extract_result = self._extractor.skipped()
            else:
                extract_result = await self._extractor.extract(
                    example.text,
                    true_val=example.label,
                )
                self._update_stats(extract_result.call_status, extract_result.status)

            row = ResultRow(
                id=example.id,
                text=example.text,
                true_label=example.label,
                predicted_decision=predicted_decision,
                selected_candidate=selected_candidate,
                judgment_scores=judgment_scores,
                theses_raw=extract_result.theses_raw,
                theses_norm=extract_result.theses_norm,
                extract_status=extract_result.status,
                classify_status=classify_status,
                extract_latency_ms=extract_result.latency_ms,
                classify_latency_ms=classify_latency_ms,
                raw_extract=extract_result.raw_extract,
            )

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
        print(f"  classify_ok    : {s['classify_ok']}")
        print(f"  classify_failed: {s['classify_failed']}")
        print("=" * 60)

    # -- main run ----------------------------------------------------------- #
    async def run(self) -> tuple[list[ResultRow], str]:
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(self._results_dir(), exist_ok=True)
        os.makedirs(self._log_dir(), exist_ok=True)
        os.makedirs(self._manifest_dir(), exist_ok=True)
        self._init_timestamp()

        if not self._dataset_artifact:
            raise ValueError(
                "dataset_artifact path is required (config [runner].dataset_artifact "
                "or --dataset-artifact)"
            )

        dataset = self._dataset_repository.load_artifact(self._dataset_artifact)
        examples: list[Record] = getattr(dataset, self._split)
        if not examples:
            raise ValueError(f"split '{self._split}' has no examples")

        dataset_fingerprint = self._dataset_fingerprint()
        self._verify_resume_identity(dataset_fingerprint)
        self._write_manifest(dataset_fingerprint)

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

        try:
            await asyncio.gather(*(self._process_example(e, i, results) for i, e in to_run))
        finally:
            await self._extractor.aclose()

        artifact_path = self._finalize_artifact()
        self._print_summary(len(examples))
        print(f"Artifact: {artifact_path}")

        return [r for r in results if r is not None], artifact_path
