"""CLI composition root: parse args, load config, assemble dependencies, invoke use cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tomllib
from typing import Any

from dynamic_prompt_core.application.ports.inbound.run_cycle_input import RunCycleInput
from dynamic_prompt_core.application.use_cases.refine_theses.refine_theses_deps import (
    RefineThesesDeps,
)
from dynamic_prompt_core.application.use_cases.refine_theses.refiner import (
    parse_artifact_name,
    print_summary,
    refine_theses,
)
from dynamic_prompt_core.application.use_cases.refine_theses.result import (
    RefineThesesInput,
)
from dynamic_prompt_core.application.use_cases.run_cycle.config import CycleConfig
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle import RunCycleResult, run_cycle
from dynamic_prompt_core.application.use_cases.run_cycle.run_cycle_deps import RunCycleDeps
from dynamic_prompt_core.infrastructure.config import ConfigError, load_config

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic prompt optimization CLI.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.toml")
    sub = parser.add_subparsers(dest="command")

    cycle = sub.add_parser("cycle", help="Run the optimization cycle")
    cycle.add_argument("--split", default="dev", help="Dataset split to use")
    cycle.add_argument("--rounds", type=int, default=3, help="Number of optimization rounds")
    cycle.add_argument("--resume", default=None, help="Resume from a state dump JSON file")

    refine = sub.add_parser("refine", help="Run teacher thesis refinement")
    refine.add_argument(
        "--results",
        required=True,
        help="Path to results_*.jsonl artifact from stage1-baseline-runner",
    )
    refine.add_argument("--run-id", default=None, help="Override run_id")
    refine.add_argument("--prompt-version", default=None, help="Override prompt_version")

    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class _NormalizerAdapter:
    """Thin adapter wrapping infrastructure.nlp module-level functions."""

    def normalize_thesis(self, text: str, lang: str = "ru") -> str:
        from dynamic_prompt_core.infrastructure.nlp import normalize_thesis

        return normalize_thesis(text, lang)

    def normalize_theses(self, theses: list[str], lang: str = "ru") -> list[str]:
        from dynamic_prompt_core.infrastructure.nlp import normalize_theses

        return normalize_theses(theses, lang)


class _DatasetRepositoryAdapter:
    """Thin adapter for dataset loading via JSONL artifact files."""

    def load_dataset(self, path: str) -> list[Any]:
        from dynamic_prompt_core.domain.models.dataset import Record

        records: list[Any] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                records.append(Record(id=row["id"], text=row["text"], label=row["label"]))
        return records

    def load_artifact(self, path: str) -> Any:
        from dynamic_prompt_core.domain.models.dataset import Dataset, Record

        dataset = Dataset(seed=42, holdout_ratio=0.2, source_path=path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                record = Record(
                    id=row["id"],
                    text=row["text"],
                    label=row["label"],
                    ambiguous=row.get("ambiguous", False),
                    notes=row.get("notes", ""),
                )
                split = row.get("split", "dev")
                if split == "dev":
                    dataset.dev.append(record)
                elif split == "holdout":
                    dataset.holdout.append(record)
                elif split == "ambiguous":
                    dataset.ambiguous.append(record)
                else:
                    dataset.dev.append(record)
        return dataset

    def write_artifact(
        self,
        dataset: Any,
        output_dir: str = "data",
        fmt: str = "jsonl",
        notes: str = "",
    ) -> str:
        raise NotImplementedError("write_artifact not needed for run_cycle")


class _EmbeddingClientAdapter:
    """Thin adapter wrapping thesis_analyzer.compute_embedding."""

    def embed_one(self, text: str) -> list[float]:
        from dynamic_prompt_core.application.use_cases.analyze_theses.thesis_analyzer import (
            compute_embedding,
        )

        return compute_embedding(text, "all-MiniLM-L6-v2").tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def build_cli_deps(config: dict[str, Any], config_path: str) -> RunCycleDeps:
    """Assemble all dependencies for the run_cycle use case.

    This is the composition root: it creates concrete infrastructure
    implementations and passes them through the typed dependency object.
    """
    from dynamic_prompt_core.application.services.classification_policy import (
        ArgmaxClassificationPolicy,
    )
    from dynamic_prompt_core.domain.models.candidate import Candidate
    from dynamic_prompt_core.infrastructure.llm import AsyncTask
    from dynamic_prompt_core.infrastructure.llm.scoring.factory import (
        build_candidate_scorer,
    )
    from dynamic_prompt_core.infrastructure.storage import PromptStore, PromptStoreConfig
    from dynamic_prompt_core.infrastructure.storage.run_repository import (
        FileRunRepository,
    )

    llm = AsyncTask(config_path=config_path)
    store_config = PromptStoreConfig.from_config(config_path)
    store = PromptStore(store_config)
    run_repo = FileRunRepository()
    normalizer = _NormalizerAdapter()
    dataset_repo = _DatasetRepositoryAdapter()
    embedding_client = _EmbeddingClientAdapter()

    with open(config_path, "rb") as f:
        toml_config = tomllib.load(f)
    model_path = toml_config.get("llm", {}).get("model_path", "model")
    scorer = build_candidate_scorer(model_path)
    policy = ArgmaxClassificationPolicy()
    candidates = tuple(
        Candidate(v)
        for v in toml_config.get("classification", {}).get("candidates", ["0", "1"])
    )

    teacher_llm_client = None
    try:
        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)
        t = toml_config.get("teacher", {})
        if t.get("endpoint"):
            from dynamic_prompt_core.infrastructure.llm.teacher_client import TeacherClient

            teacher_llm_client = TeacherClient(
                endpoint=t["endpoint"],
                model_name=t.get("model_name", ""),
                temperature=float(t.get("temperature", 0.0)),
                max_tokens=int(t.get("max_tokens", 512)),
                timeout=int(t.get("timeout", 60)),
                max_retries=int(t.get("max_retries", 3)),
                config_path=config_path,
                filter_noisy=bool(t.get("filter_noisy", True)),
                filter_interpretive=bool(t.get("filter_interpretive", True)),
                allow_additions=bool(t.get("allow_additions", True)),
            )
    except Exception as exc:
        log.debug("teacher client not configured: %s", exc)

    stop_criteria_deps = None
    try:
        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)
        if "stop_criteria" in toml_config:
            from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.deps import (
                EvaluateStopCriteriaDeps,
            )

            stop_criteria_deps = EvaluateStopCriteriaDeps(run_repository=run_repo)
    except Exception as exc:
        log.debug("stop criteria not configured: %s", exc)

    return RunCycleDeps(
        llm_client=llm,  # type: ignore[arg-type]
        prompt_repository=store,
        run_repository=run_repo,
        dataset_repository=dataset_repo,
        embedding_client=embedding_client,
        normalizer=normalizer,
        candidate_scorer=scorer,
        classification_policy=policy,
        candidates=candidates,
        teacher_llm_client=teacher_llm_client,
        stop_criteria_deps=stop_criteria_deps,
    )


def build_refine_deps(config_path: str) -> RefineThesesDeps:
    """Assemble all dependencies for the refine_theses use case.

    Creates the TeacherClient and FileRunRepository in the composition root
    and passes them through the typed dependency object.
    """
    from dynamic_prompt_core.infrastructure.llm.teacher_client import TeacherClient
    from dynamic_prompt_core.infrastructure.storage.run_repository import (
        FileRunRepository,
    )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    t = config.get("teacher", {})
    endpoint = t.get("endpoint", "")
    if not endpoint:
        raise ConfigError("Missing required config key: teacher.endpoint")
    model_name = t.get("model_name", "")
    if not model_name:
        raise ConfigError("Missing required config key: teacher.model_name")

    teacher_client = TeacherClient(
        endpoint=endpoint,
        model_name=model_name,
        temperature=float(t.get("temperature", 0.0)),
        max_tokens=int(t.get("max_tokens", 512)),
        timeout=int(t.get("timeout", 60)),
        max_retries=int(t.get("max_retries", 3)),
        config_path=config_path,
        filter_noisy=bool(t.get("filter_noisy", True)),
        filter_interpretive=bool(t.get("filter_interpretive", True)),
        allow_additions=bool(t.get("allow_additions", True)),
    )
    run_repo = FileRunRepository()

    return RefineThesesDeps(
        teacher_llm_client=teacher_client,
        run_repository=run_repo,
    )


async def run_refine(args: argparse.Namespace, config_path: str) -> int:
    deps = build_refine_deps(config_path)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    t = config.get("teacher", {})

    default_run_id, default_prompt_version = parse_artifact_name(args.results)
    run_id = args.run_id or default_run_id
    prompt_version = args.prompt_version or default_prompt_version

    inp = RefineThesesInput(
        results_path=args.results,
        run_id=run_id,
        prompt_version=prompt_version,
        output_dir=t.get("output_dir", "data/results"),
        log_path=t.get("log_path", "data/thesis_refiner.jsonl"),
        filter_noisy=bool(t.get("filter_noisy", True)),
        filter_interpretive=bool(t.get("filter_interpretive", True)),
        allow_additions=bool(t.get("allow_additions", True)),
    )

    try:
        result = await refine_theses(deps, inp)
        print_summary(result)
        print(f"Artifact: {result.artifact_path}")
        return 0
    except Exception:
        log.exception("Unrecoverable error during refinement.")
        return 1


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    config_path = args.config

    if args.command == "refine":
        return await run_refine(args, config_path)

    try:
        cfg = load_config(config_path)
        cfg["config_path"] = config_path
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    deps = build_cli_deps(cfg, config_path)
    rounds = getattr(args, "rounds", None) or 3
    split = getattr(args, "split", None) or "dev"
    resume_from = getattr(args, "resume", None)

    cycle_config = CycleConfig.from_toml(config_path)
    if rounds != cycle_config.max_rounds:
        d = cycle_config.duration
        duration_val = d[0] if isinstance(d, list) and d else d
        cycle_config = type(cycle_config)(
            **{**cycle_config.__dict__, "max_rounds": rounds, "duration": duration_val}
        )

    try:
        from dynamic_prompt_core.application.use_cases.evaluate_stop_criteria.config import (
            StopCriteriaConfig,
        )

        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)
        if "stop_criteria" in toml_config:
            stop_criteria_config = StopCriteriaConfig.from_toml(config_path)
            cycle_config = type(cycle_config)(
                **{
                    **cycle_config.__dict__,
                    "stop_criteria_config": stop_criteria_config,
                }
            )
    except Exception as exc:
        log.debug("stop criteria config not loaded: %s", exc)

    run_input = RunCycleInput(
        config_path=config_path,
        max_rounds=rounds,
        dev_split=split,
        resume_from=resume_from,
        config=cycle_config,
    )

    try:
        result: RunCycleResult = await run_cycle(deps, run_input)
        log.info(
            "Cycle complete: %d rounds, stop=%s, final=%s",
            result.total_rounds,
            result.stop_reason,
            result.final_active_version,
        )
        return 0
    except Exception:
        log.exception("Unrecoverable error during cycle execution.")
        return 1


def cli_entry() -> None:
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli_entry()
