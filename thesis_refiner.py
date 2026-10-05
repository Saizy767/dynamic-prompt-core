"""
Stage 4 teacher refinement (thesis_refiner.py).

Refines Tiny-model thesis candidates using a larger teacher model over a
separate endpoint. Loads theses from a stage1-baseline-runner results artifact,
asks the teacher model to filter noisy/interpretive theses, reformulate
unstable phrasings, and add missed theses, then writes a reloadable
theses_refined_*.jsonl artifact. Original Tiny-model theses are preserved
verbatim; per-example teacher failures never abort the run.

Quick start
-----------
    python thesis_refiner.py --results data/results/results_run-*.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from asyncTask import AsyncTask, RawResponse, ResponseStatus

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_USE_REFINED = True
DEFAULT_FILTER_NOISY = True
DEFAULT_FILTER_INTERPRETIVE = True
DEFAULT_ALLOW_ADDITIONS = True
DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_LOG_PATH = "data/thesis_refiner.jsonl"

REQUIRED_CANDIDATE_FIELDS = ("id", "theses_raw", "theses_norm", "extract_status")
_EXTRACT_STATUS_FAILED = "failed"


class ThesisRefinerError(ValueError):
    """Raised when refinement fails or an artifact is invalid."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class TeacherRefinerConfig:
    endpoint: str = ""
    model_name: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    use_refined: bool = DEFAULT_USE_REFINED
    filter_noisy: bool = DEFAULT_FILTER_NOISY
    filter_interpretive: bool = DEFAULT_FILTER_INTERPRETIVE
    allow_additions: bool = DEFAULT_ALLOW_ADDITIONS
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_path: str = DEFAULT_LOG_PATH

    @classmethod
    def from_config(
        cls, config_path: str = DEFAULT_CONFIG_PATH
    ) -> "TeacherRefinerConfig":
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        t = config.get("teacher", {})
        endpoint = t.get("endpoint", "")
        if not endpoint:
            raise ThesisRefinerError(
                "Missing required config key: teacher.endpoint"
            )
        model_name = t.get("model_name", "")
        if not model_name:
            raise ThesisRefinerError(
                "Missing required config key: teacher.model_name"
            )
        return cls(
            endpoint=endpoint,
            model_name=model_name,
            temperature=float(t.get("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=int(t.get("max_tokens", DEFAULT_MAX_TOKENS)),
            timeout=int(t.get("timeout", DEFAULT_TIMEOUT)),
            max_retries=int(t.get("max_retries", DEFAULT_MAX_RETRIES)),
            use_refined=bool(t.get("use_refined", DEFAULT_USE_REFINED)),
            filter_noisy=bool(t.get("filter_noisy", DEFAULT_FILTER_NOISY)),
            filter_interpretive=bool(
                t.get("filter_interpretive", DEFAULT_FILTER_INTERPRETIVE)
            ),
            allow_additions=bool(
                t.get("allow_additions", DEFAULT_ALLOW_ADDITIONS)
            ),
            output_dir=t.get("output_dir", DEFAULT_OUTPUT_DIR),
            log_path=t.get("log_path", DEFAULT_LOG_PATH),
        )


# --------------------------------------------------------------------------- #
#  Teacher response schema
# --------------------------------------------------------------------------- #
class ReformulateItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: str = Field(alias="from")
    to: str


class DropItem(BaseModel):
    thesis: str
    reason: str


class RefinementPlan(BaseModel):
    keep: List[str] = field(default_factory=list)
    reformulate: List[ReformulateItem] = field(default_factory=list)
    drop: List[DropItem] = field(default_factory=list)
    add: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Teacher system prompt
# --------------------------------------------------------------------------- #
def build_teacher_prompt(config: TeacherRefinerConfig) -> str:
    """Build the system prompt for the teacher model.

    Conditionally includes noisy-filter, interpretive-filter, and addition
    instructions based on the config flags.
    """
    parts: List[str] = [
        "You are a thesis refinement assistant.",
        "You receive a text and a list of candidate theses extracted by a"
        " smaller model.",
        "Return a JSON object with four arrays:",
        '- "keep": theses to preserve unchanged (extractive, informative).',
        '- "reformulate": objects {"from": <original>, "to": <new>} for theses'
        " that need a more stable extractive phrasing. The reformulated text"
        " must preserve meaning and not exceed the original length.",
        '- "drop": objects {"thesis": <text>, "reason": <why>} for theses to'
        " exclude.",
    ]
    if config.filter_noisy:
        parts.append(
            "Drop noisy theses: generic phrases applicable to any text that"
            " carry no signal for classification."
        )
    if config.filter_interpretive:
        parts.append(
            "Drop or reformulate interpretive theses (conclusions about the"
            " text) into extractive form (features present in the text)."
        )
    if config.allow_additions:
        parts.append(
            '- "add": missed theses present in the text and relevant for'
            " classification, as plain strings."
        )
    else:
        parts.append(
            'Do not add any new theses: "add" must be an empty array.'
        )
    parts.append(
        "Every input thesis must appear in exactly one of keep, reformulate, or"
        " drop. Return only the JSON object."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
#  Candidate loading
# --------------------------------------------------------------------------- #
def load_candidates(path: str) -> List[Dict[str, Any]]:
    """Load and validate a results_*.jsonl artifact.

    Each row must have id, theses_raw, theses_norm, and extract_status.
    Raises ThesisRefinerError naming the first offending record's id and the
    missing field.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id", "?")
            for field_name in REQUIRED_CANDIDATE_FIELDS:
                if field_name not in row:
                    raise ThesisRefinerError(
                        f"Record id={row_id} is missing required field"
                        f" '{field_name}' in {path}"
                    )
            rows.append(row)
    return rows


def split_candidates(
    rows: List[Dict[str, Any]],
    log_fn: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into (to_process, skipped).

    Rows with extract_status == 'failed' are skipped. The skip is logged via
    log_fn(operation, id, details) when provided.
    """
    to_process: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("extract_status") == _EXTRACT_STATUS_FAILED:
            skipped.append(row)
            if log_fn is not None:
                log_fn("skip", row["id"], {"reason": "extract_status=failed"})
        else:
            to_process.append(row)
    return to_process, skipped


# --------------------------------------------------------------------------- #
#  Teacher transport
# --------------------------------------------------------------------------- #
def _build_teacher_task(
    config: TeacherRefinerConfig,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> AsyncTask:
    """Construct an AsyncTask pointed at the teacher endpoint.

    Overrides _served_model_name so the teacher endpoint receives the correct
    model field (design D2).
    """
    task = AsyncTask(
        config_path=config_path,
        endpoint=config.endpoint,
        completion_timeout=config.timeout,
        max_retries=config.max_retries,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        truncate_tokens=2000,
    )
    task._served_model_name = config.model_name
    return task


async def _call_teacher(
    task: AsyncTask,
    session: aiohttp.ClientSession,
    text: str,
    theses: List[str],
    config: TeacherRefinerConfig,
) -> Tuple[RefinementPlan, float, Optional[dict]]:
    """Send one teacher call for an example and parse the RefinementPlan.

    Returns (plan, latency_ms, usage). Raises on any failure (network, parse,
    schema mismatch) so the caller can apply failure isolation (design D7).
    """
    system_prompt = build_teacher_prompt(config)
    user_message = (
        f"Text:\n{text}\n\nCandidate theses:\n"
        + "\n".join(f"- {t}" for t in theses)
    )
    raw: RawResponse = await task.analyze_raw(
        session,
        user_message,
        RefinementPlan,
        system_prompt=system_prompt,
        max_tokens=config.max_tokens,
        truncate_tokens=2000,
    )
    if raw.status != ResponseStatus.OK or not raw.content:
        raise ThesisRefinerError(
            f"Teacher call failed with status {raw.status.value}"
            + (f": {raw.error}" if raw.error else "")
        )
    try:
        plan = RefinementPlan.model_validate_json(raw.content)
    except ValidationError as exc:
        raise ThesisRefinerError(
            f"Teacher response did not match RefinementPlan schema: {exc}"
        ) from exc
    return plan, raw.latency_ms, raw.usage


# --------------------------------------------------------------------------- #
#  Refinement logic
# --------------------------------------------------------------------------- #
def _apply_refinement(
    theses_raw: List[str],
    plan: RefinementPlan,
    config: TeacherRefinerConfig,
    log_fn: Optional[Any] = None,
    example_id: Any = None,
) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, str]]]:
    """Build (theses_refined, filtered_out, added) from a RefinementPlan.

    theses_refined order: keep, then reformulated, then added (design D4).
    """
    theses_refined: List[str] = []
    filtered_out: List[Dict[str, str]] = []
    added: List[Dict[str, str]] = []

    # keep
    for t in plan.keep:
        theses_refined.append(t)

    # reformulate (with length enforcement, design D5)
    for item in plan.reformulate:
        if len(item.to) > len(item.from_):
            theses_refined.append(item.from_)
            if log_fn is not None:
                log_fn(
                    "reformulate",
                    example_id,
                    {"from": item.from_, "to": item.to, "kept_original": True,
                     "reason": "reformulation longer than original"},
                )
        else:
            theses_refined.append(item.to)
            if log_fn is not None:
                log_fn(
                    "reformulate",
                    example_id,
                    {"from": item.from_, "to": item.to, "kept_original": False},
                )

    # drop (noisy / interpretive filtering)
    for item in plan.drop:
        if config.filter_noisy:
            filtered_out.append({"thesis": item.thesis, "reason": item.reason})
            if log_fn is not None:
                log_fn(
                    "filter",
                    example_id,
                    {"thesis": item.thesis, "reason": item.reason},
                )
        else:
            theses_refined.append(item.thesis)

    # add
    if config.allow_additions:
        for t in plan.add:
            theses_refined.append(t)
            added.append({"text": t, "source": "teacher"})
            if log_fn is not None:
                log_fn("add", example_id, {"text": t, "source": "teacher"})

    return theses_refined, filtered_out, added


# --------------------------------------------------------------------------- #
#  Refined artifact write and reload
# --------------------------------------------------------------------------- #
def write_refined_artifact(
    records: List[Dict[str, Any]],
    run_id: str,
    prompt_version: str,
    output_dir: str,
) -> str:
    """Write refined records to a jsonl artifact.

    Filename: theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl"
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_refined(path: str) -> List[Dict[str, Any]]:
    """Load a theses_refined_*.jsonl artifact back into records.

    No teacher calls are made. Raises ThesisRefinerError naming the file on a
    corrupted line.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ThesisRefinerError(
                    f"Corrupted refined artifact {path}: {exc}"
                ) from exc
            records.append(rec)
    return records


# --------------------------------------------------------------------------- #
#  Refiner
# --------------------------------------------------------------------------- #
@dataclass
class RefinerTotals:
    processed: int = 0
    teacher_errors: int = 0
    filtered: int = 0
    reformulated: int = 0
    added: int = 0
    skipped: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0


class ThesisRefiner:
    """Drives the per-example refinement loop with failure isolation."""

    def __init__(
        self,
        config: TeacherRefinerConfig,
        task: AsyncTask,
        log_path: str = DEFAULT_LOG_PATH,
    ) -> None:
        self._config = config
        self._task = task
        self._log_path = log_path
        self._totals = RefinerTotals()

    # -- logging ----------------------------------------------------------- #
    def _log(
        self, operation: str, example_id: Any, details: Dict[str, Any]
    ) -> None:
        """Append one JSON log line. Best-effort: never blocks refinement."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "id": example_id,
            "details": details,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(
                f"WARNING: failed to write refinement log: {exc}",
                file=sys.stderr,
            )

    # -- per-example ------------------------------------------------------- #
    async def _refine_one(
        self,
        session: aiohttp.ClientSession,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Refine a single example. Never raises (design D7)."""
        example_id = row["id"]
        theses_raw_tiny: List[str] = list(row.get("theses_raw", []))
        text: str = row.get("text", "")

        try:
            plan, latency_ms, usage = await _call_teacher(
                self._task, session, text, theses_raw_tiny, self._config
            )
        except Exception as exc:
            self._totals.teacher_errors += 1
            self._log("teacher_error", example_id, {"error": str(exc)})
            return {
                "id": example_id,
                "theses_raw_tiny": theses_raw_tiny,
                "theses_refined": theses_raw_tiny,
                "filtered_out": [],
                "added": [],
            }

        # per-call accounting (design D8)
        prompt_tokens = 0
        completion_tokens = 0
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
        self._totals.total_prompt_tokens += prompt_tokens
        self._totals.total_completion_tokens += completion_tokens
        self._totals.total_latency_ms += latency_ms
        self._log(
            "teacher_call",
            example_id,
            {
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "status": "ok",
            },
        )

        theses_refined, filtered_out, added = _apply_refinement(
            theses_raw_tiny,
            plan,
            self._config,
            log_fn=self._log,
            example_id=example_id,
        )

        self._totals.processed += 1
        self._totals.filtered += len(filtered_out)
        self._totals.reformulated += sum(
            1 for item in plan.reformulate
        )
        self._totals.added += len(added)

        return {
            "id": example_id,
            "theses_raw_tiny": theses_raw_tiny,
            "theses_refined": theses_refined,
            "filtered_out": filtered_out,
            "added": added,
        }

    # -- run --------------------------------------------------------------- #
    async def run(
        self, rows: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], RefinerTotals]:
        """Refine all candidates. Returns (records, totals)."""
        to_process, skipped = split_candidates(rows, log_fn=self._log)
        self._totals.skipped = len(skipped)

        records: List[Dict[str, Any]] = []
        connector = aiohttp.TCPConnector(limit=4)
        async with aiohttp.ClientSession(connector=connector) as session:
            for row in to_process:
                rec = await self._refine_one(session, row)
                records.append(rec)

        # skipped examples keep their originals
        for row in skipped:
            records.append({
                "id": row["id"],
                "theses_raw_tiny": list(row.get("theses_raw", [])),
                "theses_refined": list(row.get("theses_raw", [])),
                "filtered_out": [],
                "added": [],
            })

        return records, self._totals

    # -- summary ----------------------------------------------------------- #
    def print_summary(self, totals: RefinerTotals) -> None:
        """Print the end-of-run summary with all counters and costs."""
        print(
            f"Refinement summary"
            f"  (processed={totals.processed},"
            f" teacher_errors={totals.teacher_errors},"
            f" filtered={totals.filtered},"
            f" reformulated={totals.reformulated},"
            f" added={totals.added},"
            f" skipped={totals.skipped})"
        )
        print(
            f"Cost  (prompt_tokens={totals.total_prompt_tokens},"
            f" completion_tokens={totals.total_completion_tokens},"
            f" total_latency_ms={totals.total_latency_ms:.1f})"
        )


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _parse_artifact_name(path: str) -> Tuple[str, str]:
    """Best-effort extraction of run_id and prompt_version from a results path."""
    base = os.path.basename(path)
    parts = base.replace(".jsonl", "").split("_")
    if len(parts) >= 4 and parts[0] == "results":
        return parts[1], parts[2]
    return "unknown", "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 teacher refinement"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results_*.jsonl artifact from stage1-baseline-runner",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-id", default=None, help="Override run_id")
    parser.add_argument(
        "--prompt-version", default=None, help="Override prompt_version"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = TeacherRefinerConfig.from_config(args.config)
    task = _build_teacher_task(config, args.config)

    rows = load_candidates(args.results)
    default_run_id, default_prompt_version = _parse_artifact_name(args.results)
    run_id = args.run_id or default_run_id
    prompt_version = args.prompt_version or default_prompt_version

    refiner = ThesisRefiner(config, task, log_path=config.log_path)
    records, totals = asyncio.run(refiner.run(rows))

    artifact_path = write_refined_artifact(
        records, run_id, prompt_version, config.output_dir
    )
    refiner._log(
        "write_artifact", None, {"path": artifact_path, "records": len(records)}
    )
    refiner.print_summary(totals)
    print(f"Artifact: {artifact_path}")


if __name__ == "__main__":
    main()
