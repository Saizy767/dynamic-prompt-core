"""
Stage 4 teacher refinement use case.

Refines Tiny-model thesis candidates using a larger teacher model over a
separate endpoint. Loads theses from a stage1-baseline-runner results artifact,
asks the teacher model to filter noisy/interpretive theses, reformulate
unstable phrasings, and add missed theses, then writes a reloadable
theses_refined_*.jsonl artifact. Original Tiny-model theses are preserved
verbatim; per-example teacher failures never abort the run.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

from dynamic_prompt_core.application.use_cases.refine_theses.refine_theses_deps import (
    RefineThesesDeps,
)
from dynamic_prompt_core.application.use_cases.refine_theses.result import (
    RefineThesesInput,
    RefineThesesResult,
)
from dynamic_prompt_core.domain.services.thesis_refinement import (
    DropEntry,
    RefinementPlanData,
    ReformulatePair,
    apply_refinement,
)

log = logging.getLogger(__name__)

REQUIRED_CANDIDATE_FIELDS = ("id", "theses_raw", "theses_norm", "extract_status")
_EXTRACT_STATUS_FAILED = "failed"


class ThesisRefinerError(ValueError):
    """Raised when refinement fails or an artifact is invalid."""


# --------------------------------------------------------------------------- #
#  Candidate loading
# --------------------------------------------------------------------------- #
def load_candidates(path: str) -> list[dict[str, Any]]:
    """Load and validate a results_*.jsonl artifact.

    Each row must have id, theses_raw, theses_norm, and extract_status.
    Raises ThesisRefinerError naming the first offending record's id and the
    missing field.
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id", "?")
            for field_name in REQUIRED_CANDIDATE_FIELDS:
                if field_name not in row:
                    raise ThesisRefinerError(
                        f"Record id={row_id} is missing required field '{field_name}' in {path}"
                    )
            rows.append(row)
    return rows


def split_candidates(
    rows: list[dict[str, Any]],
    log_fn: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into (to_process, skipped).

    Rows with extract_status == 'failed' are skipped. The skip is logged via
    log_fn(operation, id, details) when provided.
    """
    to_process: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if row.get("extract_status") == _EXTRACT_STATUS_FAILED:
            skipped.append(row)
            if log_fn is not None:
                log_fn("skip", row["id"], {"reason": "extract_status=failed"})
        else:
            to_process.append(row)
    return to_process, skipped


# --------------------------------------------------------------------------- #
#  Refined artifact write and reload
# --------------------------------------------------------------------------- #
def write_refined_artifact(
    records: list[dict[str, Any]],
    run_id: str,
    prompt_version: str,
    output_dir: str,
) -> str:
    """Write refined records to a jsonl artifact.

    Filename: theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"theses_refined_{run_id}_{prompt_version}_{timestamp}.jsonl"
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_refined(path: str) -> list[dict[str, Any]]:
    """Load a theses_refined_*.jsonl artifact back into records.

    No teacher calls are made. Raises ThesisRefinerError naming the file on a
    corrupted line.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ThesisRefinerError(f"Corrupted refined artifact {path}: {exc}") from exc
            records.append(rec)
    return records


# --------------------------------------------------------------------------- #
#  Logging helper
# --------------------------------------------------------------------------- #
def _append_log(log_path: str, operation: str, example_id: Any, details: dict[str, Any]) -> None:
    """Append one JSON log line. Best-effort: never blocks refinement."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "id": example_id,
        "details": details,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(
            f"WARNING: failed to write refinement log: {exc}",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
#  Use case
# --------------------------------------------------------------------------- #
async def refine_theses(
    deps: RefineThesesDeps,
    inp: RefineThesesInput,
) -> RefineThesesResult:
    """Run the teacher refinement use case.

    Loads candidates from the results artifact, calls the teacher model via
    the port for each example, applies the refinement plan via the domain
    service, and writes the refined' artifact. Per-example teacher failures
    are isolated: the example keeps its original theses and processing
    continues.
    """
    rows = load_candidates(inp.results_path)
    _append_log(inp.log_path, "load_candidates", None, {"count": len(rows), "path": inp.results_path})

    to_process: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if row.get("extract_status") == _EXTRACT_STATUS_FAILED:
            skipped.append(row)
            _append_log(inp.log_path, "skip", row["id"], {"reason": "extract_status=failed"})
        else:
            to_process.append(row)

    records: list[dict[str, Any]] = []
    processed = 0
    teacher_errors = 0
    filtered = 0
    reformulated = 0
    added = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0

    for row in to_process:
        example_id = row["id"]
        theses_raw_tiny: list[str] = list(row.get("theses_raw", []))
        text: str = row.get("text", "")

        try:
            review = await deps.teacher_llm_client.review_theses(text, theses_raw_tiny)
        except Exception as exc:
            teacher_errors += 1
            _append_log(inp.log_path, "teacher_error", example_id, {"error": str(exc)})
            records.append({
                "id": example_id,
                "theses_raw_tiny": theses_raw_tiny,
                "theses_refined": theses_raw_tiny,
                "filtered_out": [],
                "added": [],
            })
            continue

        prompt_tokens = 0
        completion_tokens = 0
        if review.usage:
            prompt_tokens = int(review.usage.get("prompt_tokens", 0))
            completion_tokens = int(review.usage.get("completion_tokens", 0))
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_latency_ms += review.latency_ms
        _append_log(
            inp.log_path,
            "teacher_call",
            example_id,
            {
                "latency_ms": review.latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "status": "ok",
            },
        )

        plan = RefinementPlanData(
            keep=list(review.keep),
            reformulate=[
                ReformulatePair(from_=item["from"], to=item["to"])
                for item in review.reformulate
            ],
            drop=[
                DropEntry(thesis=item["thesis"], reason=item["reason"])
                for item in review.drop
            ],
            add=list(review.add),
        )

        theses_refined, filtered_out, added_list = apply_refinement(
            theses_raw_tiny,
            plan,
            filter_noisy=inp.filter_noisy,
            filter_interpretive=inp.filter_interpretive,
            allow_additions=inp.allow_additions,
        )

        for item in review.reformulate:
            from_val = item["from"]
            to_val = item["to"]
            kept_original = len(to_val) > len(from_val)
            _append_log(
                inp.log_path,
                "reformulate",
                example_id,
                {
                    "from": from_val,
                    "to": to_val,
                    "kept_original": kept_original,
                },
            )

        for drop_item in review.drop:
            _append_log(
                inp.log_path,
                "filter",
                example_id,
                {"thesis": drop_item["thesis"], "reason": drop_item["reason"]},
            )

        for add_item in review.add:
            _append_log(
                inp.log_path,
                "add",
                example_id,
                {"text": add_item, "source": "teacher"},
            )

        processed += 1
        filtered += len(filtered_out)
        reformulated += len(review.reformulate)
        added += len(added_list)

        records.append({
            "id": example_id,
            "theses_raw_tiny": theses_raw_tiny,
            "theses_refined": theses_refined,
            "filtered_out": filtered_out,
            "added": added_list,
        })

    for row in skipped:
        records.append({
            "id": row["id"],
            "theses_raw_tiny": list(row.get("theses_raw", [])),
            "theses_refined": list(row.get("theses_raw", [])),
            "filtered_out": [],
            "added": [],
        })

    artifact_path = write_refined_artifact(
        records, inp.run_id, inp.prompt_version, inp.output_dir,
    )
    _append_log(inp.log_path, "write_artifact", None, {"path": artifact_path, "records": len(records)})

    return RefineThesesResult(
        processed=processed,
        teacher_errors=teacher_errors,
        filtered=filtered,
        reformulated=reformulated,
        added=added,
        skipped=len(skipped),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_latency_ms=total_latency_ms,
        artifact_path=artifact_path,
    )


def print_summary(result: RefineThesesResult) -> None:
    """Print the end-of-run summary with all counters and costs."""
    print(
        f"Refinement summary"
        f"  (processed={result.processed},"
        f" teacher_errors={result.teacher_errors},"
        f" filtered={result.filtered},"
        f" reformulated={result.reformulated},"
        f" added={result.added},"
        f" skipped={result.skipped})"
    )
    print(
        f"Cost  (prompt_tokens={result.total_prompt_tokens},"
        f" completion_tokens={result.total_completion_tokens},"
        f" total_latency_ms={result.total_latency_ms:.1f})"
    )


def parse_artifact_name(path: str) -> tuple[str, str]:
    """Best-effort extraction of run_id and prompt_version from a results path."""
    base = os.path.basename(path)
    parts = base.replace(".jsonl", "").split("_")
    if len(parts) >= 4 and parts[0] == "results":
        return parts[1], parts[2]
    return "unknown", "unknown"
