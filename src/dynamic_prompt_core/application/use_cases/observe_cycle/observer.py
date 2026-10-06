"""
Stage 3 cycle observer.

Loads the artifacts produced by a completed optimization cycle (per-round
reports, decision artifacts, prompt-version records, metrics, thesis dumps, and
changed-decisions lists) and renders a human- or machine-readable review of what
changed between rounds: per-round summaries, prompt-rule evolution, rejected
attempts, metric trends, changed predictions, a thesis-store summary, and a
final observation with net improvement and stop reason.

The observer is strictly read-only: it reads cycle artifacts and writes only its
own report and log. It never mutates the active version, thesis store, prompt
store, or any cycle artifact.

Usage:
    python cycle_observer.py <run_id>
    python cycle_observer.py <run_id> --config config.toml --format json --output /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_OUTPUT_FORMAT = "markdown"
DEFAULT_OUTPUT_PATH = "observer_report.md"
DEFAULT_MAX_EXAMPLE_TEXT_LENGTH = 100
DEFAULT_TOP_CLUSTERS_COUNT = 10
DEFAULT_INCLUDE_HOLDOUT = True
DEFAULT_OUTPUT_DIR = "data/results"

METRIC_KEYS = ("accuracy", "macro_f1", "minority_f1", "parse_failures")


class CycleObserverError(ValueError):
    """Raised when configuration is invalid or an artifact cannot be loaded."""


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class CycleObserverConfig:
    output_format: str = DEFAULT_OUTPUT_FORMAT
    output_path: str = DEFAULT_OUTPUT_PATH
    max_example_text_length: int = DEFAULT_MAX_EXAMPLE_TEXT_LENGTH
    top_clusters_count: int = DEFAULT_TOP_CLUSTERS_COUNT
    include_holdout: bool = DEFAULT_INCLUDE_HOLDOUT
    output_dir: str = DEFAULT_OUTPUT_DIR
    config_path: str = DEFAULT_CONFIG_PATH

    @classmethod
    def from_config(cls, config_path: str = DEFAULT_CONFIG_PATH) -> CycleObserverConfig:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        s = config.get("cycle_observer", {})
        return cls(
            output_format=s.get("output_format", DEFAULT_OUTPUT_FORMAT),
            output_path=s.get("output_path", DEFAULT_OUTPUT_PATH),
            max_example_text_length=int(
                s.get("max_example_text_length", DEFAULT_MAX_EXAMPLE_TEXT_LENGTH)
            ),
            top_clusters_count=int(s.get("top_clusters_count", DEFAULT_TOP_CLUSTERS_COUNT)),
            include_holdout=bool(s.get("include_holdout", DEFAULT_INCLUDE_HOLDOUT)),
            output_dir=s.get("output_dir", DEFAULT_OUTPUT_DIR),
            config_path=config_path,
        )


# --------------------------------------------------------------------------- #
#  Report data model
# --------------------------------------------------------------------------- #
@dataclass
class RoundSummary:
    round_number: int
    active_version_at_start: str
    new_version: str
    decision: str
    metrics_dev_before: dict[str, Any]
    metrics_dev_after: dict[str, Any]
    minority_f1_change: float | None
    changed_predictions: int
    rollback_counter: int


@dataclass
class RuleChange:
    text: str
    cluster_id: int | None
    first_appeared_version: str
    status: str  # added | removed | preserved


@dataclass
class VersionEvolution:
    from_version: str
    to_version: str
    added: list[RuleChange]
    removed: list[RuleChange]
    preserved: list[RuleChange]


@dataclass
class RejectedRule:
    cluster_id: int | None
    text: str | None
    cosine: float | None
    threshold: float | None
    reason: str


@dataclass
class RejectedAttempt:
    round: int
    version: str
    reason: str
    formulated_rules: list[dict[str, Any]]
    source_candidates: list[int]
    rejected_rules: list[RejectedRule]
    metrics_new: dict[str, Any] | None
    metrics_active: dict[str, Any] | None
    diff_macro_f1: float | None
    individual_details_persisted: bool


@dataclass
class MetricTrendRow:
    round_number: int
    accuracy: Any
    macro_f1: Any
    minority_f1: Any
    parse_failures: Any
    holdout_accuracy: Any = None
    holdout_macro_f1: Any = None
    holdout_minority_f1: Any = None
    holdout_parse_failures: Any = None


@dataclass
class MetricTrends:
    rows: list[MetricTrendRow] = field(default_factory=list)
    include_holdout: bool = True


@dataclass
class ChangedDecisionEntry:
    round_number: int
    example_id: Any
    direction: str
    text: str


@dataclass
class ChangedDecisions:
    entries: list[ChangedDecisionEntry] = field(default_factory=list)
    cycle_wide: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class ClusterSummary:
    cluster_id: int
    frequency: int
    precision: float
    representative_theses: list[str]


@dataclass
class ThesisSummary:
    total_theses: int
    cluster_count: int
    top_clusters: list[ClusterSummary]
    in_prompt_cluster_count: int
    unassigned_theses: int


@dataclass
class MetricDelta:
    name: str
    start: Any
    final: Any
    absolute: Any
    relative: Any


@dataclass
class FinalObservation:
    start_version: str
    start_metrics: dict[str, Any]
    final_version: str
    final_metrics: dict[str, Any]
    deltas: list[MetricDelta]
    stop_reason: str
    total_rounds: int
    accepted_count: int
    rollback_count: int
    improved: bool
    message: str


@dataclass
class ObservationReport:
    round_summaries: list[RoundSummary] = field(default_factory=list)
    prompt_evolution: list[VersionEvolution] = field(default_factory=list)
    rejected_attempts: list[RejectedAttempt] = field(default_factory=list)
    metric_trends: MetricTrends = field(default_factory=MetricTrends)
    changed_decisions: ChangedDecisions = field(default_factory=ChangedDecisions)
    thesis_summary: ThesisSummary | None = None
    final_observation: FinalObservation | None = None


# --------------------------------------------------------------------------- #
#  Loaded-artifact bundle
# --------------------------------------------------------------------------- #
@dataclass
class CycleArtifacts:
    reports: list[dict[str, Any]] = field(default_factory=list)
    report_paths: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    decision_paths: list[str] = field(default_factory=list)
    prompt_versions: list[dict[str, Any]] = field(default_factory=list)
    prompt_version_paths: list[str] = field(default_factory=list)
    thesis_dumps: list[dict[str, Any]] = field(default_factory=list)
    thesis_dump_paths: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    summary_path: str | None = None


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _safe_metric(metrics: dict[str, Any] | None, key: str) -> Any:
    if not metrics:
        return None
    return metrics.get(key)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate(text: str, max_length: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\u2026"


# --------------------------------------------------------------------------- #
#  Artifact discovery and loading
# --------------------------------------------------------------------------- #
def _sort_report_paths(paths: list[str]) -> list[str]:
    """Sort report paths by round number, then timestamp."""
    import re

    def key(p: str) -> tuple[int, str]:
        m = re.search(r"round(\d+)_(\w+)\.json$", p)
        if m:
            return (int(m.group(1)), m.group(2))
        return (0, p)

    return sorted(paths, key=key)


def _sort_decision_paths(paths: list[str]) -> list[str]:
    """Sort decision paths by timestamp (ascending)."""
    import re

    def key(p: str) -> str:
        m = re.search(r"_(\w+)\.json$", p)
        return m.group(1) if m else p

    return sorted(paths, key=key)


def _sort_prompt_version_paths(paths: list[str]) -> list[str]:
    """Sort prompt-version paths by version number."""
    import re

    def key(p: str) -> int:
        m = re.search(r"prompt_v(\S+?)_", p)
        if m:
            text = m.group(1)
            digits = re.findall(r"\d+", text)
            if digits:
                return int(digits[-1])
        return 0

    return sorted(paths, key=key)


def _sort_thesis_dump_paths(paths: list[str]) -> list[str]:
    """Sort thesis-dump paths by timestamp (ascending)."""
    import re

    def key(p: str) -> str:
        m = re.search(r"_(\w+)\.json$", p)
        return m.group(1) if m else p

    return sorted(paths, key=key)


def discover_artifacts(run_id: str, config: CycleObserverConfig) -> dict[str, list[str]]:
    """Glob ``output_dir`` for the artifacts of ``run_id`` (design D2).

    Returns a dict with sorted path lists: reports, decisions, prompt_versions,
    thesis_dumps, summaries.
    """
    import glob

    base = config.output_dir
    reports = glob.glob(os.path.join(base, f"report_{run_id}_round*_*.json"))
    decisions = glob.glob(os.path.join(base, f"decision_{run_id}_*_*.json"))
    prompt_versions = glob.glob(os.path.join(base, f"prompt_v*_{run_id}_*.json"))
    thesis_dumps = glob.glob(os.path.join(base, f"thesis_{run_id}_*_*.json"))
    summaries = glob.glob(os.path.join(base, f"summary_{run_id}_*.json"))

    return {
        "reports": _sort_report_paths(reports),
        "decisions": _sort_decision_paths(decisions),
        "prompt_versions": _sort_prompt_version_paths(prompt_versions),
        "thesis_dumps": _sort_thesis_dump_paths(thesis_dumps),
        "summaries": sorted(summaries),
    }


def _load_json(path: str) -> dict[str, Any]:
    """Load a JSON file, raising CycleObserverError naming the file on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))
    except FileNotFoundError as exc:
        raise CycleObserverError(f"artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CycleObserverError(f"corrupted artifact (invalid JSON): {path}") from exc
    except OSError as exc:
        raise CycleObserverError(f"failed to read artifact: {path}: {exc}") from exc


def load_cycle_artifacts(run_id: str, config: CycleObserverConfig) -> CycleArtifacts:
    """Discover and load all cycle artifacts for ``run_id`` (design D2, D3).

    Raises CycleObserverError naming the missing or corrupted file when an
    expected artifact is absent or unparseable.
    """
    discovered = discover_artifacts(run_id, config)

    bundle = CycleArtifacts()

    # Reports — at least one is expected for a completed cycle.
    for path in discovered["reports"]:
        bundle.reports.append(_load_json(path))
        bundle.report_paths.append(path)

    # Decisions.
    for path in discovered["decisions"]:
        bundle.decisions.append(_load_json(path))
        bundle.decision_paths.append(path)

    # Prompt versions.
    for path in discovered["prompt_versions"]:
        bundle.prompt_versions.append(_load_json(path))
        bundle.prompt_version_paths.append(path)

    # Thesis dumps.
    for path in discovered["thesis_dumps"]:
        bundle.thesis_dumps.append(_load_json(path))
        bundle.thesis_dump_paths.append(path)

    # Summary — exactly one is expected for a completed cycle.
    summary_paths = discovered["summaries"]
    if not summary_paths:
        raise CycleObserverError(
            f"missing final summary for run_id={run_id!r} in {config.output_dir!r}"
        )
    if len(summary_paths) > 1:
        summary_paths = sorted(summary_paths)[-1:]
    bundle.summary_path = summary_paths[0]
    bundle.summary = _load_json(summary_paths[0])

    return bundle


# --------------------------------------------------------------------------- #
#  Section builders — round summaries, prompt evolution, rejected attempts
# --------------------------------------------------------------------------- #
def build_round_summaries(
    bundle: CycleArtifacts, config: CycleObserverConfig
) -> list[RoundSummary]:
    """Produce one RoundSummary per per-round report (design D4.1).

    Rounds absent from the reports are skipped and logged.
    """
    summaries: list[RoundSummary] = []
    seen_rounds = set()
    for report in bundle.reports:
        round_number = report.get("round_number")
        if round_number is None:
            log.warning("report missing round_number; skipping: %s", report)
            continue
        if round_number in seen_rounds:
            log.warning("duplicate round %s; skipping", round_number)
            continue
        seen_rounds.add(round_number)

        metrics_before = report.get("metrics_dev_active", {}) or {}
        metrics_after = report.get("metrics_dev_new", {}) or {}
        minority_before = _as_float(metrics_before.get("minority_f1"))
        minority_after = _as_float(metrics_after.get("minority_f1"))
        if minority_before is not None and minority_after is not None:
            minority_change = minority_after - minority_before
        else:
            minority_change = None

        changed = report.get("changed_decisions", []) or []
        summaries.append(
            RoundSummary(
                round_number=round_number,
                active_version_at_start=report.get("active_version_at_start", ""),
                new_version=report.get("new_version", ""),
                decision=report.get("decision", ""),
                metrics_dev_before=metrics_before,
                metrics_dev_after=metrics_after,
                minority_f1_change=minority_change,
                changed_predictions=len(changed),
                rollback_counter=report.get("rollback_counter", 0),
            )
        )

    if bundle.reports:
        expected = {r.get("round_number") for r in bundle.reports}
        max_round = max(cast(set[int], expected)) if expected else 0
        for n in range(1, max_round + 1):
            if n not in seen_rounds:
                log.warning("round %s absent from reports; no summary produced", n)
    return summaries


def _rule_text_set(rules: list[dict[str, Any]]) -> set[str]:
    """Return the set of normalized rule texts from a rules list."""
    texts: set[str] = set()
    for r in rules:
        t = r.get("text") if isinstance(r, dict) else r
        if t is not None:
            texts.add(" ".join(str(t).split()))
    return texts


def _version_number(version_str: str) -> int:
    """Extract the trailing integer from a version string like 'classify-v2'."""
    import re

    m = re.findall(r"\d+", str(version_str))
    return int(m[-1]) if m else 0


def build_prompt_evolution(
    bundle: CycleArtifacts, config: CycleObserverConfig
) -> list[VersionEvolution]:
    """Compute added/removed/preserved rules between consecutive active versions.

    Active versions are taken from the prompt-version artifacts sorted by
    version number. The baseline (v0) has no rules in the composer artifacts, so
    only composed versions with a ``rules`` field are compared (design D4).
    """
    versions = bundle.prompt_versions
    if len(versions) < 2:
        return []

    # Map version string -> rules list for first-appearance lookup.
    first_appeared: dict[str, str] = {}
    for v in versions:
        vstr = v.get("version", "")
        for r in v.get("rules", []) or []:
            t = r.get("text") if isinstance(r, dict) else r
            if t is None:
                continue
            norm = " ".join(str(t).split())
            if norm not in first_appeared:
                first_appeared[norm] = vstr

    evolutions: list[VersionEvolution] = []
    for i in range(1, len(versions)):
        prev = versions[i - 1]
        curr = versions[i]
        prev_set = _rule_text_set(prev.get("rules", []) or [])
        curr_set = _rule_text_set(curr.get("rules", []) or [])

        from_v = prev.get("version", "")
        to_v = curr.get("version", "")

        def _change(text: str, status: str) -> RuleChange:
            norm = " ".join(str(text).split())
            cluster_id = None
            for r in curr.get("rules", []) or []:
                rt = r.get("text") if isinstance(r, dict) else r
                if rt is not None and " ".join(str(rt).split()) == norm:
                    cluster_id = r.get("cluster_id") if isinstance(r, dict) else None
                    break
            if cluster_id is None and status != "added":
                for r in prev.get("rules", []) or []:
                    rt = r.get("text") if isinstance(r, dict) else r
                    if rt is not None and " ".join(str(rt).split()) == norm:
                        cluster_id = r.get("cluster_id") if isinstance(r, dict) else None
                        break
            return RuleChange(
                text=text,
                cluster_id=cluster_id,
                first_appeared_version=first_appeared.get(norm, to_v),
                status=status,
            )

        added = [_change(t, "added") for t in sorted(curr_set - prev_set)]
        removed = [_change(t, "removed") for t in sorted(prev_set - curr_set)]
        preserved = [_change(t, "preserved") for t in sorted(prev_set & curr_set)]
        evolutions.append(
            VersionEvolution(
                from_version=from_v,
                to_version=to_v,
                added=added,
                removed=removed,
                preserved=preserved,
            )
        )
    return evolutions


def build_rejected_attempts(
    bundle: CycleArtifacts, config: CycleObserverConfig
) -> list[RejectedAttempt]:
    """Build the rejected-attempts overview from rollback history (design D5).

    Cross-references each rolled-back version's decision artifact and
    prompt-version artifact. For distortion rejections, includes rule text,
    cosine, and threshold from ``metadata.rejected_rules``; falls back to
    aggregate counts + threshold when absent.
    """
    summary = bundle.summary or {}
    rollback_history = summary.get("rollback_history", []) or []

    # Index decisions and prompt versions by new version.
    decisions_by_version: dict[str, dict[str, Any]] = {}
    for dec in bundle.decisions:
        decisions_by_version[dec.get("new_version", "")] = dec
    versions_by_version: dict[str, dict[str, Any]] = {}
    for v in bundle.prompt_versions:
        versions_by_version[v.get("version", "")] = v

    attempts: list[RejectedAttempt] = []
    for entry in rollback_history:
        version = entry.get("version", "")
        round_number = entry.get("round", 0)
        reason = entry.get("reason", "")

        dec = decisions_by_version.get(version, {})
        pv = versions_by_version.get(version, {})
        meta = pv.get("metadata", {}) or {}
        rejected_raw = meta.get("rejected_rules", []) or []
        counters = meta.get("counters", {}) or {}
        threshold = (meta.get("config", {}) or {}).get("distortion_threshold")

        individual_persisted = bool(rejected_raw)
        rejected_rules: list[RejectedRule] = []
        for rr in rejected_raw:
            rejected_rules.append(
                RejectedRule(
                    cluster_id=rr.get("cluster_id"),
                    text=rr.get("text"),
                    cosine=rr.get("cosine"),
                    threshold=rr.get("threshold"),
                    reason=rr.get("reason", ""),
                )
            )
        if not individual_persisted:
            agg = counters.get("rejected_distortion", 0)
            if agg:
                rejected_rules.append(
                    RejectedRule(
                        cluster_id=None,
                        text=None,
                        cosine=None,
                        threshold=threshold,
                        reason=f"aggregate: {agg} distortion rejection(s) (individual details not persisted)",  # noqa: E501
                    )
                )

        attempts.append(
            RejectedAttempt(
                round=round_number,
                version=version,
                reason=reason,
                formulated_rules=pv.get("rules", []) or [],
                source_candidates=pv.get("source_candidates", []) or [],
                rejected_rules=rejected_rules,
                metrics_new=dec.get("metrics_new"),
                metrics_active=dec.get("metrics_active"),
                diff_macro_f1=dec.get("diff_macro_f1"),
                individual_details_persisted=individual_persisted,
            )
        )
    return attempts


# --------------------------------------------------------------------------- #
#  Section builders — metric trends, changed decisions, thesis summary, final
# --------------------------------------------------------------------------- #
def _metric_or_na(metrics: dict[str, Any] | None, key: str) -> Any:
    if not metrics:
        return "N/A"
    if key not in metrics or metrics[key] is None:
        log.warning("metric %s absent for a round; marking N/A", key)
        return "N/A"
    return metrics[key]


def build_metric_trends(bundle: CycleArtifacts, config: CycleObserverConfig) -> MetricTrends:
    """Build one row per round from per-round reports (design D6).

    Holdout columns are included only when ``include_holdout`` is true.
    """
    rows: list[MetricTrendRow] = []
    for report in bundle.reports:
        dev = report.get("metrics_dev_new", {}) or {}
        holdout = report.get("metrics_holdout_new", {}) or {}
        row = MetricTrendRow(
            round_number=report.get("round_number", 0),
            accuracy=_metric_or_na(dev, "accuracy"),
            macro_f1=_metric_or_na(dev, "macro_f1"),
            minority_f1=_metric_or_na(dev, "minority_f1"),
            parse_failures=_metric_or_na(dev, "parse_failures"),
        )
        if config.include_holdout:
            row.holdout_accuracy = _metric_or_na(holdout, "accuracy")
            row.holdout_macro_f1 = _metric_or_na(holdout, "macro_f1")
            row.holdout_minority_f1 = _metric_or_na(holdout, "minority_f1")
            row.holdout_parse_failures = _metric_or_na(holdout, "parse_failures")
        rows.append(row)
    return MetricTrends(rows=rows, include_holdout=config.include_holdout)


def build_changed_decisions(
    bundle: CycleArtifacts, config: CycleObserverConfig
) -> ChangedDecisions:
    """Collect changed decisions per round with truncated text (design D7)."""
    entries: list[ChangedDecisionEntry] = []
    for report in bundle.reports:
        round_number = report.get("round_number", 0)
        for cd in report.get("changed_decisions", []) or []:
            text = cd.get("text", "") or ""
            entries.append(
                ChangedDecisionEntry(
                    round_number=round_number,
                    example_id=cd.get("id"),
                    direction=cd.get("direction", ""),
                    text=_truncate(text, config.max_example_text_length),
                )
            )

    cycle_wide = (bundle.summary or {}).get("all_changed_decisions", []) or []
    if not entries and not cycle_wide:
        message = "No predictions changed between versions."
    else:
        message = f"{len(entries)} changed prediction(s) across rounds."
    return ChangedDecisions(entries=entries, cycle_wide=cycle_wide, message=message)


def build_thesis_summary(
    bundle: CycleArtifacts, config: CycleObserverConfig
) -> ThesisSummary | None:
    """Summarize the latest thesis dump (design D8)."""
    if not bundle.thesis_dumps:
        return None
    dump = bundle.thesis_dumps[-1]
    theses = dump.get("theses", {}) or {}
    clusters = dump.get("clusters", {}) or {}

    total_theses = len(theses)
    cluster_count = len(clusters)

    # in_prompt cluster count: clusters that have any thesis with in_prompt=true.
    in_prompt_clusters = set()
    for entry in theses.values():
        if entry.get("in_prompt") and entry.get("cluster_id") is not None:
            in_prompt_clusters.add(entry["cluster_id"])

    # unassigned theses: cluster_id is None or -1.
    unassigned = sum(
        1 for e in theses.values() if e.get("cluster_id") is None or e.get("cluster_id") == -1
    )

    # Top-N clusters by frequency with precision and representative theses.
    cluster_list = []
    for cid_str, cdata in clusters.items():
        cid = cdata.get("cluster_id", int(cid_str))
        cluster_list.append((cid, cdata))
    cluster_list.sort(key=lambda x: x[1].get("frequency", 0), reverse=True)
    top_n = cluster_list[: config.top_clusters_count]

    top_clusters: list[ClusterSummary] = []
    for cid, cdata in top_n:
        member_norms = cdata.get("member_norms", []) or []
        reps: list[str] = []
        for norm in member_norms[:3]:
            entry = theses.get(norm)
            if entry:
                reps.append(entry.get("text_raw", ""))
        top_clusters.append(
            ClusterSummary(
                cluster_id=cid,
                frequency=cdata.get("frequency", 0),
                precision=cdata.get("precision", 0.0),
                representative_theses=reps,
            )
        )

    return ThesisSummary(
        total_theses=total_theses,
        cluster_count=cluster_count,
        top_clusters=top_clusters,
        in_prompt_cluster_count=len(in_prompt_clusters),
        unassigned_theses=unassigned,
    )


def build_final_observation(
    bundle: CycleArtifacts, config: CycleObserverConfig
) -> FinalObservation | None:
    """Build the final observation with start vs final deltas (design D9)."""
    summary = bundle.summary
    if not summary:
        return None
    if not bundle.reports:
        return None

    first_report = bundle.reports[0]
    start_version = first_report.get("active_version_at_start", "")
    start_metrics = first_report.get("metrics_dev_active", {}) or {}
    final_version = summary.get("final_active_version", "")
    final_metrics = summary.get("final_dev_metrics", {}) or {}
    stop_reason = summary.get("stop_reason", "")
    total_rounds = summary.get("total_rounds", 0)
    accepted_count = len(summary.get("accepted_history", []) or [])
    rollback_count = len(summary.get("rollback_history", []) or [])

    deltas: list[MetricDelta] = []
    improved = False
    for key in METRIC_KEYS:
        start_v = _as_float(start_metrics.get(key))
        final_v = _as_float(final_metrics.get(key))
        if start_v is not None and final_v is not None:
            absolute = final_v - start_v
            if start_v == 0:
                relative: Any = "N/A"
                log.warning("start metric %s is 0; relative marked N/A", key)
            else:
                relative = (final_v - start_v) / start_v
            if absolute > 0:
                improved = True
        else:
            absolute = None
            relative = None
        deltas.append(
            MetricDelta(
                name=key,
                start=start_v,
                final=final_v,
                absolute=absolute,
                relative=relative,
            )
        )

    if improved:
        message = "Metrics improved relative to the start version."
    else:
        message = "No improvement: metrics did not improve relative to the start version."
    return FinalObservation(
        start_version=start_version,
        start_metrics=start_metrics,
        final_version=final_version,
        final_metrics=final_metrics,
        deltas=deltas,
        stop_reason=stop_reason,
        total_rounds=total_rounds,
        accepted_count=accepted_count,
        rollback_count=rollback_count,
        improved=improved,
        message=message,
    )


# --------------------------------------------------------------------------- #
#  Rendering, report writing, and logging
# --------------------------------------------------------------------------- #
def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: ObservationReport) -> str:
    """Render the report to a markdown string (design D10)."""
    lines: list[str] = []
    lines.append("# Cycle Observer Report")
    lines.append("")

    # Round summaries
    lines.append("## Round Summaries")
    lines.append("")
    if report.round_summaries:
        lines.append(
            "| Round | Active at start | New version | Decision | Dev F1 before | Dev F1 after | Minority F1 change | Changed preds | Rollbacks |"  # noqa: E501
        )
        lines.append(
            "|-------|-----------------|-------------|----------|---------------|--------------|--------------------|---------------|-----------|"
        )
        for s in report.round_summaries:
            lines.append(
                f"| {s.round_number} | {s.active_version_at_start} | {s.new_version} | {s.decision} "  # noqa: E501
                f"| {_fmt(_as_float(s.metrics_dev_before.get('macro_f1')))} "
                f"| {_fmt(_as_float(s.metrics_dev_after.get('macro_f1')))} "
                f"| {_fmt(s.minority_f1_change)} | {s.changed_predictions} | {s.rollback_counter} |"
            )
    else:
        lines.append("_No round summaries._")
    lines.append("")

    # Prompt evolution
    lines.append("## Prompt Evolution")
    lines.append("")
    if report.prompt_evolution:
        for evo in report.prompt_evolution:
            lines.append(f"### {evo.from_version} \u2192 {evo.to_version}")
            for label, changes in (
                ("Added", evo.added),
                ("Removed", evo.removed),
                ("Preserved", evo.preserved),
            ):
                lines.append(f"- **{label}** ({len(changes)}):")
                for c in changes:
                    lines.append(
                        f"  - {c.text} _(cluster {c.cluster_id}, first in {c.first_appeared_version}, {c.status})_"  # noqa: E501
                    )
    else:
        lines.append("_No prompt-version pairs to compare._")
    lines.append("")

    # Rejected attempts
    lines.append("## Rejected Attempts")
    lines.append("")
    if report.rejected_attempts:
        for ra in report.rejected_attempts:
            lines.append(f"### Round {ra.round}: {ra.version} ({ra.reason})")
            lines.append(f"- Formulated rules: {len(ra.formulated_rules)}")
            lines.append(f"- Source candidates (clusters): {ra.source_candidates}")
            if ra.rejected_rules:
                lines.append("- Rejected rules:")
                for rr in ra.rejected_rules:
                    lines.append(
                        f"  - cluster {rr.cluster_id}: {rr.text!r} (cosine={_fmt(rr.cosine)}, threshold={_fmt(rr.threshold)}, reason={rr.reason})"  # noqa: E501
                    )
            else:
                lines.append("- No rejected rules recorded.")
            if not ra.individual_details_persisted:
                lines.append(
                    "- _Individual rejection details were not persisted; showing aggregate._"
                )
    else:
        lines.append("_No rejected attempts._")
    lines.append("")

    # Metric trends
    lines.append("## Metric Trends")
    lines.append("")
    trends = report.metric_trends
    if trends.rows:
        cols = ["Round", "Accuracy", "Macro-F1", "Minority-F1", "Parse failures"]
        if trends.include_holdout:
            cols += ["Holdout Acc", "Holdout F1", "Holdout minF1", "Holdout parse"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in trends.rows:
            vals = [
                str(row.round_number),
                _fmt(row.accuracy),
                _fmt(row.macro_f1),
                _fmt(row.minority_f1),
                _fmt(row.parse_failures),
            ]
            if trends.include_holdout:
                vals += [
                    _fmt(row.holdout_accuracy),
                    _fmt(row.holdout_macro_f1),
                    _fmt(row.holdout_minority_f1),
                    _fmt(row.holdout_parse_failures),
                ]
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("_No metric trends._")
    lines.append("")

    # Changed decisions
    lines.append("## Changed Decisions")
    lines.append("")
    cd = report.changed_decisions
    lines.append(cd.message)
    if cd.entries:
        lines.append("")
        lines.append("| Round | Example id | Direction | Text |")
        lines.append("|-------|------------|-----------|------|")
        for e in cd.entries:
            lines.append(f"| {e.round_number} | {e.example_id} | {e.direction} | {e.text} |")
    lines.append("")

    # Thesis summary
    lines.append("## Thesis Store Summary")
    lines.append("")
    ts = report.thesis_summary
    if ts:
        lines.append(f"- Total theses: {ts.total_theses}")
        lines.append(f"- Cluster count: {ts.cluster_count}")
        lines.append(f"- Clusters with `in_prompt=true`: {ts.in_prompt_cluster_count}")
        lines.append(f"- Unassigned theses: {ts.unassigned_theses}")
        lines.append(f"- Top-{len(ts.top_clusters)} clusters by frequency:")
        for cluster in ts.top_clusters:
            reps = "; ".join(cluster.representative_theses) if cluster.representative_theses else ""
            lines.append(
                f"  - cluster {cluster.cluster_id}: frequency={cluster.frequency}, precision={_fmt(cluster.precision)}, reps=[{reps}]"  # noqa: E501
            )
    else:
        lines.append("_No thesis dump available._")
    lines.append("")

    # Final observation
    lines.append("## Final Observation")
    lines.append("")
    fo = report.final_observation
    if fo:
        lines.append(f"- Start version: {fo.start_version}")
        lines.append(f"- Final version: {fo.final_version}")
        lines.append(f"- Stop reason: {fo.stop_reason}")
        lines.append(f"- Total rounds: {fo.total_rounds}")
        lines.append(f"- Accepted: {fo.accepted_count}, Rolled back: {fo.rollback_count}")
        lines.append("- Deltas:")
        for d in fo.deltas:
            lines.append(
                f"  - {d.name}: start={_fmt(d.start)}, final={_fmt(d.final)}, absolute={_fmt(d.absolute)}, relative={_fmt(d.relative)}"  # noqa: E501
            )
        lines.append(f"- {fo.message}")
    else:
        lines.append("_No final observation available._")
    lines.append("")
    return "\n".join(lines)


def render_json(report: ObservationReport) -> str:
    """Serialize the report to a JSON string (design D10)."""
    return json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str)


def write_report(report: ObservationReport, config: CycleObserverConfig) -> str:
    """Write the report to ``output_path`` in the configured format (design D10)."""
    if config.output_format == "json":
        content = render_json(report)
    else:
        content = render_markdown(report)
    path = config.output_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _observer_log_path(run_id: str, config: CycleObserverConfig) -> str:
    return os.path.join(config.output_dir, f"observer_log_{run_id}.jsonl")


def log_event(
    event: str,
    details: dict[str, Any],
    run_id: str,
    config: CycleObserverConfig,
) -> None:
    """Append one JSON event to the observer log (design D12)."""
    os.makedirs(config.output_dir, exist_ok=True)
    path = _observer_log_path(run_id, config)
    entry = {
        "timestamp": _utc_iso(),
        "event": event,
        "details": details,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
#  Orchestration and CLI
# --------------------------------------------------------------------------- #
SECTION_NAMES = (
    "round_summaries",
    "prompt_evolution",
    "rejected_attempts",
    "metric_trends",
    "changed_decisions",
    "thesis_summary",
    "final_observation",
)


def run_observer(run_id: str, config: CycleObserverConfig) -> str:
    """Load artifacts, build all sections, write the report, and log events."""
    bundle = load_cycle_artifacts(run_id, config)
    log_event(
        "artifacts_loaded",
        {
            "reports": len(bundle.reports),
            "decisions": len(bundle.decisions),
            "prompt_versions": len(bundle.prompt_versions),
            "thesis_dumps": len(bundle.thesis_dumps),
        },
        run_id,
        config,
    )

    report = ObservationReport()

    report.round_summaries = build_round_summaries(bundle, config)
    log_event(
        "section_built",
        {"section": "round_summaries", "count": len(report.round_summaries)},
        run_id,
        config,
    )

    report.prompt_evolution = build_prompt_evolution(bundle, config)
    log_event(
        "section_built",
        {"section": "prompt_evolution", "count": len(report.prompt_evolution)},
        run_id,
        config,
    )

    report.rejected_attempts = build_rejected_attempts(bundle, config)
    log_event(
        "section_built",
        {"section": "rejected_attempts", "count": len(report.rejected_attempts)},
        run_id,
        config,
    )

    report.metric_trends = build_metric_trends(bundle, config)
    log_event(
        "section_built",
        {"section": "metric_trends", "count": len(report.metric_trends.rows)},
        run_id,
        config,
    )

    report.changed_decisions = build_changed_decisions(bundle, config)
    log_event(
        "section_built",
        {"section": "changed_decisions", "count": len(report.changed_decisions.entries)},
        run_id,
        config,
    )

    report.thesis_summary = build_thesis_summary(bundle, config)
    log_event(
        "section_built",
        {"section": "thesis_summary", "present": report.thesis_summary is not None},
        run_id,
        config,
    )

    report.final_observation = build_final_observation(bundle, config)
    log_event(
        "section_built",
        {"section": "final_observation", "present": report.final_observation is not None},
        run_id,
        config,
    )

    path = write_report(report, config)
    log_event("report_written", {"path": path, "format": config.output_format}, run_id, config)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 cycle observer — render a review of a completed cycle"
    )
    parser.add_argument("run_id", help="Cycle run_id to observe")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default=None,
        help="Output format (overrides config output_format)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (overrides config output_path)",
    )
    args = parser.parse_args()

    pass  # logging configured by composition root

    config = CycleObserverConfig.from_config(args.config)
    if args.format is not None:
        config.output_format = args.format
    if args.output is not None:
        config.output_path = args.output

    path = run_observer(args.run_id, config)
    print(f"Observer report written to: {path}")


if __name__ == "__main__":
    main()
