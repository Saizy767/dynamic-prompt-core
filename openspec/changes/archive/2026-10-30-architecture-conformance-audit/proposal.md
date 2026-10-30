# Proposal

## Why

The project's designed architecture lives in `openspec/specs/` (27 specs, 346
requirements), but 10 completed changes were never archived, two capabilities
had no main spec at all (`stage3-prompt-version-store`,
`stage5-auto-stop-criteria`), and recent refactors (`f8950f9`, `7b86c93`,
`75c04d9`, `1a243fc`) changed behavior without syncing specs. The goal of this
change is to (1) restore a complete, truthful spec canon, (2) record every
spec-vs-code divergence with evidence, and (3) apply verdicts: sync stale specs
or fix code, per the mixed-arbitration policy (deliberate later commit wins by
default; ambiguous cases escalate to the project owner).

## What Changes

Phase 0 — canon repair (already done, see design.md):
- Archived all 10 completed-but-unarchived changes; synced their deltas:
  created `stage3-prompt-version-store` (16 reqs) and `stage5-auto-stop-criteria`
  (14 reqs) main specs, expanded `stage4-teacher-refinement` (13→18) and
  `stage3-cycle-orchestrator` (25→26), archived 6 changes as already-synced,
  archived 1 empty shell (`config-audit-cleanup`).

Phase 1 — machine checks (already done):
- `lint-imports` 6/6 KEPT; `mypy src/` clean (111 files); `ruff check` clean;
  `ruff format --check` fails on 106 files (ruff version drift: blank line
  after module docstring) — task T-FMT.

Phase 2 — full spec-vs-code audit (done, ledger in design.md):
- 346 requirements across 27 specs verified against `src/`, `config.toml`,
  and `tests/`; ~86 findings recorded with file:line evidence and git history.
- Auto-verdicts (spec-stale → sync spec; spec-right → code/test fix) recorded
  as tasks; ~20 ambiguous findings escalate to the project owner in batches.

Phase 3 — implementation (this change's remaining work):
- Apply auto-verdict spec syncs (~40 requirements across ~15 specs).
- Apply owner verdicts on escalated items (code fixes and/or spec syncs).
- Add missing tests demanded by specs (score invariance, non-blocking event
  loop, forbidden-symbol scan completeness).
- Re-run static checks; keep the ledger verdict column updated.

## Capabilities

Affected specs (delta specs authored during implementation):
architecture, classifier, candidate-scoring-contracts,
candidate-scoring-validation, llm-logit-scorer, llama-server-logit-scorer,
stage3-cycle-orchestrator, stage2-version-comparator,
stage2-rule-candidate-selector, stage5-auto-stop-criteria,
stage3-prompt-composer-v2, stage3-prompt-version-store, stage3-cycle-observer,
stage1-baseline-runner, stage1-metrics, stage1-thesis-analyzer, extractor,
judgment-prompt, classification-rules, dataset-and-prompt,
llm-transport-client, llm-server-launcher, run-repository,
stage4-teacher-refinement, stage4-cross-task-transfer (retirement pending
verdict V-TRANS-3).
