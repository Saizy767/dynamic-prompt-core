# Tasks

## Phase 0 — Canon repair (done)

- [x] Archive 10 completed changes; sync deltas (2 specs created, 2 expanded, 1 stale sync skipped with evidence)
- [x] Validate canon: 27 specs, 346 requirements, `openspec validate --all` clean

## Phase 1 — Machine checks (done)

- [x] lint-imports, mypy, ruff check — all clean
- [x] ruff format audit — 106-file drift identified (fix in T-FMT)

## Phase 2 — Audit (done)

- [x] 5-group spec-vs-code audit (346 requirements); ledger in design.md

## Phase 3 — Auto-verdict spec syncs (spec-stale, Section A of design.md)

- [x] T-SYNC-1: Sync zero-shot-baseline cluster (A1–A6): stage1-baseline-runner ×2, extractor, classification-rules, dataset-and-prompt ×2, classifier
- [x] T-SYNC-2: Sync comparator stripping (A7–A11): stage2-version-comparator — rewrite to `decide()`-only component (respect V-CYCLE-1 verdict for the decision-rule clause)
- [x] T-SYNC-3: Sync llm-logit-scorer (A12–A15): port naming, tokenization boundary, error location, batching — align to classifier/llama-server specs
- [x] T-SYNC-4: Sync judgment-prompt (A16–A18): naming, server-era batching, active-rules semantic criteria (respect V-SCORE-1 verdict for the softmax clause)
- [x] T-SYNC-5: Sync transport specs (A19, A23–A25): batch methods removed, smoke test location, AsyncTask naming
- [x] T-SYNC-6: Sync architecture Non-goals (A26): AsyncTaskClient → AsyncTask
- [x] T-SYNC-7: Sync candidate-scoring-validation wording (A27): candidate provenance via composition root
- [x] T-SYNC-8: Sync stage1-metrics (A28, A29 + D6-share per V-CYCLE-10): drop confidence, ok-only predicted_decision, holdout-driven decision
- [x] T-SYNC-9: Sync dataset-and-prompt (A20–A22, A29): schema locations, generative removal, file-based extraction prompt, holdout clause
- [x] T-SYNC-10: Sync orchestrator (A30–A34): RunCycleDeps fields, queue replacement, in_prompt persistence, classify-v0, stop-reason enumeration (respect V-CYCLE-1/2/3/4/5/6 verdicts)
- [x] T-SYNC-11: Sync prompt-composer-v2 (A35 + per V-CYCLE-8/9 verdicts): upstream validation, hash/rules shape, max_rule_words

## Phase 3 — Auto-verdict code/test fixes (Section B)

- [x] T-TEST-1: Add metric-invariance test (vary Judgment.score, selection unchanged → metrics unchanged)
- [x] T-TEST-2: Add non-blocking scoring-boundary test (fake blocking adapter + concurrent event-loop progress)
- [x] T-TEST-3: Add `classify_many` to full-production forbidden-symbol scan
- [x] T-FMT: Run `ruff format` (106 files); pin ruff version in requirements
- [x] T-CONFIG-1: Clean stale config.toml comments (asyncTask.py, dataset.py references)
- [x] T-TEST-4: Architectural-invariants test: add the claimed application-vs-infrastructure-classifier check (or fix docstring) — depends on V-ARCH-1

## Phase 3 — Escalation-dependent tasks (apply owner verdicts)

- [x] T-V-ARCH-1: Resolve application→infrastructure layering (code extraction + linter contract, or spec/linter amendment)
- [x] T-V-ARCH-2: Resolve dependency-inversion debt at LLM seam (port alignment, tokenizer exposure, casts, session ownership)
- [x] T-V-ARCH-3: CLI signal handling (add SIGINT/SIGTERM, or scope spec to server)
- [x] T-V-SCORE-1: Score semantics (docstring + spec sync, or remove softmax)
- [x] T-V-SCORE-2: Timeout retry (add to retry loop, or spec: fail fast)
- [x] T-V-SCORE-3: Rule validation location (classifier boundary, or upstream per spec sync)
- [x] T-V-CYCLE-1: Accept-on-tie (keep tie-breaker + sync specs, or revert to strict)
- [x] T-V-CYCLE-2: unrecoverable_error summary (write it, or sync spec)
- [x] T-V-CYCLE-3: stop_on_first_error (implement, or remove from spec)
- [x] T-V-CYCLE-4: Missing active prompt (raise, or document fallback)
- [x] T-V-CYCLE-5: Stop-criteria activation (wire by default + config section + token tracking, or document optional-off; fix dead max_rounds_reached fallback either way)
- [x] T-V-CYCLE-6: Report/summary content (implement vN_k/cluster-ids/changed-decisions/decision artifacts, or sync specs)
- [x] T-V-CYCLE-7: Observer dev-after metrics (populate metrics_dev_new, or relabel)
- [x] T-V-CYCLE-8: Composer hash + rules shape (full sha256, or sync spec)
- [x] T-V-CYCLE-9: max_rule_words (sync spec to 20, or code to 15)
- [x] T-V-CYCLE-10: Logging-detail cluster (6 small additions, or trim spec clauses)
- [x] T-V-CYCLE-11: Artifacts via run_repository port (route through port, or drop param + sync)
- [x] T-V-STAGE1-1: Request log wiring (wire log_path, or remove clause)
- [x] T-V-STAGE1-2: Thesis bank resume (implement reload, or sync spec)
- [x] T-V-TRANS-1: Empty-list atomicity (atomic/skip write, or spec exception)
- [x] T-V-TRANS-2: teacher.use_refined (implement consumer, or remove key)
- [x] T-V-TRANS-3: Cross-task-transfer spec (retire, or annotate as roadmap)

## Phase 4 — Verification

- [x] T-VERIFY-1: Re-run lint-imports, mypy, ruff check, ruff format --check, pytest
- [x] T-VERIFY-2: `openspec validate --all` clean; update ledger verdict table; archive this change
