# Design — Conformance Audit Ledger

## Method

- Surface: `src/` + `config.toml` + `tests/` (tests secondary).
- Depth: requirements-level everywhere; full scenario-level for critical specs
  (architecture, stage3-cycle-orchestrator, classifier/candidate-scoring-*,
  llm-logit-scorer, llama-server-logit-scorer, stage3-prompt-version-store).
- Arbitration (mixed): a divergence where the code matches a deliberate later
  commit (and no newer spec contradicts) is auto-verdicted **spec-stale → sync
  spec**. Where code contradicts both the spec and the archives, or archives
  contradict each other, the item escalates to the project owner. The owner is
  the final arbiter.
- Verdict vocabulary: `sync-spec` (spec edited to match deliberate code),
  `fix-code` (code/test edited to match spec), `accept` (divergence kept, spec
  annotated), `retire-spec` (capability spec removed).

## Phase 0 — Canon repair (done)

| Action | Detail |
|---|---|
| Created `openspec/specs/stage3-prompt-version-store/` | 12 base reqs + atomic-persistence fix (2 MODIFIED + 4 ADDED) = 16 reqs |
| Created `openspec/specs/stage5-auto-stop-criteria/` | 14 reqs from archived delta |
| Expanded `stage4-teacher-refinement` | +5 ADDED (port, client, use case, wiring, domain purity), 2 MODIFIED applied (use_refined override, basicConfig ban) |
| Expanded `stage3-cycle-orchestrator` | +1 ADDED (Content-based stop evaluation) |
| Archived as already-synced (evolved in canon) | stage3-prompt-composer (content = stage3-prompt-composer-v2), stage2-version-comparator, migration-to-architecture, candidate-scoring-architecture-prep, judgment-prompt, integrate-candidate-scoring-into-optimization |
| Archived as empty shell | config-audit-cleanup (no content) |
| Skipped stale sync | stage5's "Final summary" MODIFIED (rounds/10 reasons) — superseded by ae8067e cycles rewrite |

Canon now: 27 specs, 346 requirements, `openspec validate --all` clean.

## Phase 1 — Machine checks (done)

| Check | Result |
|---|---|
| `lint-imports` | 6/6 contracts KEPT |
| `mypy src/` | clean, 111 files |
| `ruff check` | clean |
| `ruff format --check` | **106 files diverge** — ruff version drift (blank line after module docstring rule). Task T-FMT: run `ruff format`, pin ruff version. |

## Phase 2 — Divergence ledger

Counts: 346 requirements audited; ~246 conform; ~86 findings:
- 40 auto spec-stale verdicts (Section A)
- 6 auto code-fix verdicts (Section B)
- 21 escalations pending owner verdict (Section C, batches ARCH/SCORE/CYCLE/STAGE1/TRANS)
- 8 unverifiable/open (Section D)

### A. Auto verdicts — spec-stale → sync spec

Root cause R1 (zero-shot baseline, commit 1a243fc 2026-10-29, pinned by
`test_v0_empty_rules_still_calls_classifier`): empty `ClassificationRules` now
goes to the classifier as an honest zero-shot baseline instead of
short-circuiting to `predicted_decision=0`.

| # | Spec | Requirement | Sync to |
|---|---|---|---|
| A1 | stage1-baseline-runner | Active prompt rules extraction by runner | empty rules → classifier (only `layers is None` short-circuits) |
| A2 | stage1-baseline-runner | Run dataset through the model | same as A1 |
| A3 | extractor | Extractor runs after classification on dev | same as A1 |
| A4 | classification-rules | Empty rules represent absence, not an error | empty rules DO reach classifier; drop `build_classification_prompt()` naming |
| A5 | dataset-and-prompt | Prompt v0 for classification | v0 has `rules=[]`; drop 3–5 rule count for v0 |
| A6 | classifier | Classifier preserves error handling policy | absent-rules clause → zero-shot baseline wording |

Root cause R2 (component stripping, commits 3c8d7a2/f8950f9/75c04d9):
`VersionComparator` reduced to `decide()`; orchestrator owns runs/metrics;
`CandidateScorer`/`ClassificationPolicy`/`ClassifyInput`/`decision.py` deleted;
`AsyncTask` batch methods and `detect_backend` removed; cross-task-transfer
deleted (see V-TRANS-3); `ThesisExtraction` moved to infra extractor.

| # | Spec | Requirement | Sync to |
|---|---|---|---|
| A7 | stage2-version-comparator | Load new/active version, Run on dev, Compute metrics | orchestrator performs these; comparator = `decide()` only |
| A8 | stage2-version-comparator | Compare metrics | only decision metric + tie-breaker evaluated |
| A9 | stage2-version-comparator | Next candidate on rollback | accumulation stops on rollback; next candidate next cycle |
| A10 | stage2-version-comparator | Decision artifact / Changed decisions / Logging of decision | not produced by comparator (see V-CYCLE-6 for report-level decision) |
| A11 | stage2-version-comparator | Domain service decision rule with tie-breaker | `domain/services/decision.py` deleted; tie-breaker lives in `comparator.decide` (but see V-CYCLE-1) |
| A12 | llm-logit-scorer | Concrete CandidateScorer implementation / Multi-candidate API / Batch scoring port | superseded by classifier spec: `Classifier.classify(text, rules)`, candidates injected at construction |
| A13 | llm-logit-scorer | Tokenization boundary determination | boundary via full-sequence prefix verification (per llama-server spec) |
| A14 | llm-logit-scorer | Application-facing scoring error | `CandidateScoringError` lives in `domain/errors/scoring.py` (never was in application) |
| A15 | llm-logit-scorer | Batched model inference (1 forward call) | llama-server: concurrent per-candidate-token HTTP requests (per llama-server spec) |
| A16 | judgment-prompt | Prompt construction remains in infrastructure | naming: `Classifier.classify` → `LLMLogitClassifier` → `ClassifierPromptBuilder` |
| A17 | judgment-prompt | Prompt compatible with batch scoring | server-era: per-candidate-token forced generation; no in-process batching |
| A18 | judgment-prompt | Semantic classification criteria preserved | hardcoded semantic prefix replaced by active rules (2ce2339) |
| A19 | stage1-baseline-runner | Batch execution | `asyncio.gather`+Semaphore concurrency; no AsyncTask batch methods |
| A20 | dataset-and-prompt | Thesis extraction schema | schema lives in `infrastructure/llm/extractor.py` |
| A21 | dataset-and-prompt | Classification schema | generative classification deleted (0b0f32b); remove requirement |
| A22 | dataset-and-prompt | Fixed extraction prompt | file-based prompt without version/hash (superseded by system-prompt-resources) |
| A23 | llm-transport-client | Call-type wrappers / Rich-result batch methods | `extract_theses_many`, `extract_theses_many_detailed`, `detect_backend` removed (f8950f9) |
| A24 | llm-transport-client | Smoke test readiness gate | script lives at `tests/integration/test_smoke.py` |
| A25 | llm-server-launcher | Shared config schema with client | `asyncTask.py` → `AsyncTask` in `infrastructure/llm/client.py` |
| A26 | architecture | Non-goals | `AsyncTaskClient` → `AsyncTask` naming |
| A27 | candidate-scoring-validation | Candidate configuration is cycle-wide | reword: candidates flow from composition root into both classifier and `RunCycleDeps` (circular letter unsatisfiable) |
| A28 | stage1-metrics | Load run artifact | drop `confidence` field; `predicted_decision` checked for ok-records only (spec self-inconsistency resolved to newer reqs) |
| A29 | stage1-metrics + dataset-and-prompt | Holdout MUST NOT participate in accept/rollback | holdout-driven decisions are the newer deliberate design (ae8067e), embraced by orchestrator spec; sync both stale clauses |
| A30 | stage3-cycle-orchestrator | Dependency object / Composition root wiring | `RunCycleDeps` actual fields: classifier, extractor, candidates, stop_criteria_deps, … (no dataset_repository/embedding_client) |
| A31 | stage3-cycle-orchestrator | Candidate queue | queue replaced (not appended) each cycle; fix spec self-contradiction to match code+test |
| A32 | stage3-cycle-orchestrator | Accept updates active version | `in_prompt` never cleared (documented rationale: prevent re-proposal) |
| A33 | stage3-cycle-orchestrator | Cycle/cluster versioning | baseline version id is `classify-v0` (not `v0`) |
| A34 | stage3-cycle-orchestrator | Final summary (stop-reason enumeration) | 9-value `VALID_STOP_REASONS` superset (stop-criteria reasons); fix internal inconsistency |
| A35 | stage3-prompt-composer-v2 | Load rule-candidate artifact | validation relocated upstream to selector; composer receives dicts |
| A36 | stage4-teacher-refinement | (none — synced in Phase 0) | — |

### B. Auto verdicts — spec is right → fix code/tests

| # | Spec | Gap | Fix |
|---|---|---|---|
| B1 | candidate-scoring-validation | no test: varying `Judgment.score` without changing selection must not change metrics | add integration/unit test (T-TEST-1) |
| B2 | candidate-scoring-validation | no test: non-blocking scoring boundary (event-loop progress during blocked model call) | add test with fake blocking adapter (T-TEST-2) |
| B3 | candidate-scoring-validation | forbidden-symbol scan omits bare `classify_many` from full-production scan | add to `FORBIDDEN_SYMBOLS` (T-TEST-3) |
| B4 | — | `ruff format` drift on 106 files | run `ruff format`; pin ruff version (T-FMT) |
| B5 | config.toml | stale comments referencing deleted scripts (`asyncTask.py`, `dataset.py`) | clean comments (T-CONFIG-1) |
| B6 | tests/contract | `test_architectural_invariants.py` docstring claims an application-vs-infrastructure-classifier check that the file does not contain | add the check or fix docstring — decided by V-ARCH-1 outcome (T-TEST-4) |

### C. Escalations — pending owner verdict

Presented in batches (ARCH → SCORE → CYCLE → STAGE1 → TRANS). Each entry:
spec position, code fact, evidence, options. Verdicts recorded in Section E.

**Batch ARCH**

- **V-ARCH-1 — application→infrastructure layering violations.**
  Spec (architecture, "Layered project structure"): application SHALL be
  independent of infrastructure; linter SHALL forbid it. Code: 5 application
  modules import infrastructure (`report.py:21`, `composer.py:38`,
  `thesis_analyzer.py:32` → private `_atomic_write`; `runner.py:38` →
  `infrastructure.data.writer`; `runner.py:252,278` → `run_metadata`).
  `application-isolated` linter contract forbids only `interfaces`, so this
  passes silently. Introduced by f8950f9/1a243fc with no recorded intent to
  change the rule. Options: (a) fix code — extract atomic-write into a
  shared utility (e.g. domain or a public application service), route
  run-repository writes through the port, add `dynamic_prompt_core.infrastructure`
  to the contract's `forbidden_modules`; (b) accept as pragmatic exception —
  amend spec + linter to permit `application → infrastructure.storage/data`.
- **V-ARCH-2 — dependency-inversion debt at the LLM seam.**
  Spec (architecture, "Dependency inversion"): concrete implementations SHALL
  NOT leak into application; ports remain pure. Code: `AsyncTask.extract_theses`
  requires a leading `aiohttp.ClientSession` param absent from the `LLMClient`
  port — silenced by `# type: ignore[arg-type]` in `main.py:171`;
  `steps.py:512-514` reads private `_tokenizer` attr (self-incriminating
  comment); `steps.py:228` uses `cast(Any, deps.llm_client)`; application use
  cases create/manage `aiohttp.ClientSession` directly (runner.py:24,505;
  composer.py:25,465); infrastructure implements ports without importing them.
  Options: (a) fix code — align `AsyncTask` to the port (self-managed session),
  expose tokenizer via port method, drop casts; (b) partial — fix port
  signature + tokenizer exposure, keep session management where it is;
  (c) amend spec to bless current shape.
- **V-ARCH-3 — CLI signal handling.**
  Spec (architecture, "Graceful shutdown"): entry points SHALL handle
  SIGINT/SIGTERM via controlled cancellation. Code: server entrypoint conforms
  (`interfaces/server/main.py:51-58`); CLI has none — SIGTERM kills it
  mid-cycle, skipping the mandatory cleanup in `finally` (main.py:328-336).
  Options: (a) add SIGINT/SIGTERM handling to CLI (state dump survives
  Ctrl-C/SIGTERM); (b) scope the requirement to the server entrypoint only.

**Batch SCORE**

- **V-SCORE-1 — score semantics: ranking signal vs softmax probability.**
  Spec (candidate-scoring-contracts, "Score is an opaque ranking signal") +
  README: score is NOT a probability/confidence. Code: `Judgment` docstring
  asserts "probability in [0,1] derived from softmax… sums to 1.0"
  (judgment.py:13-17); classifier stores softmax probabilities as scores
  (classifier.py:132-137, deliberate 2bb49c3 "binary softmax scoring");
  judgment-prompt spec forbids softmax (stale). Behavior conforms (any finite
  score accepted; ranking unchanged). Options: (a) keep softmax storage, fix
  the docstring, sync both specs to describe softmax as stored convenience;
  (b) remove softmax — store raw mean log-prob, sync specs minimally;
  (c) fix docstring only, leave specs as-is (contradiction remains).
- **V-SCORE-2 — llama-server timeouts not retried.**
  Spec (llama-server-logit-scorer, "HTTP error translation and transient
  retry"): connection refused, disconnect, AND timeout SHALL be retried.
  Code: only `ServerDisconnectedError`/`ClientConnectorError` retried (3×,
  0.5s); timeouts raise immediately (llama_server_model_adapter.py:155-164,
  tokenizer_adapter.py:113-122). Options: (a) add timeouts to the retry loop;
  (b) amend spec: timeouts fail fast (deliberate — timeout already bounds
  latency).
- **V-SCORE-3 — rule validation at the classifier boundary.**
  Spec (classifier "Validation at classifier boundary" + classification-rules
  "Rule count constraint"): classifier SHALL validate rules (3–5) before
  scoring. Code: no validation in `LLMLogitClassifier.classify`; validation
  lives upstream (`classification_rules_from_artifact` in runner; composer
  enforces `max_rules=5` at compose time only). Options: (a) add validation to
  the classifier (reject >max rules at boundary); (b) sync specs — validation
  is upstream's job, classifier trusts `ClassificationRules`.

**Batch CYCLE**

- **V-CYCLE-1 — accept-on-tie semantics (archives contradict).**
  Spec (orchestrator "Run a single round" + comparator "Decision rule"):
  accept only on STRICT improvement; tie → rollback; tie-breaker SHALL NOT
  participate. Code: tie + tie-breaker strictly higher → ACCEPT
  (comparator.py:97-103, deliberate 3c8d7a2 same day the spec was synced
  aaf6951 — archives contradict each other). Affects greedy accumulation:
  with (a) a tie-promoting candidate can be accepted. Options: (a) keep code,
  sync both specs (tie-breaker participates in decision); (b) revert code to
  strict improvement (tie → rollback), keep specs.
- **V-CYCLE-2 — unrecoverable_error never recorded.**
  Spec (orchestrator "Error handling"/"Final summary"): cycle stops with
  reason `unrecoverable_error`, summary written. Code: exception re-raised
  (run_cycle.py:336-341), no summary, CLI exits 1; the reason string is dead.
  Options: (a) catch, write summary with `unrecoverable_error`, re-raise/exit;
  (b) sync spec: unrecoverable errors propagate, no summary.
- **V-CYCLE-3 — `stop_on_first_error` never built.**
  Spec (orchestrator "Load configuration"/"Configurable parameters"): config
  key with stop-immediately behavior. Code: no such key anywhere. Options:
  (a) implement; (b) remove from spec.
- **V-CYCLE-4 — silent v0 fallback on missing active prompt.**
  Spec (orchestrator "Load initial state"): raise naming the missing active
  prompt. Code: silent fallback to `CLASSIFICATION_PROMPT_V0` with
  `log.debug` (run_cycle.py:168-169; also `prompt_start="best"` + empty store
  → v0, run_cycle.py:106-111). Options: (a) raise (fail fast — a cycle
  silently starting from v0 can overwrite lineage); (b) sync spec: fallback
  is intended (PromptStore seeds v0 anyway).
- **V-CYCLE-5 — stop-criteria effectively OFF + dead code.**
  Spec (orchestrator "Content-based stop evaluation", synced Phase 0): run_cycle
  SHALL call `evaluate_stop_criteria` each round. Code: call is conditional on
  `deps.stop_criteria_deps` (run_cycle.py:257); CLI wires it only when
  `[stop_criteria]` exists in config.toml (main.py:159-168) — **the section is
  absent from shipped config.toml**, so the feature never runs;
  `total_teacher_tokens=0` hardcoded (run_cycle.py:275); dead fallback
  `"max_rounds_reached"` (run_cycle.py:233) not in `VALID_STOP_REASONS`
  (would crash `write_summary` if reached); spec's hard-limit list
  (`max_rounds`, `max_consecutive_rollbacks`, `candidate_queue_exhausted`)
  does not match code's actual hard limits (`max_cycles`, `no_improvement`).
  Options: (a) make it real — wire by default, add `[stop_criteria]` section
  to config.toml, track teacher tokens, remove dead fallback, sync spec
  terminology to cycles; (b) keep optional-off — document in spec, fix dead
  fallback + terminology only.
- **V-CYCLE-6 — report/summary content gaps.**
  Spec (orchestrator "Per-round report"/"Final summary" + comparator
  "Decision artifact"/"Changed decisions"): reports SHALL contain `vN_k`
  intermediate attempts, accepted cluster ids, changed decisions; decision
  artifacts SHALL be persisted. Code: none of these exist —
  `changed_decisions` always `[]` (steps.py:701), no decision artifacts ever
  written (observer globs for them in vain, observer.py:329), summary has no
  changed-decisions field (report.py:119-129). Options: (a) implement report
  fields + decision artifacts (observer lights up); (b) sync specs to current
  content (drop the fields).
- **V-CYCLE-7 — observer shows holdout as dev-after.**
  Spec (cycle-observer "Summarize each round"): key dev metrics before/after.
  Code: `metrics_dev_new` never populated by the pipeline; observer falls back
  to holdout-new with admitting comments (observer.py:444-449,683) — the
  "after" column is holdout mislabeled as dev. Options: (a) populate
  `metrics_dev_new` in the cycle pipeline (dev run of accepted version);
  (b) relabel column + sync spec.
- **V-CYCLE-8 — composer artifact hash + rules shape.**
  Spec (prompt-composer-v2 "Prompt version artifact"): `hash` = sha256 of
  text; `rules` = list of rule texts. Code: hash truncated to 16 hex chars
  (composer.py:382; PromptStore independently stores full sha256 — two hashes
  for the same text); `rules` = list of `{cluster_id, text}` dicts (satisfies
  the spec's own lineage requirement, contradicts its artifact requirement).
  Options: (a) full sha256 in composer artifact + sync rules clause to dicts;
  (b) sync spec to 16-hex + dicts.
- **V-CYCLE-9 — `max_rule_words` default 15 vs 20.**
  Spec: default 15. Code: `DEFAULT_MAX_RULE_WORDS = 20` (composer.py:48),
  config.toml sets 20, formulation prompt hardcodes "at most 20 words".
  Options: (a) sync spec to 20; (b) change code+config+prompt to 15.
- **V-CYCLE-10 — logging-detail gaps cluster (6 minor).**
  Spec demands logged details that code computes but does not log (or vice
  versa): selector counters (selector.py:236-241 → artifact only);
  stop-eval checked-criteria + metric values (evaluate.py:115-124 → triggered
  only); composer counters (composer.py:541-547 → metadata only);
  embedding batch size at startup (thesis_analyzer.py:212 → device only);
  prompt-store log file path (prompt_store.py:482-502); metrics failed-SHARE
  vs count (metrics.py:447-453). Options: (a) add the logging (6 small code
  fixes); (b) trim the spec clauses.
- **V-CYCLE-11 — artifacts bypass the run_repository port.**
  Spec (orchestrator reports "via run_repository"; stop-criteria "via
  run_repository"): persist through the port. Code: `write_report`/
  `write_summary`/`write_stop_decision_artifact` accept the port parameter
  and ignore it, writing via private `_atomic_write`/plain `open()`
  (report.py:21,87; artifact.py:28-57). Options: (a) route through the port
  (add write methods to the port); (b) drop the unused parameter + sync specs.
  Note: folds into V-ARCH-1 if option (a) chosen there.

**Batch STAGE1**

- **V-STAGE1-1 — request log never wired.**
  Spec (baseline-runner "Logging requests"): `log_path` passed to `AsyncTask`,
  all requests logged to jsonl. Code: `AsyncTask` constructed without
  `log_path` in the only composition root (main.py:119) — `_log_writer=None`;
  runner prints a `requests_*.jsonl` path it never wires (runner.py:225-227).
  No request log exists for any past run. Options: (a) wire `log_path`;
  (b) remove the clause + the dead path computation.
- **V-STAGE1-2 — thesis bank: no resume, no cross-run accumulation.**
  Spec (thesis-analyzer "Thesis bank dump"/"Thesis bank"): dump SHALL be
  loadable to resume without recomputing embeddings/clusters; `frequency`
  accumulates across all runs. Code: no reload path — every cycle rebuilds
  the bank and recomputes embeddings (steps.py:129-161); only
  `in_prompt` flags carry over (thesis_analyzer.py:585-605). Options:
  (a) implement bank reload (perf win: embeddings are the expensive step);
  (b) sync spec: bank is per-cycle, flags carry over.

**Batch TRANS**

- **V-TRANS-1 — run-repository empty-list write is non-atomic.**
  Spec (run-repository "Atomic result persistence"): SHALL NOT open target
  for direct in-place overwrite. Code: empty `save_results([])` truncates the
  target in place (run_repository.py:114-117), pinned by
  `test_save_empty_results`. Options: (a) make empty write atomic (or skip
  write); (b) spec exception for empty payload.
- **V-TRANS-2 — `teacher.use_refined` is dead config.**
  Spec (teacher-refinement "Configurable use of refined theses", synced in
  Phase 0): `use_refined=false` → downstream uses `theses_raw_tiny`. Code: no
  consumer — key read by nothing (only config.toml:285); downstream
  unconditionally takes refined theses (steps.py:338); consumer removed in
  38078ec the same day the scenario was written, never re-implemented. A user
  setting `use_refined=false` silently gets refined theses anyway. Options:
  (a) implement the consumer (respect the flag in cycle steps); (b) remove
  key from spec + config.toml.
- **V-TRANS-3 — cross-task-transfer: spec without code (confirm retirement).**
  Spec: 13 requirements, active in canon. Code: zero — implementation deleted
  as dead code in 3c8d7a2 ("1087 lines, imported non-existent
  run_cycle.orchestrator"). Auto-verdict per policy would be `retire-spec`;
  confirming because it removes a whole capability. Options: (a) retire —
  delete `openspec/specs/stage4-cross-task-transfer/`, note as roadmap in
  README if desired; (b) keep spec as unimplemented roadmap (annotate
  Purpose: "not implemented").

### D. Unverifiable / open

| # | Item | Note |
|---|---|---|
| D1 | architecture: No immutable files / Refactoring policy / Contract evolution / Granular modules / Architecture principle | policy statements, no testable content |
| D2 | architecture: CI enforcement | **no CI pipeline exists in the repo at all** — open question for the owner (out of audit scope) |
| D3 | llama-server: logit_bias equivalence (bias does not distort log-prob) | validated live against llama.cpp 0.5.0 per module docstring; not reproducible from repo |
| D4 | system-prompt-resources: packaging after `pip install` | packaging config verified; install not performed |
| D5 | server launcher: live spawn/port-release behavior | no live server in audit; inspection only |
| D6 | teacher judgment quality | depends on live teacher model |
| D7 | candidate-scoring-validation: application test transitively imports `transformers` via transport package `__init__` | low-confidence minor; consider lazy imports if cared about |
| D8 | config.toml `[stop_criteria]` section absent | folded into V-CYCLE-5 |

### E. Verdict record

| Verdict | Item | Decision | Date |
|---|---|---|---|
| V-ARCH-1 | application→infrastructure layering | **(a) fix code** — shared atomic-write utility, writes via port, `infrastructure` added to `application-isolated` forbidden_modules | 2026-10-29 |
| V-ARCH-2 | dependency-inversion debt | **(a) fix code** — AsyncTask owns its session, tokenizer exposed via port method, remove `type: ignore`/`cast(Any)`/private-attr reach | 2026-10-29 |
| V-ARCH-3 | CLI signal handling | **(a) fix code** — add SIGINT/SIGTERM controlled cancellation to CLI entrypoint | 2026-10-29 |
| V-SCORE-1 | score semantics | **(a) keep softmax** — fix `Judgment` docstring (stop claiming probability), sync contracts + judgment-prompt specs: softmax is stored convenience, contract remains opaque ranking signal | 2026-10-29 |
| V-SCORE-2 | timeout retry | **(b) sync spec** — timeouts fail fast (no retry); retry covers disconnect/connector errors only | 2026-10-29 |
| V-SCORE-3 | rule validation | **(b) sync spec** — validation is upstream's job (validating constructor + composer max_rules); classifier trusts `ClassificationRules` | 2026-10-29 |
| V-CYCLE-1 | accept-on-tie | **(a) keep code, sync specs** — tie + tie-breaker strict improvement → accept; decision rule clauses updated in orchestrator + comparator | 2026-10-29 |
| V-CYCLE-2 | unrecoverable_error | **(a) fix code** — write summary with `unrecoverable_error`, then re-raise | 2026-10-29 |
| V-CYCLE-3 | stop_on_first_error | **(b) sync spec** — remove the never-built key/behavior from spec | 2026-10-29 |
| V-CYCLE-4 | silent v0 fallback | **(a) fix code** — raise naming the missing active prompt (store seeds v0 on fresh runs, so fresh runs unaffected) | 2026-10-29 |
| V-CYCLE-5 | stop-criteria off | **(b) keep optional-off** — document conditional wiring in spec; fix dead `max_rounds_reached` fallback and rounds→cycles terminology regardless | 2026-10-29 |
| V-CYCLE-6 | report content gaps | **(b) sync specs** — reports match current content; remove observer's dead decision-artifact glob | 2026-10-29 |
| V-CYCLE-7 | observer dev-after | **(b) relabel** — show holdout honestly; sync spec; no extra dev run | 2026-10-29 |
| V-CYCLE-8 | composer hash + rules shape | **(a) fix code** — full sha256 in composer artifact; sync rules clause to `{cluster_id, text}` dicts | 2026-10-29 |
| V-CYCLE-9 | max_rule_words | **(a) sync spec** — default is 20 | 2026-10-29 |
| V-CYCLE-10 | logging cluster | **(a) fix code** — add the 6 logging details (selector counters, stop-eval detail, composer counters, embedding batch size, store file path, failed share) | 2026-10-29 |
| V-CYCLE-11 | port bypass | **(a) fix code** — route report/summary/stop-decision writes through the run_repository port (part of V-ARCH-1 work) | 2026-10-29 |
| V-STAGE1-1 | request log | **(a) fix code** — wire `log_path` into `AsyncTask` construction in the composition root | 2026-10-29 |
| V-STAGE1-2 | thesis bank resume | **(b) sync spec** — bank is per-cycle; only `in_prompt` flags carry over | 2026-10-29 |
| V-TRANS-1 | empty-list atomicity | **(a) fix code** — empty `save_results` writes atomically (or skips) | 2026-10-29 |
| V-TRANS-2 | teacher.use_refined | **(a) fix code** — implement the consumer: respect `use_refined` in cycle steps | 2026-10-29 |
| V-TRANS-3 | cross-task-transfer | **(a) retire spec** — delete `openspec/specs/stage4-cross-task-transfer/`; capability returns via a new change if revived | 2026-10-29 |

All 21 escalations resolved. Auto-verdicts (Sections A/B) stand as recorded.

## Implementation status (2026-10-29)

All verdicts implemented in this change:

- Spec syncs: Sections A1–A35 applied across 19 specs; `stage4-cross-task-transfer`
  retired (V-TRANS-3a); comparator spec rewritten to the `decide()`-only component.
- Code fixes: `domain/io/atomic.py` shared helper; `RunRepository.save_artifact`
  port method (reports/summaries/stop-decisions routed through the port);
  `DatasetRepository` port + `FileDatasetRepository`; `run_metadata` moved to
  application; `application-isolated` linter contract now forbids
  `infrastructure` (6/6 KEPT); AsyncTask owns its HTTP session (`aclose`),
  `count_tokens` on the LLMClient port, `type: ignore`/`cast(Any)`/`_tokenizer`
  reach removed; Extractor port lost the session leak; CLI SIGINT/SIGTERM
  controlled cancellation; `unrecoverable_error` summary written before
  re-raise; missing active prompt raises (no silent v0 fallback); dead
  `max_rounds_reached` fallback fixed; observer's dead decision-artifact glob
  removed and holdout metrics labeled honestly; full sha256 in prompt
  artifacts; 6 logging details added; request log wired
  (`requests_{run_id}.jsonl`); `teacher.use_refined` consumer implemented;
  `save_results([])` atomic; `Judgment` docstring de-probabilized;
  `classify_many` in the full forbidden-symbol scan.
- Tests: metric-invariance test and non-blocking event-loop test added;
  architectural-invariants docstring corrected; `ruff format` applied
  (113 files) and ruff pinned `>=0.14,<0.17`; config.toml stale comments
  cleaned.
- Verification: lint-imports 6/6 KEPT; mypy clean (115 files); ruff check
  clean; ruff format clean; pytest 440 passed; `openspec validate --all`
  27 specs, 0 invalid.
- Remaining: archive this change (owner's call).
