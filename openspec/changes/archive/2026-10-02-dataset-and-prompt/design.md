# Design

## Context

See proposal.md for motivation. The project currently has `asyncTask.py`
(transport + parsing + logging, spec `llm-transport-client`) and a raw
`data/train.csv` (8561 rows, columns `id,keyword,location,text,target` where
`target` is `0`/`1`). `smoke_test.py` uses throwaway placeholder Pydantic models.
There is no split, no prompt store, and no real schemas. Stage 1 (runner,
metrics) is the next stage and needs this starting point.

The existing `AsyncTask.classify` / `AsyncTask.extract_theses` wrappers accept a
`system_prompt` and a Pydantic `model` per call, so this change only needs to
supply the prompts and schemas they consume — no changes to `asyncTask.py`.

## Goals / Non-Goals

**Goals:**

- Provide a dataset loader/preparer with validation and a reproducible
  dev/holdout/ambiguous split.
- Provide prompt v0 (classification) and the fixed extraction prompt as
  versioned, hashed artifacts rendered by a dependency-free templating helper.
- Provide the `ClassificationResult` and `ThesisExtraction` Pydantic schemas for
  structured output via the existing client wrappers.
- Persist the prepared dataset as a reproducible artifact.

**Non-Goals:**

- Stage 1 runner, metrics, and analyzer (separate change).
- Stage 2+ automatic prompt optimization (separate change).
- Loading from external sources (API, DB, cloud).
- Per-attempt or per-example logging of dataset rows (the transport layer
  already logs requests).

## Decisions

### D1: Module layout — `dataset.py`, `prompts.py`, `schemas.py`

```
dataset.py   # load, validate, split, persist
prompts.py   # v0 classification prompt, fixed extraction prompt, renderer
schemas.py   # ClassificationResult, ThesisExtraction (Pydantic v2)
```

**Rationale**: Small, focused modules matching the three concerns. The project
is currently flat (`asyncTask.py`, `server_launcher.py`, `smoke_test.py`), so a
flat layout with three new files fits. `schemas.py` is separate from `prompts.py`
because the client imports schemas by type on every call, while prompts are
fetched by version.

**Alternative**: A single `dataset_and_prompt.py`. Rejected — mixes three
unrelated import surfaces and grows unwieldy once Stage 1 adds a runner.

### D2: Dataset format — jsonl primary, parquet optional via lazy import

The loader detects format by extension. jsonl is read with the standard library
(`json` per line). parquet is read via `pyarrow`/`pandas` imported lazily, so the
hard dependency stays on jsonl and parquet is only required when a `.parquet`
path is given.

**Rationale**: The source data is CSV today, but the spec requires jsonl/parquet
I/O. jsonl keeps the standard-library-only path and matches the transport
layer's logging format. parquet support is added without forcing a new
dependency on every user.

**Alternative**: pandas for both. Rejected — pulls in a heavy dependency for
jsonl, which the standard library handles.

### D3: Source CSV → jsonl conversion is out of scope; loader reads jsonl/parquet

The loader reads jsonl or parquet as the spec requires. Converting the existing
`data/train.csv` to jsonl is a one-time data-prep step performed before the
first run (documented in tasks), not a loader responsibility. The CSV columns
map to the required fields: `id`→`id`, `text`→`text`, `target`→`label`.

**Rationale**: Keeping CSV parsing out of the loader avoids a third format path
and keeps the loader's validation surface small. The spec names jsonl/parquet
only.

### D4: Split via deterministic shuffle on stable sort key

```
records = sorted(non_ambiguous, key=lambda r: r.id)   # stable input order
rng = random.Random(seed)
rng.shuffle(records)
cut = int(len(records) * holdout_ratio)
holdout = records[:cut]
dev = records[cut:]
```

Ambiguous records (`ambiguous=true`) are removed before the shuffle and placed
in a separate `ambiguous` group. The split is reproducible because the input is
sorted by `id` (deterministic order) before `random.Random(seed).shuffle`.

**Rationale**: Sorting by `id` before shuffling removes any dependence on file
row order, making reproducibility robust to rewrites that reorder rows.
`random.Random` with a fixed seed is deterministic across Python versions for a
given seed (CPython guarantee).

**Alternative**: `sklearn.model_selection.train_test_split`. Rejected — adds a
heavy dependency for a shuffle + slice, and its shuffle is the same
`random.Random` underneath.

### D5: holdout_ratio from config, default 0.2

`holdout_ratio` is read from `[dataset].holdout_ratio` in `config.toml` when
present, defaulting to `0.2`. `seed` is read from `[dataset].seed`, defaulting
to a fixed constant (e.g. `42`). The source path is `[dataset].path`.

**Rationale**: Centralizing the three knobs in `config.toml` (already read by
`asyncTask.py`) keeps configuration in one place and makes the split
reproducible by configuration rather than by call-site arguments.

### D6: Prompt as a layered dataclass + dependency-free renderer

```
@dataclass
class PromptLayer:
    role: str
    task: str
    rules: list[str]
    output_contract: str
    fallback: str

def render(p: PromptLayer) -> str:   # plain f-string join, no Jinja/mustache
    ...

@dataclass
class PromptArtifact:
    version: str
    layers: PromptLayer
    text: str          # rendered
    sha256: str        # hash of text
```

`render` joins the five layers with section separators. The hash is
`sha256(text)[:16]`, matching the transport layer's `system_prompt_hash` style.
Assembly checks `3 <= len(rules) <= 5` and raises otherwise.

**Rationale**: A dataclass + f-string renderer has zero external dependencies and
is trivially testable. Storing the rendered text and hash on the artifact means
callers pass `artifact.text` to `AsyncTask.classify(system_prompt=...)` and can
log `artifact.sha256` for traceability.

**Alternative**: Jinja2 templates. Rejected — a five-layer prompt does not need
a template engine, and the spec forbids external dependencies for rendering.

### D7: Extraction prompt is a constant PromptArtifact, not layered

The fixed extraction prompt is stored as a single `PromptArtifact` with a
constant `version` (e.g. `"extract-v0"`) and a frozen text body. It is not
assembled from layers and has no rules-count constraint.

**Rationale**: The spec requires it to be fixed and non-optimized. A constant
artifact communicates "do not mutate" more clearly than a layered prompt that
happens to never change.

### D8: Schemas use Pydantic v2 field validators

```python
class ClassificationResult(BaseModel):
    decision: Literal[0, 1]
    confidence: conint(ge=0, le=100)

class ThesisExtraction(BaseModel):
    theses: list[Annotated[str, StringConstraints(min_length=...)]]
    # 3–5 theses, each 2–6 words enforced via a field_validator
```

Word-count (2–6 words per thesis) is enforced with a `@field_validator` that
splits on whitespace, since Pydantic cannot express "words" natively. Thesis
count (3–5) is `min_length=3, max_length=5` on the list.

**Rationale**: `Literal[0, 1]` rejects `decision=2` at validation time.
`conint(ge=0, le=100)` bounds confidence. The `field_validator` for word count
is the only non-declarative piece and is unavoidable given "words" is not a
Pydantic primitive. These schemas are passed directly to
`AsyncTask.classify(model=ClassificationResult)` and
`AsyncTask.extract_theses(model=ThesisExtraction)`.

### D9: Artifact filename includes seed and timestamp

```
data/prepared_<seed>_<YYYYMMDDTHHMMSSZ>.jsonl
```

The artifact records `split` (`dev`/`holdout`/`ambiguous`) and `notes` (free
text, e.g. the source filename and ratio) per record. Reload re-derives the
three groups from the `split` field.

**Rationale**: The spec requires the filename to carry the seed and a timestamp.
Including the seed makes artifacts comparable across runs with different seeds;
the timestamp prevents overwrite collisions.

## Risks / Trade-offs

- **[CSV→jsonl conversion step]** The loader reads jsonl/parquet, but the
  current source is CSV. → Mitigation: tasks include a one-time conversion step
  (a small script or inline `csv`→`jsonl`), and the README/tasks document it.
  The loader itself stays format-focused.

- **[parquet dependency is optional]** A user who passes a `.parquet` path
  without `pyarrow`/`pandas` installed gets an `ImportError` at call time, not
  at install time. → Mitigation: the error message names the missing package and
  the pip install command. jsonl remains the zero-dependency default.

- **[Word-count validation is locale-naive]** Splitting on whitespace counts
  tokens, not linguistic words; punctuation attached to a word counts as one.
  → Mitigation: acceptable for thesis extraction where theses are short phrases.
  If the model emits punctuation-heavy theses, revisit in Stage 2+.

- **[random.Random cross-version stability]** `random.Random(seed).shuffle` is
  stable across CPython versions for a given seed, but not guaranteed across
  alternative Python implementations (PyPy). → Mitigation: the project targets
  CPython. Document the invariant.

- **[holdout_ratio edge cases]** With a small dataset, `int(len * ratio)` can
  produce a 0-size holdout or leave dev empty. → Mitigation: assert
  `0 < cut < len(records)` after computing the cut and raise a clear error
  otherwise.
