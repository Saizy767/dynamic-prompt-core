# Spec Delta

## Purpose

Define the GGUF-backed logit provider that generates raw next-token logits from
a local GGUF model file using `llama-cpp-python`, implementing the existing
scoring seams so the shared classifier and evaluator consume GGUF models
without Hugging Face Transformers.

## ADDED Requirements

### Requirement: GGUF provider implements existing scoring seams
The GGUF model adapter SHALL implement the `BatchedCausalLanguageModel` seam
and the GGUF tokenizer adapter SHALL implement the `BatchTokenizerAdapter`
seam defined by `llm-logit-scorer`. The shared `LLMLogitCandidateScorer`,
`LogitScorer`, and prompt-construction components SHALL remain unchanged. The
GGUF provider SHALL NOT contain domain-specific acceptance rules, candidate
scoring logic, or metric calculation; those remain owned by the existing
shared components. The GGUF provider SHALL NOT introduce a parallel scoring
abstraction.

The GGUF model adapter SHALL match the existing seam contract exactly: return
type (`BatchedLogits` containing `SequenceLogits`), semantic shape
`[batch_size, sequence_length, vocabulary_size]`, element representation
(plain Python floats, identical to the existing adapter which converts
tensors via `.tolist()`), per-item ordering, causal logit alignment, and
error behavior. The adapter SHALL NOT substitute a different array/tensor
type or device placement for the existing contract. The `LogitScorer` SHALL
receive structurally identical semantics from both the Hugging Face and GGUF
providers.

#### Scenario: Model adapter satisfies batch seam
- **WHEN** the GGUF model adapter is inspected
- **THEN** it implements `BatchedCausalLanguageModel.forward_batch` returning `BatchedLogits` with semantic shape `[B, S, V]`

#### Scenario: Tokenizer adapter satisfies batch seam
- **WHEN** the GGUF tokenizer adapter is inspected
- **THEN** it implements `BatchTokenizerAdapter.encode_batch` returning `BatchTokenization` with per-item prefix/candidate boundaries

#### Scenario: No duplicated scoring logic
- **WHEN** the GGUF provider is inspected
- **THEN** it contains no candidate scoring, metric calculation, or acceptance logic

#### Scenario: Output structure matches existing contract
- **WHEN** the GGUF model adapter returns logits
- **THEN** the return type, shape, element representation, per-item ordering, and causal alignment are identical to the existing Hugging Face adapter contract

#### Scenario: LogitScorer receives identical structural semantics
- **WHEN** the `LogitScorer` consumes logits from the GGUF provider
- **THEN** it receives the same structural semantics (shape, element type, position alignment) as from the Hugging Face provider

### Requirement: GGUF model loaded from local path once per instance
The provider SHALL load the model from a configured local `model_path`
pointing to a GGUF file. The model SHALL be initialized once per provider
instance and SHALL NOT be reloaded for each dataset example or scoring
invocation. The provider SHALL NOT download models automatically. A missing
or unreadable model file SHALL fail during initialization with an actionable
error.

#### Scenario: Model loaded from local path
- **WHEN** the provider is initialized with a valid `model_path`
- **THEN** the GGUF model is loaded from that local file

#### Scenario: Model initialized once
- **WHEN** the provider evaluates multiple scoring invocations
- **THEN** the model is loaded a single time at initialization, not per invocation

#### Scenario: Missing model file fails at init
- **WHEN** the provider is initialized with a `model_path` that does not exist or is unreadable
- **THEN** initialization raises an actionable error before any scoring occurs

#### Scenario: No automatic download
- **WHEN** the provider is initialized
- **THEN** it does not download a model from a remote source

### Requirement: GGUF configuration parameters
The provider SHALL accept and validate these configuration parameters:
`model_path` (required), `n_ctx`, `n_batch`, `n_threads`, `n_gpu_layers`,
`seed`, `verbose`, and `logits_all`. The context size (`n_ctx`) SHALL be
validated against the effective prompt length plus any continuation tokens
needed for classification. The provider SHALL NOT silently change
model-specific parameters to accommodate an invalid configuration. Invalid
configuration values SHALL fail during initialization.

#### Scenario: Required model path validated
- **WHEN** the provider is initialized without a `model_path`
- **THEN** initialization raises an error naming the missing parameter

#### Scenario: Context size validated against prompt length
- **WHEN** the effective prompt plus continuation tokens exceeds `n_ctx`
- **THEN** the provider raises a context-overflow error rather than silently truncating

#### Scenario: Invalid configuration fails at init
- **WHEN** the provider is initialized with an unsupported parameter value
- **THEN** initialization raises an error before any scoring occurs

### Requirement: Optional GGUF dependency isolation
`llama-cpp-python` SHALL be an optional dependency. Installing or importing
the core package SHALL NOT require the GGUF runtime when the Hugging Face
backend is selected. The GGUF adapter SHALL fail with a clear, actionable
configuration error if the optional dependency is missing. The optional
dependency SHALL be version-pinned in the project dependency manifest.

#### Scenario: Core install does not require GGUF runtime
- **WHEN** the core package is installed without the optional GGUF dependency
- **THEN** importing the core package succeeds and the Hugging Face backend remains usable

#### Scenario: Missing optional dependency fails clearly
- **WHEN** the GGUF backend is selected but `llama-cpp-python` is not installed
- **THEN** initialization raises an actionable error naming the missing dependency and how to install it

#### Scenario: Dependency version pinned
- **WHEN** the dependency manifest is inspected
- **THEN** `llama-cpp-python` is pinned to a specific version in an optional dependency group

### Requirement: GGUF tokenizer is authoritative
The GGUF tokenizer SHALL be authoritative for the GGUF model. The provider
SHALL NOT reuse Hugging Face token IDs with a GGUF model unless tokenizer
compatibility has been explicitly established. The provider SHALL preserve
the exact prompt text supplied by the existing application, except for
documented model-specific chat-template rendering. If the existing classifier
uses plain completion prompts, the GGUF implementation SHALL NOT silently
convert them into chat messages. The selected prompt format SHALL be included
in experiment metadata.

#### Scenario: GGUF tokenizer used for token IDs
- **WHEN** the GGUF provider tokenizes input
- **THEN** token IDs come from the GGUF model's tokenizer, not a Hugging Face tokenizer

#### Scenario: Prompt text preserved
- **WHEN** the GGUF provider receives a prompt
- **THEN** the exact prompt text is preserved except for documented chat-template rendering

#### Scenario: No silent chat conversion
- **WHEN** the existing classifier uses plain completion prompts
- **THEN** the GGUF provider does not convert them into chat messages

#### Scenario: Prompt format recorded in metadata
- **WHEN** an evaluation run is recorded
- **THEN** the selected prompt format is included in the experiment metadata

### Requirement: GGUF tokenization boundary compatibility check
The GGUF tokenizer adapter SHALL determine the prefix/candidate boundary by
encoding the full prefix-plus-candidate text with the GGUF tokenizer using
explicitly configured special-token (BOS) behavior, then encoding the prefix
separately using the same BOS and special-token policy, then checking whether
the prefix token sequence is an exact prefix of the full token sequence. If
the prefix sequence is an exact prefix of the full sequence, the candidate
continuation SHALL be the remaining full-sequence tokens. If it is not an
exact prefix, the adapter SHALL use an explicitly defined fallback or reject
the example with a clear compatibility error; it SHALL NOT silently score a
different token sequence than the one produced by encoding the full prompt
text. The adapter SHALL handle an empty candidate without adding BOS twice.
The prefix-is-prefix check is a compatibility check, not a universal solution
to token-boundary ambiguity: if the tokenizer merges tokens across the
boundary such that no unique token-level division reproduces the original
scoring semantics, the project SHALL explicitly define the intended boundary
or reject the configuration. The adapter SHALL NOT assume that separately
encoding the prefix and candidate and concatenating them is equivalent to
encoding the full prompt text.

#### Scenario: Boundary determined by full encoding with prefix-prefix check
- **WHEN** the GGUF tokenizer adapter tokenizes a scoring prompt
- **THEN** it encodes the full prefix-plus-candidate text and verifies the prefix is an exact prefix of the full token sequence

#### Scenario: Candidate continuation is full-sequence remainder
- **WHEN** the prefix token sequence is an exact prefix of the full token sequence
- **THEN** the candidate continuation tokens are the remaining full-sequence tokens after the prefix

#### Scenario: Boundary mismatch rejects or uses explicit fallback
- **WHEN** the prefix token sequence is not an exact prefix of the full token sequence
- **THEN** the adapter rejects the example with a compatibility error or applies an explicitly defined fallback, and does not silently score a different token sequence

#### Scenario: Empty candidate handled without double BOS
- **WHEN** the candidate value is empty
- **THEN** the adapter handles it without adding BOS or special tokens twice

#### Scenario: No silent separate-encoding assumption
- **WHEN** the adapter determines candidate token positions
- **THEN** it does not assume that encoding the prefix and candidate separately and concatenating equals encoding the full prompt text

### Requirement: Context isolation between evaluations
Inference context reuse SHALL NOT allow tokens or cached state from one
classification example to affect another example's score. Each scoring
invocation SHALL be evaluated independently. The provider SHALL reset or
correctly isolate the inference context before each evaluation. Repeated
evaluation of the same input SHALL produce consistent outputs under the
chosen runtime configuration.

#### Scenario: No state contamination between examples
- **WHEN** two different examples are evaluated sequentially on the same provider instance
- **THEN** the second example's score is not affected by the first example's tokens or cached state

#### Scenario: Repeated evaluation is consistent
- **WHEN** the same input is evaluated twice on the same provider instance
- **THEN** both evaluations produce identical scores

#### Scenario: Context reset before evaluation
- **WHEN** the provider begins a new evaluation
- **THEN** the inference context is reset or isolated so prior state does not influence the result

### Requirement: Concurrency safety
Concurrent calls SHALL either use safe context isolation or be serialized
using a documented synchronization mechanism. If parallel evaluation is
supported, its worker model SHALL be explicit. The provider SHALL NOT assume
a single mutable inference context is safe for simultaneous evaluations. The
provider SHALL handle context overflow, inference exceptions, and
cancellation without corrupting the state of subsequent evaluations.

#### Scenario: Concurrent calls do not corrupt state
- **WHEN** multiple scoring calls are made concurrently on the same provider instance
- **THEN** each call returns a correct result independent of the others, or calls are serialized by a documented mechanism

#### Scenario: Cancellation does not corrupt subsequent evaluations
- **WHEN** a scoring call is interrupted or cancelled
- **THEN** subsequent evaluations on the same provider instance produce correct results

#### Scenario: Worker model is explicit
- **WHEN** parallel evaluation is supported
- **THEN** the concurrency/worker model is documented in the implementation

### Requirement: Resource management and shutdown
The provider SHALL define how model instances and inference contexts are
released at shutdown. The model instance SHALL be reused to avoid unnecessary
loading overhead. Resource release SHALL NOT leak file handles, memory, or
GPU resources.

#### Scenario: Model instance reused
- **WHEN** multiple evaluations are performed
- **THEN** the same model instance is reused without reloading

#### Scenario: Resources released at shutdown
- **WHEN** the provider is shut down or garbage-collected
- **THEN** model instances and inference contexts are released without leaking resources

### Requirement: Logit vocabulary and numerical validation
The provider SHALL validate that the output vocabulary dimension matches the
loaded model's vocabulary and that all required token positions are present.
The provider SHALL reject logits containing NaN or infinite values. The
provider SHALL distinguish raw logits from log-probabilities and normalized
scores; it SHALL NOT expose probabilities as if they were raw logits. Logits
SHALL be represented as a one-dimensional array of vocabulary scores for a
single position, or two-dimensional when multiple positions are explicitly
requested. The provider SHALL read the model's logit buffer once per
evaluation and index into it for the required positions; it SHALL NOT
repeatedly access or re-materialize the full-vocabulary logit buffer inside a
loop over candidate token positions. The provider SHALL measure and document
peak memory for realistic context lengths and vocabulary sizes. The provider
SHALL NOT claim partial-position logit reads are supported until a working,
tested implementation proves it against the concrete runtime API.

#### Scenario: Vocabulary dimension validated
- **WHEN** the provider reads logits from the model
- **THEN** the vocabulary dimension matches the loaded model's vocabulary size

#### Scenario: NaN logits rejected
- **WHEN** the model returns logits containing NaN
- **THEN** the provider raises an error rather than passing invalid values to the scorer

#### Scenario: Infinite logits rejected
- **WHEN** the model returns logits containing positive or negative infinity
- **THEN** the provider raises an error rather than passing invalid values to the scorer

#### Scenario: Raw logits distinguished from probabilities
- **WHEN** the provider exposes logit outputs
- **THEN** raw logits, log-probabilities, and normalized scores are clearly distinguished and probabilities are not exposed as raw logits

#### Scenario: Logit buffer read once per evaluation
- **WHEN** the provider extracts logits for multiple candidate positions
- **THEN** it reads the model logit buffer once and indexes into it, not once per position

#### Scenario: Peak memory measured and documented
- **WHEN** the provider is benchmarked
- **THEN** peak memory for realistic context lengths and vocabulary sizes is measured and documented

#### Scenario: No unproven partial-read claim
- **WHEN** the provider documentation is inspected
- **THEN** partial-position logit reads are claimed only if a working tested implementation proves them against the concrete runtime API

### Requirement: Experiment metadata recording
Every evaluation run SHALL record: the backend identifier, model file content
identity (a checksum or equivalent reliable content identity, not the path
alone), GGUF quantization and model metadata where available, runtime and
relevant dependency versions, tokenizer and prompt-template configuration,
context/batch/thread/GPU-offload settings, dataset fingerprint, prompt
version, and classification scoring method. Settings that can change scoring
results (model contents, tokenizer, prompt template, scoring method, dataset)
SHALL be distinguished from non-semantic settings (logging verbosity) in the
recorded identity. Resume operations SHALL verify the recorded content
identity and semantic settings against the current provider; a mismatch in a
result-changing setting SHALL either reject the resume or explicitly record
the run as a new experiment — it SHALL NOT silently continue with a different
configuration. A backend change SHALL NOT implicitly promote a candidate or
bypass existing rollback and acceptance rules.

#### Scenario: Backend identity recorded
- **WHEN** an evaluation run is recorded
- **THEN** the backend identifier is included in the experiment metadata

#### Scenario: Model content identity recorded
- **WHEN** an evaluation run is recorded
- **THEN** a checksum or equivalent content identity (not the path alone), GGUF metadata, and quantization are included

#### Scenario: Runtime and settings recorded
- **WHEN** an evaluation run is recorded
- **THEN** runtime versions, tokenizer config, prompt template, and inference settings are included

#### Scenario: Semantic settings distinguished from non-semantic
- **WHEN** experiment identity is recorded
- **THEN** result-changing settings (model contents, tokenizer, prompt template, scoring method, dataset) are distinguishable from non-semantic settings

#### Scenario: Resume rejects or records mismatch
- **WHEN** a resume operation finds a result-changing setting mismatch
- **THEN** it rejects the resume or explicitly records the run as a new experiment, and does not silently continue with a different configuration

#### Scenario: Backend change does not bypass acceptance
- **WHEN** the backend changes between runs
- **THEN** no candidate is implicitly promoted and existing rollback and acceptance rules remain in effect

### Requirement: Actionable GGUF error handling
The provider SHALL define actionable errors for: missing optional runtime
dependency, missing or invalid GGUF model, unsupported model architecture or
tokenizer configuration, context-window overflow, empty or invalid tokenized
prompt, invalid logits shape or values, invalid label tokenization, inference
failure or timeout, and resource exhaustion. Errors SHALL include enough
context for debugging without logging sensitive input text or secrets by
default. Transient failures MAY use bounded retries only when retrying is
safe; invalid configurations and corrupted model files SHALL NOT be retried
indefinitely.

#### Scenario: Missing dependency error is actionable
- **WHEN** the GGUF runtime dependency is missing
- **THEN** the error names the missing dependency and how to install it

#### Scenario: Context overflow error is actionable
- **WHEN** a prompt exceeds the configured context window
- **THEN** the error names the overflow, the prompt length, and the context size

#### Scenario: Errors exclude sensitive text by default
- **WHEN** an inference error occurs
- **THEN** the error message includes debugging context but does not log sensitive input text or secrets by default

#### Scenario: Invalid configuration not retried indefinitely
- **WHEN** an invalid configuration or corrupted model causes a failure
- **THEN** the provider does not retry the operation indefinitely
