# Judgment Prompt Specification

## Purpose

Define the semantic contract for the candidate-evaluation prompt used by the
logit scorer: the prompt establishes classification context for evaluating one
explicit candidate, preserves semantic criteria from the existing
classification task, excludes generation-protocol instructions, and exposes the
prefix/candidate boundary for tokenization.

## Requirements

### Requirement: Judgment prompt purpose
The judgment prompt SHALL establish the context in which one explicit candidate
is evaluated as a possible classification of the input. The prompt SHALL NOT
ask the model to produce a final classification, generate a JSON object, emit a
confidence value, or produce a classification response. The model-facing task
is to evaluate how well the candidate describes or matches the input according
to the classification criteria. The candidate itself is the continuation whose
token likelihood is measured by the logit scorer.

#### Scenario: Prompt evaluates one candidate
- **WHEN** a judgment prompt is constructed for an input and a candidate
- **THEN** the prompt establishes the context for evaluating that single candidate against the input

#### Scenario: Prompt does not request classification generation
- **WHEN** a judgment prompt is inspected
- **THEN** it does not ask the model to produce a final classification, JSON object, confidence value, or classification response

#### Scenario: Candidate is the scored continuation
- **WHEN** a judgment prompt is constructed
- **THEN** the candidate is positioned as the continuation whose token likelihood is measured by the scorer

### Requirement: Judgment prompt structure
The judgment prompt SHALL consist of a contextual prefix followed by the
candidate continuation. The prefix SHALL contain the classification input text
and semantic classification criteria, terminated by a candidate label. The
candidate continuation SHALL be the exact `Candidate.value`. The candidate
continuation MUST be the final textual component of the prompt: no semantic
instructions or other generated text may appear between the candidate label and
the candidate value. The prompt SHALL expose the boundary between the prefix
and the candidate continuation so the tokenizer can determine candidate
continuation positions structurally.

#### Scenario: Prompt has prefix and candidate continuation
- **WHEN** a judgment prompt is constructed for an input and a candidate
- **THEN** the prompt consists of a contextual prefix followed by the candidate continuation

#### Scenario: Prefix contains input text
- **WHEN** a judgment prompt is constructed with input text `T`
- **THEN** the prefix contains `T`

#### Scenario: Prefix contains semantic criteria
- **WHEN** a judgment prompt is constructed
- **THEN** the prefix contains semantic classification criteria that define what constitutes a good classification

#### Scenario: Prefix terminates with candidate label
- **WHEN** a judgment prompt prefix is inspected
- **THEN** it terminates with a candidate label marking the start of the candidate continuation

#### Scenario: Candidate continuation is the candidate value
- **WHEN** a judgment prompt is constructed for a candidate with value `V`
- **THEN** the candidate continuation is exactly `V`

#### Scenario: Candidate continuation is the final component
- **WHEN** a judgment prompt is constructed
- **THEN** no semantic instructions or other generated text appears between the candidate label and the candidate value, and the candidate value is the final textual component of the prompt

#### Scenario: Prefix/candidate boundary is explicit
- **WHEN** a judgment prompt is constructed
- **THEN** the boundary between the prefix and the candidate continuation is explicitly exposed to the tokenizer

### Requirement: Semantic classification criteria preserved
The judgment prompt SHALL carry the semantic classification criteria that
define what constitutes a good classification as the active classification
rules supplied by the application: the prompt's rules section contains the
active rule texts, and the fixed prompt frame instructs the model to evaluate
how well the candidate matches the input text according to those rules. A
hardcoded semantic prefix SHALL NOT replace the active rules.

#### Scenario: Active rules carried as criteria
- **WHEN** the application supplies active classification rules
- **THEN** the judgment prompt's rules section contains those rule texts

#### Scenario: No hardcoded semantic prefix
- **WHEN** the prompt builder is inspected
- **THEN** semantic criteria come from the active rules, not from a hardcoded prefix

#### Scenario: Empty rules render empty section
- **WHEN** the active rules list is empty (zero-shot baseline)
- **THEN** the judgment prompt is built with an empty rules section

### Requirement: Generation-protocol instructions excluded
The judgment prompt SHALL NOT contain instructions that exist solely to control
generated output format. The prompt SHALL NOT instruct the model to return
JSON, use a specific response schema, return a confidence number, or output
only a classification object. These instructions belong to the generative
classification protocol and have no semantic role in candidate logit scoring.
Selection-oriented task instructions are not generation-protocol instructions;
they are rewritten into candidate-evaluation semantics by the
classification-selection rewriting requirement.

#### Scenario: No JSON output instruction
- **WHEN** a judgment prompt is inspected
- **THEN** it does not contain instructions to return JSON output

#### Scenario: No confidence value instruction
- **WHEN** a judgment prompt is inspected
- **THEN** it does not contain instructions to return a confidence value

#### Scenario: No response schema instruction
- **WHEN** a judgment prompt is inspected
- **THEN** it does not contain instructions to use a specific response schema or structured-output format

#### Scenario: No response-format-only instruction
- **WHEN** a judgment prompt is inspected
- **THEN** it does not contain instructions that exist solely to control generated output format (e.g., "no markdown", "no extra text", "output only the classification object")

### Requirement: Classification-selection instructions rewritten
Instructions in the existing prompt that direct the model to choose the best
class or select a label SHALL be rewritten into candidate-evaluation semantics
in the judgment prompt. The rewritten instructions SHALL preserve the original
task semantics without implying that the model must perform the final
multi-class selection. The prompt SHALL describe evaluation of the supplied
candidate only.

#### Scenario: Selection instruction rewritten to evaluation
- **WHEN** the existing prompt contains an instruction to choose the best class
- **THEN** the judgment prompt rewrites it into an instruction to evaluate how well the candidate matches the input

#### Scenario: No implied final selection
- **WHEN** a judgment prompt is inspected
- **THEN** it does not imply that the model must perform the final classification selection

### Requirement: Candidate-specific prompt
Each candidate SHALL receive its own logical judgment prompt. The candidate set
SHALL NOT be embedded into one prompt asking the model to choose among several
candidates. For `N` candidates, the scorer SHALL construct `N` candidate-specific
prompts, each pairing the same input with one candidate. The model evaluates
`input + candidate` for each candidate independently, not `input + candidate_1
+ candidate_2 + ...` and then a generated choice.

#### Scenario: One prompt per candidate
- **WHEN** a judgment prompt is constructed for input `T` and candidates `[A, B, C]`
- **THEN** three separate prompts are constructed: `(T, A)`, `(T, B)`, and `(T, C)`

#### Scenario: Candidates not embedded together
- **WHEN** a judgment prompt is constructed for multiple candidates
- **THEN** no single prompt contains more than one candidate

#### Scenario: Same input across candidate prompts
- **WHEN** candidate-specific prompts are constructed for the same input
- **THEN** each prompt contains the same input text

### Requirement: No implied access to other candidates
The judgment prompt SHALL describe evaluation of the supplied candidate only.
The prompt SHALL NOT instruct the model to compare the candidate against an
implicitly known candidate list. Candidate comparison happens outside the model
through the `ClassificationPolicy`, not inside the prompt.

#### Scenario: No candidate list in prompt
- **WHEN** a judgment prompt is constructed for one candidate
- **THEN** the prompt does not contain or reference other candidates

#### Scenario: No comparison instruction
- **WHEN** a judgment prompt is inspected
- **THEN** it does not instruct the model to compare the candidate against other candidates

### Requirement: Candidate text preserved verbatim
The candidate value SHALL be inserted into the candidate continuation position
without semantic normalization. The prompt builder SHALL NOT strip whitespace,
lowercase, casefold, collapse whitespace, rewrite punctuation, alter
capitalization, or otherwise normalize the candidate text. The candidate value
supplied by the application is the exact candidate whose continuation is scored.

#### Scenario: Candidate value not stripped
- **WHEN** a candidate with value `"  sports  "` is used in a judgment prompt
- **THEN** the candidate continuation is `"  sports  "`, not `"sports"`

#### Scenario: Candidate value not lowercased
- **WHEN** a candidate with value `"Sports"` is used in a judgment prompt
- **THEN** the candidate continuation is `"Sports"`, not `"sports"`

#### Scenario: Candidate value not case-folded
- **WHEN** candidates `"A"` and `"a"` are used in judgment prompts
- **THEN** their candidate continuations are `"A"` and `"a"` respectively, not case-folded to a common form

#### Scenario: Candidate punctuation preserved
- **WHEN** a candidate with value `"billing / payments"` is used in a judgment prompt
- **THEN** the candidate continuation is `"billing / payments"` with punctuation preserved

#### Scenario: Candidate Unicode preserved
- **WHEN** a candidate with Unicode text is used in a judgment prompt
- **THEN** the Unicode text is preserved in the candidate continuation

### Requirement: Input text preserved as context
The classification input SHALL remain the primary context for candidate
judgment. The prompt builder SHALL NOT summarize, rewrite, classify, or
otherwise transform the input as part of this migration. The prompt changes the
task framing, not the input content.

#### Scenario: Input not transformed
- **WHEN** a judgment prompt is constructed with input text `T`
- **THEN** `T` appears in the prompt unchanged, without summarization, rewriting, or classification

### Requirement: Prompt construction remains in infrastructure
The judgment prompt SHALL be constructed inside the infrastructure layer. The
application SHALL NOT construct prompts, provide prompt fragments, or provide
prompt templates. The application continues to call `Classifier.classify(text,
rules)` with semantic data only. The dependency direction remains
`application` → `Classifier` → `LLMLogitClassifier` → prompt builder
→ tokenizer/model adapter. The application remains unaware of prompt strings,
token boundaries, candidate continuation positions, tokenizer behavior, logits,
or model-specific prompt formatting.

#### Scenario: Prompt built in infrastructure
- **WHEN** a judgment prompt is constructed
- **THEN** the construction occurs inside the infrastructure layer

#### Scenario: Application provides semantic data only
- **WHEN** the application invokes the classifier
- **THEN** it provides text and rules, not prompt fragments or prompt templates

#### Scenario: Application unaware of prompt internals
- **WHEN** the application invokes the classifier
- **THEN** it is unaware of prompt strings, token boundaries, candidate continuation positions, tokenizer behavior, logits, or model-specific formatting

### Requirement: Prompt construction separate from scoring
The judgment prompt builder SHALL be responsible only for constructing the
logical model input. It SHALL NOT tokenize, execute the model, calculate
log-probabilities, choose a candidate, or produce a `Judgment`. This separation
allows prompt behavior to be tested independently from model execution.

#### Scenario: Prompt builder does not tokenize
- **WHEN** the prompt builder is inspected
- **THEN** it does not perform tokenization

#### Scenario: Prompt builder does not execute model
- **WHEN** the prompt builder is inspected
- **THEN** it does not execute model inference

#### Scenario: Prompt builder does not score
- **WHEN** the prompt builder is inspected
- **THEN** it does not calculate log-probabilities or produce judgments

### Requirement: Prompt compatible with batch scoring
The judgment prompt SHALL work with the model-seam batch architecture. For one
`classify(text, rules)` invocation, the classifier SHALL construct one judgment
prompt per constructor candidate and process them through one `forward_batch`
call on the model seam. The prompt SHALL NOT reintroduce per-candidate scoring
invocations at the application contract level. How the seam physically executes
the batch is a backend detail (the llama-server backend issues concurrent
per-candidate-token HTTP requests).

#### Scenario: One prompt per candidate in batch
- **WHEN** the classifier evaluates constructor candidates `[A, B, C]`
- **THEN** three judgment prompts are constructed and processed through one `forward_batch` call

#### Scenario: Application sees one invocation
- **WHEN** the classifier evaluates multiple candidates
- **THEN** the application contract is a single `classify` call regardless of the backend's execution strategy

### Requirement: Prompt does not change scoring algorithm
The judgment prompt SHALL NOT change the scoring algorithm. The ranking signal
SHALL remain the mean candidate-token log-probability of the candidate
continuation. The prompt changes only the contextual framing of the candidate.
The classifier MAY additionally store a monotone normalization of the ranking
signals (softmax across candidate mean log-probabilities) as judgment scores;
such normalization SHALL NOT change the ranking, the selected candidate, or
the mean log-probability computation itself. The prompt SHALL NOT introduce
confidence calibration, temperature scaling, generated explanations, or
generated JSON.

#### Scenario: Scoring algorithm unchanged
- **WHEN** the judgment prompt is used with the logit scorer
- **THEN** the ranking signal remains the mean candidate-token log-probability

#### Scenario: Normalization does not change ranking
- **WHEN** softmax normalization is stored as judgment scores
- **THEN** the selected candidate and the candidate ordering are identical to those derived from the raw mean log-probabilities


