# Spec Delta

## MODIFIED Requirements

### Requirement: Embeddings for normalized theses
The component SHALL compute embeddings for normalized theses (`text_norm`). The
embedding SHALL be used for clustering. The component SHALL support selecting the
embedding model via config. The component SHALL compute embeddings on a
configurable device (`"auto"`, `"mps"`, `"cuda"`, `"cpu"`). When `"auto"` is
selected, the component SHALL detect the first available GPU backend in the
order MPS, CUDA, then CPU. The component SHALL batch-encode all unique normalized
theses in a single `encode` call before the clustering loop, rather than
encoding each thesis individually inside the loop. The component SHALL log the
resolved device at startup. When the requested device is unavailable, the
component SHALL fall back to CPU with a WARNING and produce identical embeddings.

#### Scenario: Embedding computed once
- **WHEN** a thesis is already in the bank with an embedding
- **THEN** the embedding is not recomputed

#### Scenario: Embedding stored
- **WHEN** a new thesis is added to the bank
- **THEN** its embedding is computed and stored in the entry

#### Scenario: Batch encoding used
- **WHEN** the analysis processes N unique normalized theses
- **THEN** all N embeddings are computed in a single batch encode call before clustering begins

#### Scenario: GPU device resolved from auto
- **WHEN** `embedding_device = "auto"` and MPS is available
- **THEN** the embedding model is loaded on the MPS device and the resolved device is logged

#### Scenario: GPU fallback to CPU
- **WHEN** `embedding_device = "mps"` and MPS is not available
- **THEN** the embedding model is loaded on CPU, a WARNING is logged, and the resulting embeddings are identical to a CPU-only run

#### Scenario: Device configuration respected
- **WHEN** `embedding_device = "cpu"`
- **THEN** the embedding model is loaded on CPU regardless of GPU availability

## ADDED Requirements

### Requirement: Embedding device configuration
The component SHALL accept an `embedding_device` configuration field with values
`"auto"`, `"mps"`, `"cuda"`, or `"cpu"`. The default SHALL be `"auto"`. The
component SHALL accept an `embedding_batch_size` configuration field controlling
the batch size for the batch encode call. The default SHALL be 32. The resolved
device and batch size SHALL be logged at startup.

#### Scenario: Default device is auto
- **WHEN** no `embedding_device` is specified in config
- **THEN** the component uses `"auto"` and selects the first available GPU backend

#### Scenario: Invalid device rejected
- **WHEN** `embedding_device` is set to an unrecognized value
- **THEN** the component raises a configuration error

#### Scenario: Batch size respected
- **WHEN** `embedding_batch_size = 64` and 100 unique theses are processed
- **THEN** the batch encode call is invoked with a batch size of 64
