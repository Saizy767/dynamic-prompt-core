# Spec Delta

## Purpose

Start, health-check, and manage the lifecycle of an OpenAI-compatible
inference server (llama.cpp or vLLM) from a single `config.toml`, handing off
the live endpoint to clients like `asyncTask.py`.

## ADDED Requirements

### Requirement: Launch server from config
The launcher SHALL read `config.toml` and start the inference backend named by
`[llm].backend` as a subprocess, using the model path and server parameters
defined in the config file.

#### Scenario: llama.cpp server starts
- **WHEN** `[llm].backend` is `llamacpp` and `gguf_path` points at an existing GGUF file
- **THEN** the launcher starts `llama-server` with the model and server flags from config

#### Scenario: vLLM server starts on Linux
- **WHEN** `[llm].backend` is `vllm` and the platform is Linux
- **THEN** the launcher starts `vllm serve` with the model and server flags from config

### Requirement: Readiness gate before handoff
The launcher SHALL poll `GET /v1/models` on the configured host and port and
MUST NOT report readiness until the server responds with HTTP 200, or until a
configurable startup timeout elapses.

#### Scenario: Server becomes ready
- **WHEN** the server responds 200 on `/v1/models` within the timeout
- **THEN** the launcher reports the server ready and exposes the endpoint

#### Scenario: Server fails to start in time
- **WHEN** `/v1/models` does not return 200 before the startup timeout
- **THEN** the launcher terminates the subprocess and reports a startup failure naming the timeout

### Requirement: Endpoint handoff
The launcher SHALL expose the live endpoint URL (scheme, host, port) so that
clients can connect to the running server without hardcoding addresses.

#### Scenario: Client receives endpoint
- **WHEN** the server is ready
- **THEN** the launcher provides the endpoint URL derived from `[llm].host` and `[llm].port`

### Requirement: Clean shutdown
The launcher SHALL terminate the server subprocess on SIGINT, SIGTERM, and on
normal launcher exit, releasing the bound port.

#### Scenario: Interrupt during run
- **WHEN** the launcher receives SIGINT while the server is running
- **THEN** the child process is terminated and the port is released

#### Scenario: Launcher exits normally
- **WHEN** the launcher completes its run and exits
- **THEN** the child process is terminated and no orphaned server remains

### Requirement: vLLM refusal on macOS
The launcher SHALL refuse to start the `vllm` backend on macOS (darwin) with a
clear, actionable message, rather than attempting a launch that cannot succeed.

#### Scenario: vLLM selected on darwin
- **WHEN** `[llm].backend` is `vllm` and the platform is darwin
- **THEN** the launcher prints a message explaining vLLM requires Linux and exits without spawning a process

### Requirement: Config validation
The launcher SHALL validate that required config keys are present and
well-formed before attempting to start a server, and SHALL fail fast with a
message naming the missing or invalid key.

#### Scenario: Missing model path
- **WHEN** `config.toml` is missing `[llm].model_path`
- **THEN** the launcher exits with an error message naming `model_path` as the missing key

#### Scenario: Unknown backend
- **WHEN** `[llm].backend` is a value other than `llamacpp` or `vllm`
- **THEN** the launcher exits with an error message naming the invalid backend value

### Requirement: Shared config schema with client
The launcher SHALL read the same `config.toml` `[llm]` section that
`asyncTask.py` reads, preserving existing keys `model_path` and
`served_model_name`, and adding launcher-only keys that the client ignores.

#### Scenario: Client and launcher agree on model identity
- **WHEN** the launcher starts a server from `config.toml` and `asyncTask.py` connects to it
- **THEN** both use the same `served_model_name` so the client's model field matches the server's served name
