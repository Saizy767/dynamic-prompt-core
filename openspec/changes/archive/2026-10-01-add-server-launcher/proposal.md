# Proposal

## Why

`asyncTask.py` is an async client that talks to an already-running
OpenAI-compatible server (vLLM / llama.cpp). Today that server must be started
manually in a separate terminal before any inference can happen. There is no
config-driven way to launch it, and `config.toml` — which `asyncTask.py`
already tries to read — does not exist yet. This change adds a launcher that
starts the inference server from `config.toml`, waits for it to be ready, and
hands off the live endpoint, closing the gap between "model on disk" and
"client can connect."

## What Changes

- Add a launcher script that reads `config.toml`, spawns the selected backend
  (`llama-server` or `vllm`) as a subprocess, polls `GET /v1/models` until the
  server is healthy (with timeout), and returns the live endpoint URL. It
  handles clean shutdown via signals (SIGINT/SIGTERM) and kills the child on
  exit.
- Create `config.toml` with a shared `[llm]` schema. The launcher and
  `asyncTask.py` agree on this one file. Existing keys `model_path` and
  `served_model_name` (already read by `asyncTask.py`) are preserved; new keys
  are added: `backend`, `host`, `port`, `ctx_size`, `gpu_layers`, `threads`,
  plus a small set of backend-specific overrides.
- `llama.cpp` (`llama-server`) is the fully working backend now: macOS/Metal,
  GGUF model (`model/qwen2.5-3b-instruct-q4_k_m.gguf`).
- `vLLM` backend is scaffolded: the launcher builds the correct `vllm serve`
  argv from config, but refuses to run on darwin with a clear message pointing
  to Linux. Full vLLM support is deferred to a Linux GPU environment.
- No changes to `asyncTask.py`. It already reads `[llm].model_path` and
  `[llm].served_model_name`; the launcher only adds keys that `asyncTask.py`
  does not touch.

## Capabilities

### New Capabilities
- `llm-server-launcher`: Start, health-check, and manage the lifecycle of an
  OpenAI-compatible inference server (llama.cpp now, vLLM later) from
  `config.toml`. Hand off the live endpoint to clients like `asyncTask.py`.

### Modified Capabilities
<!-- None. No existing specs in the project. -->

## Impact

- **New files:** launcher script (e.g. `server_launcher.py`), `config.toml`
  (seed).
- **Unchanged:** `asyncTask.py`, `data/`, `model/`.
- **Dependencies:** Python stdlib only for the launcher (`subprocess`,
  `tomllib`, `signal`, `logging`). No new third-party packages.
- **Runtime:** requires `llama-server` on PATH (already installed via Homebrew
  on this machine). `vllm` is not required for the llama.cpp path.
