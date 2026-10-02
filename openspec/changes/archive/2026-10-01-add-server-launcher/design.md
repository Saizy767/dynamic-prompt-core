# Design

## Context

`asyncTask.py` is an async client for an already-running OpenAI-compatible
server. It reads `[llm].model_path` and `[llm].served_model_name` from
`config.toml` (which does not exist yet). The model on disk is a GGUF
(`model/qwen2.5-3b-instruct-q4_k_m.gguf`); `llama-server` is installed via
Homebrew; `vllm` is not installed and has poor macOS support. See proposal.md
for motivation.

## Goals / Non-Goals

**Goals:**
- Start `llama-server` from `config.toml` and hand off a ready endpoint.
- Scaffold the `vllm` path (correct argv, darwin refusal) for Linux later.
- One `config.toml` shared with `asyncTask.py` by convention, no code coupling.
- Sync, stdlib-only launcher — no new third-party dependencies.

**Non-Goals:**
- Orchestrating a full inference run (dataset → metrics → log). That is the
  README's "Оркестратор", a separate change.
- Modifying `asyncTask.py`.
- Auto-downloading models, multi-model routing, or GPU autodetection.
- Real vLLM execution on macOS (impossible — deferred to Linux).

## Decisions

### Sync API, not async
The launcher is a start → wait → hand-off primitive: inherently sequential and
blocking. A synchronous API (`subprocess.Popen` + `urllib.request` for the
health poll) is simpler to call from a CLI entry point and needs no event loop.
Callers that want async can wrap with `asyncio.to_thread`.

**Alternative considered:** async with `asyncio.create_subprocess_exec` +
`aiohttp`, to match `asyncTask.py`'s style. Rejected — adds event-loop
machinery for a startup primitive that has no concurrency need.

### Process-group lifecycle
Spawn with `start_new_session=True` (POSIX `setsid`) so the server and any
threads it forks live in their own process group. Shutdown kills the whole
group via `os.killpg(os.getpgid(pid), signal.SIGTERM)`, then escalates to
`SIGKILL` after a grace period. This avoids orphaned server processes holding
the port.

### Readiness polling
Poll `GET /v1/models` every 0.5 s up to `startup_timeout` (configurable; default
30 s for `llamacpp`, 120 s for `vllm` since model load is slow). Each attempt
uses a short socket timeout. The endpoint shape matches `asyncTask.detect_backend`
(asyncTask.py:288), so the two agree on what "ready" means.

### Config schema: flat `[llm]` + optional backend sub-tables
```
[llm]
backend = "llamacpp"          # or "vllm"
model_path = "model"          # tokenizer dir / HF repo (asyncTask.py + vLLM)
gguf_path = "model/...gguf"   # GGUF file, llamacpp only (launcher-only key)
served_model_name = "qwen2.5-3b-instruct"
host = "127.0.0.1"
port = 8080
ctx_size = 4096
gpu_layers = 0                # llamacpp: -ngl; vllm: ignored (uses --gpu-memory-utilization)
threads = 4                   # llamacpp: -t; vllm: ignored
startup_timeout = 30          # seconds to wait for /v1/models

[llm.llamacpp]                # optional, backend-specific extra flags
# extra_args = []

[llm.vllm]                    # optional
# gpu_memory_utilization = 0.9
# tensor_parallel_size = 1
```
Common keys live flat under `[llm]`; backend-specific extras go in sub-tables
so the common case stays simple. `model_path` is the tokenizer directory / HF
repo (read by `asyncTask.py` for `AutoTokenizer` and by vLLM as the serve
target). `gguf_path` is a launcher-only key for the llamacpp GGUF file;
`asyncTask.py` ignores it. `asyncTask.py` reads only `model_path` and
`served_model_name` and ignores the rest — no change to it.

### `served_model_name` must match the server's reported id
`llama-server` reports the model id via `/v1/models`. `asyncTask.py` sends
`served_model_name` as the `model` field in chat payloads. If they mismatch the
server rejects the request. The seed `config.toml` sets `served_model_name` to
the value the server actually reports; this is documented in the config comments
rather than handled in code.

### Backend argv builders (internal)
```
llamacpp:  llama-server -m <model_path> --host <host> --port <port>
           -c <ctx_size> -ngl <gpu_layers> -t <threads> --jinja
vllm:      vllm serve <model_path> --host <host> --port <port>
           --max-model-len <ctx_size> [--gpu-memory-utilization ...]
```
A builder function per backend turns the config dict into an argv list. This is
the only backend-specific logic; everything above it is backend-agnostic.

### Module layout
```
server_launcher.py
  load_config(path) -> dict            # tomllib, validate required keys
  build_argv(backend, config) -> list  # per-backend argv
  wait_for_ready(endpoint, timeout)    # poll GET /v1/models
  ServerLauncher                       # Popen + lifecycle + signal handling
  main()                               # CLI: start, block until Ctrl+C
config.toml                            # seed config
```

## Risks / Trade-offs

- **[Port collision]** A previous server may still hold the port. → Mitigation:
  `wait_for_ready` distinguishes "port bound but not ready" from "port free";
  on timeout, log a hint that the port may be in use and exit non-zero.
- **[vLLM argv drift]** vLLM CLI flags change across versions; the scaffolded
  argv may break on a future Linux vLLM. → Mitigation: keep the vLLM builder
  minimal and documented; real validation happens on Linux when vLLM is
  available.
- **[served_model_name mismatch]** If the user edits `served_model_name` to
  something the server doesn't report, `asyncTask.py` gets 404s. → Mitigation:
  document the coupling in `config.toml` comments; a future change could verify
  the name against `/v1/models` after readiness.
- **[No Windows support]** `start_new_session` and `os.killpg` are POSIX-only.
  → Mitigation: acceptable — target is macOS/Linux. Guard with a platform check
  if Windows is ever needed.
