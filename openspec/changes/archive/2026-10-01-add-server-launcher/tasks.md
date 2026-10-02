# Tasks

## 1. Config foundation

- [x] 1.1 Create seed `config.toml` with the `[llm]` schema from design (`backend`, `model_path`, `served_model_name`, `host`, `port`, `ctx_size`, `gpu_layers`, `threads`, `startup_timeout`) pointing at `model/qwen2.5-3b-instruct-q4_k_m.gguf`, with comments documenting each key and the `served_model_name` coupling. Verify it parses with `python -c "import tomllib; tomllib.load(open('config.toml','rb'))"`.
- [x] 1.2 Implement `load_config(path)` in `server_launcher.py`: read TOML via `tomllib`, validate required keys (`backend`, `model_path`, `served_model_name`, `host`, `port`), fail fast with a message naming the missing/invalid key. Verify a missing key raises an error naming that key, and that the seed `config.toml` loads cleanly.

## 2. Backend argv builders

- [x] 2.1 Implement `build_argv("llamacpp", config)` returning the `llama-server` argv (`-m`, `--host`, `--port`, `-c`, `-ngl`, `-t`, `--jinja`) plus any `[llm.llamacpp].extra_args`. Verify the argv contains the configured values and only flags valid per `llama-server --help`.
- [x] 2.2 Implement `build_argv("vllm", config)` returning the `vllm serve` argv (`--host`, `--port`, `--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size` from `[llm.vllm]`). Verify the argv shape matches vLLM's `serve` subcommand.
- [x] 2.3 Add the darwin refusal: when backend is `vllm` and `sys.platform == "darwin"`, print a clear message naming Linux as the requirement and exit non-zero without spawning. Verify the refusal message appears and no subprocess is created.

## 3. Readiness and lifecycle

- [x] 3.1 Implement `wait_for_ready(endpoint, timeout, interval=0.5)`: poll `GET /v1/models` via `urllib.request` until HTTP 200 or timeout. Return `True` on ready, `False` on timeout. Verify it returns `True` when a local HTTP server returns 200, and `False` (within ~timeout) when nothing answers.
- [x] 3.2 Implement `ServerLauncher`: `start()` spawns the subprocess with `start_new_session=True`, calls `wait_for_ready`, and exposes `.endpoint`; `stop()` sends `SIGTERM` to the process group via `os.killpg` and escalates to `SIGKILL` after a grace period. Verify `stop()` terminates the child and the port is released (checkable with a second bind or `lsof`).
- [x] 3.3 Wire SIGINT/SIGTERM handlers to call `stop()`. Verify that sending SIGINT to a running launcher kills the child process and leaves no orphan holding the port.

## 4. CLI entry point and integration

- [x] 4.1 Implement `main()` / `__main__` block: load config, create `ServerLauncher`, `start()`, log the endpoint and PID, block until interrupted, then `stop()`. Support `--config <path>`. Verify `python server_launcher.py --config config.toml` starts `llama-server`, logs a ready endpoint, and shuts down cleanly on Ctrl+C.
- [x] 4.2 End-to-end smoke test against the real `llama-server` + existing GGUF: start the launcher, confirm `GET /v1/models` returns 200 with the configured `served_model_name`, confirm `asyncTask.AsyncTask` can construct using the handed-off endpoint and `config.toml`, then shut down. Verify no orphaned process holds port 8080 after exit.
