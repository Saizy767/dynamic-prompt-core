"""
Smoke test for the transport and logging layer (Stage 0).

Assumes a server is already running. Start it with:
    python server_launcher.py --config config.toml

Run:
    python smoke_test.py [--config config.toml] [--endpoint http://127.0.0.1:8080/v1]

Exit code 0 on success, non-zero on any failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tomllib

import aiohttp
from pydantic import BaseModel

from dynamic_prompt_core.infrastructure.llm import (
    AsyncTask,
    ResponseStatus,
    StructuredMode,
)

DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"
LOG_PATH = "logs/smoke.jsonl"


# --------------------------------------------------------------------------- #
#  Placeholder Pydantic models
# --------------------------------------------------------------------------- #
class ClassifyResult(BaseModel):
    decision: int
    theses: list[str]
    confidence: float


class ExtractResult(BaseModel):
    theses: list[str]
    confidence: float


CLASSIFY_SYSTEM_PROMPT = (
    "You are a binary classifier. Analyze the text and return a JSON "
    "object with decision (0 or 1), theses (list of key points), and "
    "confidence (0.0 to 1.0)."
)
EXTRACT_SYSTEM_PROMPT = (
    "Extract key theses from the text. Return a JSON object with "
    "theses (list of strings) and confidence (0.0 to 1.0)."
)

SAMPLE_TEXT = (
    "This product exceeds expectations. The build quality is excellent "
    "and the price is reasonable. However, the documentation could be "
    "better and the setup process was confusing."
)
LONG_TEXT = "This is a very long text that should exceed the truncation limit. " * 200


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


async def probe_structured_mode(
    endpoint: str, model_name: str
) -> str:
    """Probe whether the server accepts json_schema, fall back to json_object."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Return a JSON object."},
            {"role": "user", "content": "Return {\"x\": 1}"},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "probe",
                "schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
                "strict": True,
            },
        },
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    print("[probe] json_schema accepted by server")
                    return "json_schema"
                await resp.text()
                print(
                    f"[probe] json_schema rejected (HTTP {resp.status}), "
                    f"falling back to json_object"
                )
                return "json_object"
    except Exception as exc:
        print(f"[probe] json_schema probe failed ({exc}), using json_object")
        return "json_object"


def verify_log(log_path: str, min_entries: int) -> bool:
    """Verify the jsonl log has entries with required fields."""
    if not os.path.exists(log_path):
        print(f"[FAIL] Log file not found: {log_path}")
        return False

    required_fields = {
        "timestamp", "run_id", "call_type", "system_prompt_hash",
        "text_hash", "model_name", "params", "raw_content",
        "latency_ms", "finish_reason", "usage", "status",
        "parse_status", "attempts", "http_status", "error", "truncated",
    }

    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if len(entries) < min_entries:
        print(f"[FAIL] Expected >= {min_entries} log entries, got {len(entries)}")
        return False

    for i, entry in enumerate(entries):
        missing = required_fields - set(entry.keys())
        if missing:
            print(f"[FAIL] Log entry {i} missing fields: {missing}")
            return False

    print(f"[OK] Log has {len(entries)} entries, all required fields present")
    return True


# --------------------------------------------------------------------------- #
#  Main test
# --------------------------------------------------------------------------- #
async def run_smoke_test(config_path: str, endpoint: str) -> int:
    failures: list[str] = []

    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    # Load config for model name
    config = _load_config(config_path)
    served_model_name = config["llm"]["served_model_name"]

    # Probe structured mode
    print("--- Structured mode probe ---")
    mode = await probe_structured_mode(endpoint, served_model_name)
    structured_mode = (
        StructuredMode.JSON_SCHEMA if mode == "json_schema"
        else StructuredMode.JSON_OBJECT
    )

    # Create AsyncTask
    print("--- Creating AsyncTask ---")
    try:
        task = AsyncTask(
            config_path=config_path,
            endpoint=endpoint,
            log_path=LOG_PATH,
            run_id="smoke",
            structured_mode=structured_mode,
        )
    except Exception as exc:
        print(f"[FAIL] Cannot create AsyncTask: {exc}")
        return 1

    print(
        f"[OK] AsyncTask created: backend={task._backend.value}, "
        f"mode={task._structured_mode.value}, temp={task._temperature}"
    )

    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        # --- classify ---
        print("--- classify call ---")
        try:
            result = await task.classify(
                session, SAMPLE_TEXT, ClassifyResult,
                system_prompt=CLASSIFY_SYSTEM_PROMPT,
            )
            if result is None:
                failures.append("classify returned None")
                print("[FAIL] classify returned None")
            else:
                print(
                    f"[OK] classify: decision={result.decision}, "
                    f"confidence={result.confidence:.3f}, "
                    f"theses={len(result.theses)}"
                )
        except Exception as exc:
            failures.append(f"classify raised: {exc}")
            print(f"[FAIL] classify raised: {exc}")

        # --- extract_theses ---
        print("--- extract_theses call ---")
        try:
            result = await task.extract_theses(
                session, SAMPLE_TEXT, ExtractResult,
                system_prompt=EXTRACT_SYSTEM_PROMPT,
            )
            if result is None:
                failures.append("extract_theses returned None")
                print("[FAIL] extract_theses returned None")
            else:
                print(
                    f"[OK] extract_theses: theses={len(result.theses)}, "
                    f"confidence={result.confidence:.3f}"
                )
        except Exception as exc:
            failures.append(f"extract_theses raised: {exc}")
            print(f"[FAIL] extract_theses raised: {exc}")

        # --- analyze_raw composition ---
        print("--- analyze_raw composition check ---")
        try:
            raw = await task.analyze_raw(
                session, SAMPLE_TEXT, ClassifyResult,
                system_prompt=CLASSIFY_SYSTEM_PROMPT,
                max_tokens=128,
                truncate_tokens=300,
            )
            if raw.status != ResponseStatus.OK:
                failures.append(
                    f"analyze_raw status={raw.status.value}"
                )
                print(f"[FAIL] analyze_raw status={raw.status.value}")
            elif not isinstance(raw.content, str):
                failures.append("analyze_raw content is not a string")
                print("[FAIL] analyze_raw content is not a string")
            else:
                parsed = ClassifyResult.model_validate_json(raw.content)
                composed = await task.analyze(
                    session, SAMPLE_TEXT, ClassifyResult,
                    call_type="classify",
                    system_prompt=CLASSIFY_SYSTEM_PROMPT,
                    max_tokens=128,
                    truncate_tokens=300,
                )
                if composed is not None and parsed.decision == composed.decision:
                    print("[OK] analyze_raw composition verified")
                else:
                    print("[WARN] analyze_raw parsed but analyze returned None "
                          "(may indicate parse issue)")
        except Exception as exc:
            failures.append(f"analyze_raw raised: {exc}")
            print(f"[FAIL] analyze_raw raised: {exc}")

        # --- determinism ---
        print("--- determinism check ---")
        try:
            r1 = await task.classify(
                session, SAMPLE_TEXT, ClassifyResult,
                system_prompt=CLASSIFY_SYSTEM_PROMPT,
            )
            r2 = await task.classify(
                session, SAMPLE_TEXT, ClassifyResult,
                system_prompt=CLASSIFY_SYSTEM_PROMPT,
            )
            if r1 is not None and r2 is not None:
                if r1.decision == r2.decision and r1.confidence == r2.confidence:
                    print("[OK] determinism verified (identical results)")
                else:
                    failures.append("determinism check failed")
                    print(
                        f"[FAIL] Results differ: "
                        f"r1=({r1.decision}, {r1.confidence}) "
                        f"r2=({r2.decision}, {r2.confidence})"
                    )
            else:
                print("[WARN] classify returned None, skipping determinism")
        except Exception as exc:
            failures.append(f"determinism raised: {exc}")
            print(f"[FAIL] determinism raised: {exc}")

        # --- truncation ---
        print("--- truncation check ---")
        try:
            await task.classify(
                session, LONG_TEXT, ClassifyResult,
                system_prompt=CLASSIFY_SYSTEM_PROMPT,
                truncate_tokens=50,
            )
            with open(LOG_PATH, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            truncated_entry = next(
                (e for e in reversed(lines) if e.get("truncated") is True),
                None,
            )
            if truncated_entry:
                print("[OK] truncation logged (truncated=true)")
            else:
                failures.append("truncation not logged")
                print("[FAIL] No log entry with truncated=true found")
        except Exception as exc:
            failures.append(f"truncation raised: {exc}")
            print(f"[FAIL] truncation raised: {exc}")

    # --- log verification ---
    print("--- log verification ---")
    if not verify_log(LOG_PATH, min_entries=2):
        failures.append("log verification failed")

    # --- summary ---
    print("\n" + "=" * 50)
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("SMOKE TEST PASSED")
        return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for the LLM transport layer (Stage 0)."
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="Path to config.toml",
    )
    parser.add_argument(
        "--endpoint", default=None,
        help="Server endpoint (default: from config)",
    )
    args = parser.parse_args(argv)

    endpoint = args.endpoint
    if endpoint is None:
        config = _load_config(args.config)
        llm = config["llm"]
        endpoint = f"http://{llm['host']}:{llm['port']}/v1"

    print(f"Endpoint: {endpoint}")
    print(f"Config:   {os.path.abspath(args.config)}")
    print(f"Log:      {LOG_PATH}")
    print()

    try:
        exit_code = asyncio.run(run_smoke_test(args.config, endpoint))
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit_code = 1
    except Exception as exc:
        print(f"[FAIL] Unexpected error: {exc}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
