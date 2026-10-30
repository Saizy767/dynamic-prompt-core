"""Atomic file writes shared by all layers (stdlib-only).

shortcuts: single shared implementation, no per-caller re-verification loops
"""

from __future__ import annotations

import itertools
import json
import os

from dynamic_prompt_core.domain.errors.io import AtomicWriteError

_write_counter = itertools.count()


def atomic_write(
    path: str, payload: str, counter: itertools.count[int], *, jsonl: bool = False
) -> None:
    """Write *payload* to *path* atomically via a unique temp file.

    The payload is written to a unique temporary file in the target directory,
    flushed and fsynced, read back and validated as JSON (per line when
    *jsonl* is True, as a single document otherwise), then moved into place
    with ``os.replace``.  The parent directory is best-effort fsynced.  On any
    failure the target file is left untouched and ``AtomicWriteError`` is
    raised.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    counter_val = next(counter)
    tmp = os.path.join(
        directory,
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{counter_val}",
    )
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, encoding="utf-8") as vf:
            if jsonl:
                for line in vf:
                    line = line.strip()
                    if line:
                        json.loads(line)
            else:
                json.load(vf)
        os.replace(tmp, path)
        _fsync_directory(directory)
    except Exception as exc:
        raise AtomicWriteError(f"failed to write file {path}: {exc}") from exc
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _fsync_directory(directory: str) -> None:
    """Best-effort fsync of a directory for crash durability."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


__all__ = ["atomic_write"]
