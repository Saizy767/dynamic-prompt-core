"""Server readiness polling."""
from __future__ import annotations

import time
import urllib.request


def wait_for_ready(
    endpoint: str, timeout: float, interval: float = 0.5
) -> bool:
    """Poll GET {endpoint}/models until HTTP 200 or timeout elapses.

    Returns True if the server became ready, False on timeout.
    """
    url = endpoint.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False
