from __future__ import annotations

import json
import time


def parse_json_body(body: bytes):
    try:
        return json.loads(body)
    except ValueError:
        return None


def pod_status(pod: dict | None, default: str | None = None) -> str | None:
    if not pod:
        return default
    return pod.get("desiredStatus", "UNKNOWN")


def is_terminal_status(status: str | None) -> bool:
    return status in ("EXITED", "TERMINATED")


def poll_attempts(stop_event, count: int, interval: float):
    for tick in range(count):
        if stop_event.is_set():
            return
        yield tick
        stop_event.wait(interval)


def deadline_ticks(stop_event, timeout: float, interval: float):
    deadline = time.monotonic() + timeout
    tick = 0
    while time.monotonic() < deadline and not stop_event.is_set():
        yield tick
        tick += 1
        stop_event.wait(interval)
