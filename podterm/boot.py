"""Boot progress — parses RunPod machine logs into image-pull layer states.

Machine log lines (from api_get_machine_logs) look like:
    2026-06-13T00:26:51Z create container ghcr.io/sysreq/gpt-golf-train:latest
    2026-06-13T00:26:52Z b257450ba61a Pulling fs layer
    2026-06-13T00:26:53Z b257450ba61a Downloading
    2026-06-13T00:26:56Z b257450ba61a Pull complete
    2026-06-13T00:28:01Z start container

Each layer hash is tracked through its docker-pull lifecycle; total progress
assumes equal-sized layers (the log carries no byte counts).
"""

from __future__ import annotations

import re

_RE_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s+")
_RE_LAYER = re.compile(r"^([0-9a-f]{10,16})\s+(.+)$")

# Docker pull statuses → layer state, in lifecycle order. Transitions are
# forward-only so replayed log lines can never regress a layer.
_LAYER_STATES = {
    "Pulling fs layer": "pending",
    "Waiting": "waiting",
    "Downloading": "downloading",
    "Verifying Checksum": "verifying",
    "Download complete": "downloaded",
    "Extracting": "extracting",
    "Pull complete": "complete",
    "Already exists": "complete",
}
_STATE_RANK = {s: i for i, s in enumerate(
    ["pending", "waiting", "downloading", "verifying", "downloaded", "extracting", "complete"])}
# Per-layer progress fraction at each state (equal-weight layers)
_STATE_PROGRESS = {
    "pending": 0.0, "waiting": 0.0, "downloading": 0.4, "verifying": 0.65,
    "downloaded": 0.7, "extracting": 0.85, "complete": 1.0,
}


class PullTracker:
    """Accumulates machine log lines into a boot/pull progress snapshot."""

    def __init__(self) -> None:
        self.layers: dict[str, str] = {}  # layer id → state, insertion-ordered
        self.image: str | None = None
        self.message: str | None = None  # latest non-layer line, timestamp stripped
        self.container_started = False
        self._seen = 0  # lines already ingested (the API returns the full list)
        self._messages_seen: set[str] = set()

    def ingest(self, lines: list[str]) -> tuple[list[str], bool]:
        """Feed the full log list; returns (new non-layer messages, state changed)."""
        if len(lines) < self._seen:  # log rotated/reset — replay is safe
            self._seen = 0
        new_messages: list[str] = []
        changed = False
        for line in lines[self._seen:]:
            text = _RE_TIMESTAMP.sub("", line).strip()
            if not text:
                continue
            m = _RE_LAYER.match(text)
            if m and m.group(2) in _LAYER_STATES:
                layer_id, state = m.group(1), _LAYER_STATES[m.group(2)]
                prev = self.layers.get(layer_id)
                if prev is None or _STATE_RANK[state] > _STATE_RANK[prev]:
                    self.layers[layer_id] = state
                    changed = True
            elif not m:  # non-layer machine event (create volume/container, start, …)
                if text.startswith("create container "):
                    self.image = text.removeprefix("create container ")
                if text.startswith("start container"):
                    self.container_started = True
                if self.message != text:
                    self.message = text
                    changed = True
                if text not in self._messages_seen:
                    self._messages_seen.add(text)
                    new_messages.append(text)
        self._seen = len(lines)
        return new_messages, changed

    def snapshot(self, done: bool = False, message: str | None = None) -> dict:
        states = list(self.layers.values())
        total = len(states)
        complete = sum(1 for s in states if s == "complete")
        pct = (sum(_STATE_PROGRESS[s] for s in states) / total * 100) if total else 0.0
        if done or self.container_started:
            stage = "Container started"
        elif total and complete == total:
            stage = "Starting container"
        elif total:
            stage = "Pulling image"
        else:
            stage = "Provisioning pod"
        return {
            "t": "pull",
            "stage": stage,
            "image": self.image,
            "message": message or self.message,
            "layers": [{"id": lid, "state": s} for lid, s in self.layers.items()],
            "total": total,
            "complete": complete,
            "pct": round(100.0 if done else pct, 1),
            "done": done or self.container_started,
        }
