#!/usr/bin/env python3
"""Pod event daemon — serves structured JSONL events + the raw teed log over HTTP.

Stdlib only: starts before `uv sync` (system python3). PodTerm pulls via the RunPod
HTTP proxy. Env: EVENTS_FILE, LOG_FILE, EVENTD_TOKEN, EVENTD_PORT (default 8765).
"""

import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

EVENTS_FILE = os.environ.get("EVENTS_FILE", "")
LOG_FILE = os.environ.get("LOG_FILE", "")
TOKEN = os.environ.get("EVENTD_TOKEN", "")
PORT = int(os.environ.get("EVENTD_PORT", "8765"))
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/workspace/snapshots")
ACK_FILE = os.environ.get("SNAPSHOT_ACK_FILE", "/workspace/SNAPSHOT_ACK")


class EventStore:
    """Incremental reader of the events JSONL file. seq == file line index."""

    def __init__(self, path):
        self.path = path; self.lines = []; self.pos = 0; self.lock = threading.Lock()

    def refresh(self):
        with self.lock:
            try: size = os.stat(self.path).st_size
            except OSError: return
            if size < self.pos: self.lines = []; self.pos = 0  # truncated: re-read
            if size == self.pos: return
            with open(self.path, "rb") as f:
                f.seek(self.pos); chunk = f.read(size - self.pos)
            end = chunk.rfind(b"\n")
            if end < 0: return  # no complete line yet
            self.pos += end + 1
            for raw in chunk[: end + 1].splitlines():
                try: self.lines.append(json.loads(raw))
                except ValueError: self.lines.append({"t": "raw", "line": raw.decode(errors="replace")})


store = EventStore(EVENTS_FILE)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args): pass  # long-polls every 25s would flood the daemon log

    def _json(self, code, obj, headers=()):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers: self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _authed(self):
        if not TOKEN: return True
        if self.headers.get("Authorization") == f"Bearer {TOKEN}": return True
        self._json(401, {"error": "unauthorized"}); return False

    def do_GET(self):
        url = urlparse(self.path); q = parse_qs(url.query)
        if not self._authed(): return
        if url.path == "/health": return self._health()
        if url.path == "/events": return self._events(q)
        if url.path == "/log": return self._log(q)
        if url.path == "/snapshot": return self._snapshot(q)
        if url.path == "/snapshot/ack": return self._snapshot_ack(q)
        self._json(404, {"error": "not found"})

    def _health(self):
        store.refresh()
        try: log_size = os.stat(LOG_FILE).st_size
        except OSError: log_size = 0
        self._json(200, {"ok": True, "events": len(store.lines), "log_size": log_size})

    def _events(self, q):
        since = int(q.get("since", ["0"])[0]); wait = min(float(q.get("wait", ["0"])[0]), 45.0)
        if not (0 <= wait): wait = 0.0  # NaN/negative guard
        deadline = time.monotonic() + wait
        while True:
            store.refresh()
            with store.lock:
                since = min(since, len(store.lines))  # clamp so a truncated file can't strand the client
                batch = store.lines[since : since + 500]
            if batch or time.monotonic() >= deadline: break
            time.sleep(0.25)
        events = [{**e, "seq": since + i} for i, e in enumerate(batch)]
        self._json(200, {"events": events, "next": since + len(events)})

    def _log(self, q):
        offset = int(q.get("offset", ["0"])[0]); limit = int(q.get("limit", ["262144"])[0])
        try: size = os.stat(LOG_FILE).st_size
        except OSError: size = 0
        if offset > size: data = b""; offset = 0  # log recreated: client restarts
        else:
            with open(LOG_FILE, "rb") as f: f.seek(offset); data = f.read(limit)
        self.send_response(200); self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Log-Offset", str(offset + len(data))); self.send_header("X-Log-Size", str(size))
        self.end_headers(); self.wfile.write(data)

    def _snapshot(self, q):
        """Stream a model snapshot for PodTerm to run diagnostics off-pod."""
        try: step = int(q.get("step", [""])[0])
        except (ValueError, IndexError): return self._json(400, {"error": "bad step"})
        path = os.path.join(SNAPSHOT_DIR, f"step{step}.pt")  # filename built from an int — no path traversal
        try: size = os.path.getsize(path)
        except OSError: return self._json(404, {"error": "snapshot not found"})
        self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="step{step}.pt"')
        self.end_headers()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(262144), b""): self.wfile.write(chunk)

    def _snapshot_ack(self, q):
        """PodTerm acks the final snapshot download → release bootstrap.sh's teardown wait."""
        try:
            with open(ACK_FILE, "w") as f: f.write(q.get("step", [""])[0])
        except OSError: pass
        self._json(200, {"ok": True})


def main():
    if not TOKEN: print("WARNING: EVENTD_TOKEN unset — auth disabled", flush=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler); server.daemon_threads = True
    print(f"pod_eventd listening on :{PORT} events={EVENTS_FILE} log={LOG_FILE}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
