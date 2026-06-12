"""Pod event polling — pulls structured events + raw log from the on-pod daemon.

The pod runs gpt-golf's scripts/pod_eventd.py (port 8765, exposed as /http in the
RunPod template), reachable at https://{pod_id}-8765.proxy.runpod.net. Auth is a
per-launch bearer token injected via the EVENTD_TOKEN template env var.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request

from podterm.config import EVENTD_PORT
from podterm.helpers import get_ssh_key_path
from podterm.runpod import api_get_pod, api_get_ssh_info

# Queue items are (pod_id, kind, payload) tuples; kind is "log" or "event"
LogQueue = queue.Queue[tuple[str, str, dict]]


class PodPoller(threading.Thread):
    """Background thread: waits for a pod's event daemon, then pulls events + raw log."""

    def __init__(
        self,
        pod_id: str,
        pod_name: str,
        token: str,
        log_queue: LogQueue,
        base_url: str | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"poll-{pod_id[:8]}")
        self.pod_id = pod_id
        self.pod_name = pod_name
        self.token = token
        self.log_queue = log_queue
        override = os.environ.get("PODTERM_EVENTS_URL")
        self.base_url = base_url or override or f"https://{pod_id}-{EVENTD_PORT}.proxy.runpod.net"
        # Local-test hook: an explicit URL means there is no real RunPod pod to query
        self.skip_pod_checks = override is not None and base_url is None
        self._stop_event = threading.Event()
        self._local_log = None
        self._log_lock = threading.Lock()  # emit_log is called from both pull loops

    def stop(self) -> None:
        self._stop_event.set()

    def emit_log(self, text: str) -> None:
        self.log_queue.put((self.pod_id, "log", {"line": text}))
        with self._log_lock:
            if self._local_log:
                self._local_log.write(text + "\n")
                self._local_log.flush()

    # -- HTTP ---------------------------------------------------------------

    def _request(self, path: str, timeout: float) -> tuple[int, object, bytes]:
        """GET base_url+path. Returns (status, headers, body); status 0 on connection error."""
        # Cloudflare (in front of proxy.runpod.net) 403s the default Python-urllib UA
        req = urllib.request.Request(
            self.base_url + path,
            headers={"Authorization": f"Bearer {self.token}", "User-Agent": "podterm/2.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, b""
        except Exception:
            return 0, None, b""

    # -- Lifecycle ----------------------------------------------------------

    def run(self) -> None:
        log_dir = os.path.join(os.getcwd(), ".cache", "logs")
        os.makedirs(log_dir, exist_ok=True)
        safe_name = (self.pod_name or self.pod_id).replace("/", "-")
        self._local_log = open(os.path.join(log_dir, f"{safe_name}.log"), "a")
        log_thread = None
        try:
            self.emit_log(f"Waiting for pod {self.pod_name}...")
            if not self.skip_pod_checks and not self._wait_for_running():
                return
            log_size = self._wait_for_daemon()
            if log_size is None:
                return
            if not self.skip_pod_checks:
                self._emit_ssh_info()
            # Raw log tails from near the end (like tail -n 200); events replay from 0
            log_thread = threading.Thread(
                target=self._log_loop, args=(max(0, log_size - 65536),),
                daemon=True, name=f"log-{self.pod_id[:8]}",
            )
            log_thread.start()
            self._events_loop()
        finally:
            self._stop_event.set()
            if log_thread:
                log_thread.join()
            self._local_log.close()
            self._local_log = None

    def _wait_for_running(self) -> bool:
        """Poll until the pod reports RUNNING (~6 min). None from the API is transient."""
        for tick in range(120):
            if self._stop_event.is_set():
                return False
            pod = api_get_pod(self.pod_id)
            if pod:
                status = pod.get("desiredStatus", "UNKNOWN")
                if status in ("EXITED", "TERMINATED"):
                    self.emit_log(f"Pod {status}.")
                    return False
                if status == "RUNNING":
                    return True
                if tick % 3 == 0:
                    self.emit_log(f"  Status: {status}  (waiting...)")
            time.sleep(3)
        self.emit_log("Timed out waiting for pod to start.")
        return False

    def _wait_for_daemon(self) -> int | None:
        """Poll /health until the daemon answers (proxy 502/503s while the pod boots).

        Up to 15 min — covers image pull. Returns the daemon's log_size, or None.
        """
        deadline = time.monotonic() + 900
        announced = False
        tick = 0
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                return None
            status, _, body = self._request("/health", timeout=10)
            if status == 200:
                try:
                    health = json.loads(body)
                except ValueError:
                    health = {}
                self.emit_log(f"Event daemon connected  |  {self.base_url}")
                return int(health.get("log_size", 0))
            if status == 401:
                self.emit_log("Event daemon auth failed (token mismatch).")
                return None
            if not announced:
                self.emit_log("  Waiting for event daemon (pod booting)...")
                announced = True
            tick += 1
            if not self.skip_pod_checks and tick % 10 == 0:
                pod = api_get_pod(self.pod_id)
                if pod and pod.get("desiredStatus") in ("EXITED", "TERMINATED"):
                    self.emit_log(f"Pod {pod['desiredStatus']}.")
                    return None
            time.sleep(3)
        self.emit_log("Timed out waiting for event daemon.")
        return None

    def _emit_ssh_info(self) -> None:
        """Surface the SSH convenience line (debug access is still via plain SSH)."""
        info = api_get_ssh_info(self.pod_id) or {}
        ip, port = info.get("ip"), info.get("port")
        if not (ip and port):
            return
        key = get_ssh_key_path() or (info.get("ssh_key") or {}).get("path", "~/.ssh/id_ed25519")
        self.emit_log(f"SSH ready  |  ssh root@{ip} -p {port} -i {key}")

    # -- Pull loops ----------------------------------------------------------

    def _events_loop(self) -> None:
        """Long-poll /events from cursor 0. Replays are safe: db writes are idempotent."""
        cur = 0
        finished = False
        gone_checks = 0
        backoff = 2.0
        while not self._stop_event.is_set():
            status, _, body = self._request(f"/events?since={cur}&wait=25", timeout=40)
            data = None
            if status == 200:
                try:
                    data = json.loads(body)
                except ValueError:
                    data = None  # 200 with a non-JSON body (proxy interstitial): treat as transient
            if data is not None:
                gone_checks = 0
                backoff = 2.0
                for ev in data.get("events", []):
                    if ev.get("t") == "phase" and "Training finished" in str(ev.get("phase", "")):
                        finished = True
                    self.log_queue.put((self.pod_id, "event", ev))
                cur = int(data.get("next", cur))
                continue
            if status == 401:
                self.emit_log("Event daemon auth failed (token mismatch).")
                return
            # Transient failure (conn error / proxy 5xx): check whether the pod is gone.
            # Two consecutive confirmations so one flaky runpodctl call can't end the run.
            if self.skip_pod_checks:
                pod_gone = False
            else:
                pod = api_get_pod(self.pod_id)
                pod_gone = pod is None or pod.get("desiredStatus") in ("EXITED", "TERMINATED")
            if pod_gone:
                gone_checks += 1
                if gone_checks >= 2:
                    if not finished:
                        self.log_queue.put((self.pod_id, "event", {"t": "pod_gone"}))
                    self.emit_log("Pod stopped.")
                    return
            else:
                gone_checks = 0
            self._stop_event.wait(backoff)
            backoff = min(backoff * 2, 30.0)

    def _log_loop(self, offset: int) -> None:
        """Poll /log every 2s for raw log bytes; emit complete lines."""
        buf = b""
        while not self._stop_event.is_set():
            status, headers, body = self._request(f"/log?offset={offset}&limit=262144", timeout=15)
            if status == 200:
                new_offset = int(headers.get("X-Log-Offset", offset + len(body)))
                if new_offset < offset:  # log file recreated — daemon reset us to 0
                    buf = b""
                offset = new_offset
                if body:
                    buf += body
                    *lines, buf = buf.split(b"\n")
                    for raw in lines:
                        self.emit_log(raw.decode("utf-8", errors="replace").rstrip("\r"))
            self._stop_event.wait(2.0)
