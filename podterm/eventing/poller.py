from __future__ import annotations

import os
import queue
import threading

from podterm.config import EVENTD_PORT
from podterm.eventing.client import EventdClient
from podterm.eventing.startup import PodStartup
from podterm.eventing.streams import EventStream, LogTailer, LOG_TAIL_BYTES
from podterm.eventing.util import is_terminal_status, pod_status
from podterm.runpod.api import api_get_pod_lookup

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
        self._client = EventdClient(self.base_url, token)
        # Local-test hook: an explicit URL means there is no real RunPod pod to query
        self.skip_pod_checks = override is not None and base_url is None
        self._stop_event = threading.Event()
        self._local_log = None
        self._log_lock = threading.Lock()  # emit_log is called from both pull loops
        self._startup = PodStartup(
            self.pod_id,
            self.base_url,
            self.skip_pod_checks,
            lambda path, timeout: self._request(path, timeout),
            self.emit_log,
            self._emit_event,
            self._stop_event,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def emit_log(self, text: str) -> None:
        self.log_queue.put((self.pod_id, "log", {"line": text}))
        with self._log_lock:
            if self._local_log:
                self._local_log.write(text + "\n")
                self._local_log.flush()

    def _request(self, path: str, timeout: float) -> tuple[int, object, bytes]:
        return self._client.get(path, timeout)

    def _emit_event(self, payload: dict) -> None:
        self.log_queue.put((self.pod_id, "event", payload))

    def _check_pod_gone(self) -> bool:
        """Whether the pod is *definitively* gone — used to finalize the run.

        Only a confirmed absence (RunPod 404) or a terminal pod status counts.
        A transient CLI/network failure reports `unavailable`, which we treat as
        "still here": EventStream requires N consecutive confirmations before
        emitting `pod_gone`, and a single outage must never finalize a live run.
        """
        if self.skip_pod_checks:
            return False
        result = api_get_pod_lookup(self.pod_id)
        if result.is_definitively_gone:
            return True
        if result.pod is not None and is_terminal_status(pod_status(result.pod)):
            return True
        return False

    def _boot_tick(self) -> None:
        """Pull machine logs once; emit host lines + a pull snapshot on change."""
        self._startup.boot_tick()

    def _emit_boot_done(self, message: str) -> None:
        """Final pull snapshot so the UI leaves boot mode no matter how boot ended."""
        self._startup.boot_done(message)

    def run(self) -> None:
        log_dir = os.path.join(os.getcwd(), ".cache", "logs")
        os.makedirs(log_dir, exist_ok=True)
        safe_name = (self.pod_name or self.pod_id).replace("/", "-")
        self._local_log = open(os.path.join(log_dir, f"{safe_name}.log"), "a")
        log_thread = None
        try:
            self.emit_log(f"Waiting for pod {self.pod_name}...")
            if not self.skip_pod_checks and not self._wait_for_running():
                self._emit_boot_done("Pod stopped before reaching RUNNING.")
                return
            log_size = self._wait_for_daemon()
            if log_size is None:
                self._emit_boot_done("Gave up waiting for the event daemon.")
                return
            self._emit_boot_done("Container up — event daemon connected.")
            if not self.skip_pod_checks:
                self._emit_ssh_info()
            log_thread = threading.Thread(
                target=self._log_loop, args=(max(0, log_size - LOG_TAIL_BYTES),),
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
        return self._startup.wait_for_running()

    def _wait_for_daemon(self) -> int | None:
        """Poll /health until the daemon answers (proxy 502/503s while the pod boots).

        Up to 15 min — covers image pull. Returns the daemon's log_size, or None.
        """
        return self._startup.wait_for_daemon()

    def _emit_ssh_info(self) -> None:
        """Surface the SSH convenience line (debug access is still via plain SSH)."""
        self._startup.emit_ssh_info()

    def _events_loop(self) -> None:
        """Long-poll /events from cursor 0. Replays are safe: db writes are idempotent."""
        EventStream(
            self.pod_id,
            self._request,
            self.emit_log,
            self._emit_event,
            self._stop_event,
            self._check_pod_gone,
        ).run()

    def _log_loop(self, offset: int) -> None:
        """Poll /log every 2s for raw log bytes; emit complete lines."""
        LogTailer(self._request, self.emit_log, self._stop_event).run(offset)
