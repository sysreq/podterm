from __future__ import annotations

from podterm.boot import PullTracker
from podterm.config import boot_progress_enabled
from podterm.helpers import get_ssh_key_path
from podterm.runpod import api_get_machine_logs, api_get_pod, api_get_ssh_info

from .streams import Request
from .util import deadline_ticks, is_terminal_status, parse_json_body, pod_status, poll_attempts

POD_START_ATTEMPTS = 120
POD_POLL_SEC = 3.0
POD_STATUS_LOG_EVERY = 3
DAEMON_WAIT_SEC = 900.0
DAEMON_HEALTH_TIMEOUT_SEC = 10.0
DAEMON_POD_CHECK_EVERY = 10


class PodStartup:
    """Wait for RunPod and the pod-side event daemon to become usable."""

    def __init__(
        self,
        pod_id: str,
        base_url: str,
        skip_pod_checks: bool,
        request: Request,
        emit_log,
        emit_event,
        stop_event,
    ) -> None:
        self.pod_id = pod_id
        self.base_url = base_url
        self.skip_pod_checks = skip_pod_checks
        self.request = request
        self.emit_log = emit_log
        self.emit_event = emit_event
        self.stop_event = stop_event
        self.pull = PullTracker()
        self.boot_enabled = boot_progress_enabled()

    def pod_gone(self) -> bool:
        if self.skip_pod_checks:
            return False
        pod = api_get_pod(self.pod_id)
        return pod is None or is_terminal_status(pod_status(pod))

    def boot_tick(self) -> None:
        if not self.boot_enabled or self.skip_pod_checks:
            return
        lines = api_get_machine_logs(self.pod_id)
        if not lines:
            return
        new_messages, changed = self.pull.ingest(lines)
        for msg in new_messages:
            self.emit_log(f"[host] {msg}")
        if changed:
            self.emit_event(self.pull.snapshot())

    def boot_done(self, message: str) -> None:
        if self.boot_enabled:
            self.emit_event(self.pull.snapshot(done=True, message=message))

    def wait_for_running(self) -> bool:
        for tick in poll_attempts(self.stop_event, POD_START_ATTEMPTS, POD_POLL_SEC):
            self.boot_tick()
            pod = api_get_pod(self.pod_id)
            match pod_status(pod):
                case None:
                    continue
                case "RUNNING":
                    return True
                case "EXITED" | "TERMINATED" as status:
                    self.emit_log(f"Pod {status}.")
                    return False
                case status if tick % POD_STATUS_LOG_EVERY == 0:
                    self.emit_log(f"  Status: {status}  (waiting...)")
        if self.stop_event.is_set():
            return False
        self.emit_log("Timed out waiting for pod to start.")
        return False

    def wait_for_daemon(self) -> int | None:
        announced = False
        for tick in deadline_ticks(self.stop_event, DAEMON_WAIT_SEC, POD_POLL_SEC):
            self.boot_tick()
            status, _, body = self.request("/health", DAEMON_HEALTH_TIMEOUT_SEC)
            match status:
                case 200:
                    health = parse_json_body(body) or {}
                    self.emit_log(f"Event daemon connected  |  {self.base_url}")
                    return int(health.get("log_size", 0))
                case 401:
                    self.emit_log("Event daemon auth failed (token mismatch).")
                    return None
            if not announced:
                self.emit_log("  Waiting for event daemon (pod booting)...")
                announced = True
            if self._daemon_pod_stopped(tick):
                return None
        if self.stop_event.is_set():
            return None
        self.emit_log("Timed out waiting for event daemon.")
        return None

    def emit_ssh_info(self) -> None:
        info = api_get_ssh_info(self.pod_id) or {}
        ip, port = info.get("ip"), info.get("port")
        if not (ip and port):
            return
        key = get_ssh_key_path() or (info.get("ssh_key") or {}).get("path", "~/.ssh/id_ed25519")
        self.emit_log(f"SSH ready  |  ssh root@{ip} -p {port} -i {key}")

    def _daemon_pod_stopped(self, tick: int) -> bool:
        if self.skip_pod_checks or (tick + 1) % DAEMON_POD_CHECK_EVERY != 0:
            return False
        pod = api_get_pod(self.pod_id)
        status = pod_status(pod)
        if not is_terminal_status(status):
            return False
        self.emit_log(f"Pod {status}.")
        return True
