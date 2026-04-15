"""SSH connection and log streaming via background threads."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time

from podterm.helpers import get_ssh_key_path
from podterm.runpod import api_get_pod, api_get_ssh_info

# ---------------------------------------------------------------------------
# Log message types
# ---------------------------------------------------------------------------

# Queue items are (pod_id, line_text) tuples
LogQueue = queue.Queue[tuple[str, str]]


# ---------------------------------------------------------------------------
# SSH tail thread
# ---------------------------------------------------------------------------


class SshTailThread(threading.Thread):
    """Background thread that waits for a pod's SSH to come up, then tails its log."""

    def __init__(self, pod_id: str, pod_name: str, log_queue: LogQueue) -> None:
        super().__init__(daemon=True, name=f"ssh-{pod_id[:8]}")
        self.pod_id = pod_id
        self.pod_name = pod_name
        self.log_queue = log_queue
        self._proc: subprocess.Popen | None = None
        self._stop_event = threading.Event()
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self.ssh_key: str | None = None

    def emit(self, text: str) -> None:
        self.log_queue.put((self.pod_id, text))

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self) -> None:
        # Also write to a local log file
        log_dir = os.path.join(os.getcwd(), ".cache", "logs")
        os.makedirs(log_dir, exist_ok=True)
        safe_name = (self.pod_name or self.pod_id).replace("/", "-")
        local_log_path = os.path.join(log_dir, f"{safe_name}.log")
        local_log = open(local_log_path, "a")

        def emit(text: str) -> None:
            self.log_queue.put((self.pod_id, text))
            local_log.write(text + "\n")
            local_log.flush()

        try:
            emit(f"Waiting for pod {self.pod_name}...")
            self._wait_for_ssh(emit)
            if self._stop_event.is_set() or not self.ssh_host:
                return
            self._tail(emit)
        finally:
            local_log.close()

    def _wait_for_ssh(self, emit) -> None:
        """Poll until SSH info is available or timeout (6 min)."""
        for tick in range(120):
            if self._stop_event.is_set():
                return
            try:
                pod = api_get_pod(self.pod_id)
            except Exception:
                time.sleep(5)
                continue

            if not pod:
                emit("Pod not found.")
                return

            status = pod.get("desiredStatus", "UNKNOWN")
            if status in ("EXITED", "TERMINATED"):
                emit(f"Pod {status}.")
                return

            try:
                ssh_info = api_get_ssh_info(self.pod_id)
                if ssh_info:
                    self.ssh_host = ssh_info.get("ip")
                    self.ssh_port = ssh_info.get("port")
            except Exception:
                pass

            if self.ssh_host and self.ssh_port:
                break
            if tick % 3 == 0:
                emit(f"  Status: {status}  (waiting for SSH...)")
            time.sleep(3)
        else:
            emit("Timed out waiting for SSH.")
            return

        # Resolve SSH key
        self.ssh_key = get_ssh_key_path()
        if not self.ssh_key:
            ssh_info = api_get_ssh_info(self.pod_id) or {}
            key_info = ssh_info.get("ssh_key") or {}
            self.ssh_key = os.path.expanduser(key_info.get("path", "~/.ssh/id_ed25519"))

        emit(f"SSH ready  |  ssh root@{self.ssh_host} -p {self.ssh_port} -i {self.ssh_key}")

    def _tail(self, emit) -> None:
        """Open SSH and tail the remote training log."""
        try:
            self._proc = subprocess.Popen(
                [
                    "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                    "-i", self.ssh_key, "-p", str(self.ssh_port), f"root@{self.ssh_host}",
                    f"LOG=/workspace/logs/{self.pod_id}.log; "
                    "for i in $(seq 1 120); do [ -f \"$LOG\" ] && break; sleep 1; done; "
                    "[ -f \"$LOG\" ] || { echo 'Log file never appeared'; exit 1; }; "
                    "tail -n 200 -f \"$LOG\"",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in self._proc.stdout:
                if self._stop_event.is_set():
                    break
                emit(line.rstrip())
            self._proc.wait()
            emit(f"SSH exited ({self._proc.returncode})")
        except Exception as exc:
            emit(f"SSH error: {exc}")
        finally:
            self._proc = None
