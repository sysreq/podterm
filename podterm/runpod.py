"""RunPod API layer — runpodctl CLI, plus two read-only direct API calls the
CLI can't cover: a GraphQL query for utilization telemetry and the hapi
machine-log endpoint for boot/image-pull progress."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

from podterm.config import (
    DEFAULT_CLOUD_TYPE,
    DEFAULT_CONTAINER_DISK_GB,
    DEFAULT_DATACENTER,
    DEFAULT_GPU,
    DEFAULT_VOLUME_DISK_GB,
    EVENTD_PORT,
    POD_PREFIX,
    TEMPLATE_NAME,
)

TEMPLATE_PORTS = f"22/tcp,{EVENTD_PORT}/http"

# ---------------------------------------------------------------------------
# Low-level RPC
# ---------------------------------------------------------------------------


def _rpc(*args: str, timeout: int = 30) -> str:
    """Run a runpodctl command and return stdout. Raises on failure."""
    result = subprocess.run(
        ["runpodctl", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"runpodctl {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def _rpc_json(*args: str, **kwargs: object) -> dict | list:
    return json.loads(_rpc(*args, **kwargs))


# ---------------------------------------------------------------------------
# Datacenters & GPUs
# ---------------------------------------------------------------------------


def get_datacenters() -> list[dict]:
    """Fetch datacenters that have GPUs available."""
    try:
        dcs = _rpc_json("datacenter", "list")
        return [dc for dc in dcs if dc.get("gpuAvailability")]
    except Exception:
        return [{"id": DEFAULT_DATACENTER, "name": DEFAULT_DATACENTER, "location": "United States",
                 "gpuAvailability": [{"displayName": "RTX 5090", "gpuId": DEFAULT_GPU, "stockStatus": "High"}]}]


def get_network_volume(datacenter: str) -> str | None:
    """Return the first network volume ID in the given datacenter, or None."""
    try:
        for vol in _rpc_json("network-volume", "list"):
            if vol.get("dataCenterId") == datacenter:
                return vol["id"]
    except Exception:
        pass
    return None


def get_available_gpus(datacenter: str = DEFAULT_DATACENTER) -> list[tuple[str, str]]:
    """Fetch GPUs available at a specific datacenter."""
    try:
        for dc in _rpc_json("datacenter", "list"):
            if dc.get("id") == datacenter:
                gpus = dc.get("gpuAvailability") or []
                return [(f"{g['displayName']} ({g['stockStatus']})", g["gpuId"])
                        for g in gpus]
        return [("RTX 5090", DEFAULT_GPU)]
    except Exception:
        return [("RTX 5090", DEFAULT_GPU)]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def find_template() -> str | None:
    try:
        templates = _rpc_json("template", "list", "--type", "user")
        if not isinstance(templates, list):
            return None
        for t in templates:
            if t.get("name") == TEMPLATE_NAME:
                return t["id"]
    except Exception:
        pass
    return None


def create_or_update_template(image: str, env_dict: dict) -> str:
    env_json = json.dumps(env_dict)
    existing_id = find_template()
    if existing_id:
        _rpc("template", "update", existing_id,
             "--image", image, "--env", env_json, "--ports", TEMPLATE_PORTS)
        return existing_id
    data = _rpc_json("template", "create",
                     "--name", TEMPLATE_NAME, "--image", image,
                     "--container-disk-in-gb", str(DEFAULT_CONTAINER_DISK_GB),
                     "--volume-in-gb", str(DEFAULT_VOLUME_DISK_GB),
                     "--env", env_json, "--ports", TEMPLATE_PORTS)
    return data["id"]


# ---------------------------------------------------------------------------
# Pods
# ---------------------------------------------------------------------------


def api_list_pods() -> list[dict]:
    return _rpc_json("pod", "list", "--all")


def get_gpt_golf_pods() -> list[dict]:
    return [p for p in api_list_pods() if p.get("name", "").startswith(f"{POD_PREFIX}-")]


def api_get_pod(pod_id: str) -> dict:
    return _rpc_json("pod", "get", pod_id)


def detect_redis_server() -> str | None:
    """Find a running pod with 'redis' in the image and return ip:port."""
    for pod in api_list_pods():
        if pod.get("desiredStatus") != "RUNNING":
            continue
        if "redis" not in (pod.get("imageName") or "").lower():
            continue
        details = api_get_pod(pod["id"])
        ssh = details.get("ssh") or {}
        ip = ssh.get("ip")
        port = ssh.get("port")
        if ip and port:
            return f"{ip}:{port + 1}"
    return None


def api_create_pod(
    name: str,
    gpu: str,
    template_id: str,
    cloud_type: str = DEFAULT_CLOUD_TYPE,
    datacenter: str | None = None,
    network_volume: str | None = None,
    gpu_count: int = 1,
) -> dict:
    cmd = ["pod", "create",
           "--name", name, "--gpu-id", gpu, "--template-id", template_id,
           "--cloud-type", cloud_type,
           "--container-disk-in-gb", str(DEFAULT_CONTAINER_DISK_GB),
           "--volume-in-gb", str(DEFAULT_VOLUME_DISK_GB)]
    if gpu_count > 1:
        cmd += ["--gpu-count", str(gpu_count)]
    if datacenter:
        cmd += ["--data-center-ids", datacenter]
    if network_volume:
        cmd += ["--network-volume-id", network_volume]
    return _rpc_json(*cmd)


def api_terminate_pod(pod_id: str) -> None:
    _rpc("pod", "delete", pod_id)


def api_get_pod(pod_id: str, include_machine: bool = False) -> dict | None:
    try:
        args = ["pod", "get", pod_id]
        if include_machine:
            args.append("--include-machine")
        return _rpc_json(*args)
    except Exception:
        return None


def api_get_ssh_info(pod_id: str) -> dict | None:
    """Get SSH connection info for a pod."""
    try:
        return _rpc_json("ssh", "info", pod_id)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Machine logs (host-level: volume create, image pull, container start)
#
# The hapi.runpod.net logs endpoint is the console's internal backend and only
# accepts a short-lived (~60s) Clerk *console* session JWT — the RunPod API key
# gets 403. We mint a fresh JWT on demand from the user's `__client` cookie
# (RUNPOD_CONSOLE_CLIENT_COOKIE, set in .env), the same way the console does.
# All of this is best-effort: any failure returns [] and the boot panel just
# never appears. See memory hapi-logs-auth / chrome-cookie-read-blocked.
# ---------------------------------------------------------------------------

_CLERK_BASE = "https://clerk.runpod.io/v1"
_CLERK_QS = "__clerk_api_version=2025-11-10&_clerk_js_version=5.125.13"
_HAPI_LOGS = "https://hapi.runpod.net/v1/pod/{}/logs"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"

# Cached across calls: the derived session id, and the last minted JWT + its
# expiry (epoch seconds). Tokens live ~60s; re-mint a few seconds early.
_console_sid: str | None = None
_console_jwt: tuple[str, float] | None = None  # (jwt, exp_epoch)


def _console_cookie() -> str | None:
    return os.environ.get("RUNPOD_CONSOLE_CLIENT_COOKIE") or None


def _clerk_request(path: str, cookie: str) -> dict:
    req = urllib.request.Request(
        f"{_CLERK_BASE}/{path}?{_CLERK_QS}",
        data=b"",  # POST; GET endpoints tolerate an empty body here
        headers={
            "Origin": "https://console.runpod.io",
            "Referer": "https://console.runpod.io/",
            "User-Agent": _BROWSER_UA,
            "Cookie": f"__client={cookie}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _console_session_id(cookie: str) -> str | None:
    global _console_sid
    if _console_sid:
        return _console_sid
    try:
        # GET /client lists the active session(s); empty-body POST is rejected,
        # so issue it as a real GET via a tweaked request.
        req = urllib.request.Request(
            f"{_CLERK_BASE}/client?{_CLERK_QS}",
            headers={"Origin": "https://console.runpod.io", "User-Agent": _BROWSER_UA,
                     "Cookie": f"__client={cookie}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        r = data.get("response") or data
        _console_sid = r.get("last_active_session_id") or (
            (r.get("sessions") or [{}])[0].get("id"))
    except Exception:
        _console_sid = None
    return _console_sid


def _console_jwt_fresh() -> str | None:
    """Return a valid console JWT, minting (and caching) one if needed."""
    global _console_jwt
    import time as _time
    if _console_jwt and _console_jwt[1] - _time.time() > 8:
        return _console_jwt[0]
    cookie = _console_cookie()
    if not cookie:
        return None
    sid = _console_session_id(cookie)
    if not sid:
        return None
    try:
        data = _clerk_request(f"client/sessions/{sid}/tokens", cookie)
        jwt = data.get("jwt")
    except Exception:
        return None
    if not jwt:
        return None
    # Trust the JWT's own exp claim rather than guessing a fixed TTL.
    exp = _time.time() + 55
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp", exp)
    except Exception:
        pass
    _console_jwt = (jwt, float(exp))
    return jwt


def api_get_machine_logs(pod_id: str) -> list[str]:
    """Host+container boot log lines for a pod (image pull, container start).

    Covers the window before the in-container event daemon is reachable, which
    neither runpodctl nor the daemon can see. Best-effort: returns [] on any
    failure (no console cookie configured, token mint failed, pod gone, …).
    """
    jwt = _console_jwt_fresh()
    if not jwt:
        return []
    req = urllib.request.Request(
        _HAPI_LOGS.format(pod_id),
        headers={"Authorization": f"Bearer {jwt}", "Origin": "https://console.runpod.io",
                 "User-Agent": _BROWSER_UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []
    # hapi returns {"container": [...]} (and/or other buckets); flatten to lines.
    if isinstance(data, dict):
        lines: list[str] = []
        for v in data.values():
            if isinstance(v, list):
                lines.extend(str(x) for x in v)
        return lines
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Utilization telemetry (GraphQL)
# ---------------------------------------------------------------------------

_GRAPHQL_URL = "https://api.runpod.io/graphql"

_TELEMETRY_QUERY = """query {
  myself { pods { id desiredStatus vcpuCount memoryInGb
    machine { cpuType { displayName } gpuType { memoryInGb } }
    runtime {
      uptimeInSeconds
      container { cpuPercent memoryPercent }
      gpus { gpuUtilPercent memoryUtilPercent }
    }
  } }
}"""


def _api_key() -> str | None:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    try:
        raw = (Path.home() / ".runpod" / "config.toml").read_text()
        m = re.search(r"apikey\s*=\s*['\"]?([A-Za-z0-9_\-]+)", raw, re.IGNORECASE)
        return m.group(1) if m else None
    except Exception:
        return None


def api_get_telemetry() -> dict[str, dict]:
    """Live CPU/GPU/memory utilization for all pods, keyed by pod id.

    Pods without a runtime (stopped/booting) are omitted. Multi-GPU pods
    report the mean across GPUs.
    """
    key = _api_key()
    if not key:
        return {}
    req = urllib.request.Request(
        _GRAPHQL_URL,
        data=json.dumps({"query": _TELEMETRY_QUERY}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "podterm/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    out: dict[str, dict] = {}
    for pod in (((data.get("data") or {}).get("myself") or {}).get("pods") or []):
        rt = pod.get("runtime")
        if not rt:
            continue
        gpus = rt.get("gpus") or []
        container = rt.get("container") or {}
        machine = pod.get("machine") or {}
        cpu_type = machine.get("cpuType") or {}
        gpu_type = machine.get("gpuType") or {}
        out[pod["id"]] = {
            "uptime_s": rt.get("uptimeInSeconds"),
            "cpu_pct": container.get("cpuPercent"),
            "mem_pct": container.get("memoryPercent"),
            "gpu_util_pct": sum(g.get("gpuUtilPercent") or 0 for g in gpus) / len(gpus) if gpus else None,
            "gpu_mem_pct": sum(g.get("memoryUtilPercent") or 0 for g in gpus) / len(gpus) if gpus else None,
            "gpu_count": len(gpus),
            # Static context riding along so the percentages mean something
            "cpu_name": cpu_type.get("displayName"),
            "vcpu_count": pod.get("vcpuCount"),
            "ram_total_gb": pod.get("memoryInGb"),
            "gpu_mem_total_gb": gpu_type.get("memoryInGb"),
        }
    return out
