"""RunPod API layer — runpodctl CLI, plus one read-only GraphQL query for
utilization telemetry (the REST API behind runpodctl has no telemetry fields)."""

from __future__ import annotations

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
