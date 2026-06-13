"""Read-only direct RunPod API calls the runpodctl CLI can't cover.

Two sanctioned exceptions to the CLI-only rule (see CLAUDE.md):
  - api_get_telemetry()    GraphQL myself.pods.runtime — utilization fields
  - api_get_machine_logs() hapi.runpod.net — host-level boot/image-pull lines

Both are read-only and best-effort (return empty on any failure).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from podterm.runpod.console import _BROWSER_UA, console_jwt_fresh

# ---------------------------------------------------------------------------
# Machine logs (host-level: volume create, image pull, container start)
# ---------------------------------------------------------------------------

_HAPI_LOGS = "https://hapi.runpod.net/v1/pod/{}/logs"


def api_get_machine_logs(pod_id: str) -> list[str]:
    """Host+container boot log lines for a pod (image pull, container start).

    Covers the window before the in-container event daemon is reachable, which
    neither runpodctl nor the daemon can see. Best-effort: returns [] on any
    failure (no console cookie configured, token mint failed, pod gone, …).
    """
    jwt = console_jwt_fresh()
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
