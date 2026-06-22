"""Read-only direct RunPod API calls the runpodctl CLI can't cover.

Two sanctioned exceptions to the CLI-only rule (see CLAUDE.md):
  - api_get_telemetry()    GraphQL myself.pods.runtime — utilization fields
  - api_get_machine_logs() hapi.runpod.net — host-level boot/image-pull lines

Both are read-only and best-effort (return empty on any failure).
"""

from __future__ import annotations

import enum
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from podterm.runpod.console import _BROWSER_UA, console_jwt_fresh

log = logging.getLogger("podterm.runpod.api")

# ---------------------------------------------------------------------------
# Pod lookup result (distinguishes "gone" from "couldn't reach RunPod")
# ---------------------------------------------------------------------------
#
# `api_get_pod()` (cli.py) collapses every failure to None, which a caller
# cannot tell apart from "the pod genuinely no longer exists". The poller's
# pod-gone check finalizes a run on a couple of such Nones, so a transient CLI
# or network hiccup could finalize a still-running pod. `api_get_pod_lookup()`
# below preserves the distinction:
#   - found       the pod exists; `.pod` holds its dict
#   - not_found   RunPod answered definitively that the pod is absent (404)
#   - unavailable we couldn't get a definitive answer (CLI error, timeout,
#                 network failure, invalid JSON) — never proof the pod is gone


class PodLookupState(str, enum.Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PodLookupResult:
    state: PodLookupState
    pod: dict | None = None

    @property
    def is_definitively_gone(self) -> bool:
        """True only when RunPod confirmed the pod is absent — never on outage."""
        return self.state is PodLookupState.NOT_FOUND


# runpodctl phrases that indicate a definitive "this pod does not exist", as
# opposed to a transport/auth/parse failure. Matched case-insensitively.
_POD_ABSENT_MARKERS = (
    "not found",
    "no pod",
    "does not exist",
    "doesn't exist",
    "no such pod",
)

# Transport/auth/transient phrases. These take PRECEDENCE over absence markers:
# misreading a transient failure as "gone" would finalize a still-running run
# (the dangerous direction), so anything that smells transient stays UNAVAILABLE
# even when it also happens to contain an absence word like "not found".
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "connect",
    "network",
    "unreachable",
    "temporarily",
    "try again",
    "rate limit",
    "too many requests",
    "unauthorized",
    "forbidden",
    "401",
    "403",
    "500",
    "502",
    "503",
    "504",
    "bad gateway",
    "gateway",
    "service unavailable",
    "internal server error",
    "eof",
    "reset by peer",
    "expecting value",  # json.JSONDecodeError text — empty/garbage CLI output
)


def _classify_lookup_error(exc: Exception) -> PodLookupState:
    msg = str(exc).lower()
    # Transient signals win: never finalize a live run on a flaky lookup.
    if any(marker in msg for marker in _TRANSIENT_MARKERS):
        return PodLookupState.UNAVAILABLE
    if any(marker in msg for marker in _POD_ABSENT_MARKERS):
        return PodLookupState.NOT_FOUND
    # Unknown error shape → assume transient (the safe direction).
    return PodLookupState.UNAVAILABLE


def _pod_from_payload(payload: object) -> dict | None:
    """Extract the pod dict from a `pod get` result (dict, or list of one)."""
    if isinstance(payload, dict):
        return payload or None
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item:
                return item
    return None


def api_get_pod_lookup(pod_id: str, include_machine: bool = False) -> PodLookupResult:
    """Look up a pod, distinguishing "gone" from "couldn't reach RunPod".

    Unlike `api_get_pod()`, this never reports `not_found` for a transport
    failure: callers that finalize runs on a missing pod must only act on a
    definitive `not_found`.
    """
    # Imported locally so this read-only api module stays independent of the
    # CLI module's import-time surface and to avoid any import cycle.
    from podterm.runpod.cli import _rpc_json

    args = ["pod", "get", pod_id]
    if include_machine:
        args.append("--include-machine")
    try:
        payload = _rpc_json(*args)
    except Exception as exc:  # RuntimeError / RpcError / json / timeout
        state = _classify_lookup_error(exc)
        if state is PodLookupState.UNAVAILABLE:
            log.warning("pod lookup unavailable pod=%s: %s", pod_id, exc)
        return PodLookupResult(state)
    pod = _pod_from_payload(payload)
    if pod is None:
        # A successful call that returned an empty/null result means the pod is
        # genuinely absent.
        return PodLookupResult(PodLookupState.NOT_FOUND)
    return PodLookupResult(PodLookupState.FOUND, pod)


def terminate_best_effort(pod_id: str) -> None:
    """Terminate a pod, swallowing all errors — used for launch-saga rollback."""
    try:
        from podterm.runpod.cli import api_terminate_pod

        api_terminate_pod(pod_id)
    except Exception:
        log.exception("best-effort terminate failed pod=%s", pod_id)


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
