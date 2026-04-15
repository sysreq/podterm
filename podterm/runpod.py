"""RunPod API layer — all interaction via the runpodctl CLI."""

from __future__ import annotations

import json
import subprocess

from podterm.config import (
    DEFAULT_CLOUD_TYPE,
    DEFAULT_CONTAINER_DISK_GB,
    DEFAULT_DATACENTER,
    DEFAULT_GPU,
    DEFAULT_VOLUME_DISK_GB,
    POD_PREFIX,
    TEMPLATE_NAME,
)

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
             "--image", image, "--env", env_json, "--ports", "22/tcp")
        return existing_id
    data = _rpc_json("template", "create",
                     "--name", TEMPLATE_NAME, "--image", image,
                     "--container-disk-in-gb", str(DEFAULT_CONTAINER_DISK_GB),
                     "--volume-in-gb", str(DEFAULT_VOLUME_DISK_GB),
                     "--env", env_json, "--ports", "22/tcp")
    return data["id"]


# ---------------------------------------------------------------------------
# Pods
# ---------------------------------------------------------------------------


def api_list_pods() -> list[dict]:
    return _rpc_json("pod", "list", "--all")


def get_gpt_golf_pods() -> list[dict]:
    return [p for p in api_list_pods() if p.get("name", "").startswith(f"{POD_PREFIX}-")]


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
           "--global-networking",
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


def api_get_pod(pod_id: str) -> dict | None:
    try:
        return _rpc_json("pod", "get", pod_id)
    except Exception:
        return None


def api_get_ssh_info(pod_id: str) -> dict | None:
    """Get SSH connection info for a pod."""
    try:
        return _rpc_json("ssh", "info", pod_id)
    except Exception:
        return None
