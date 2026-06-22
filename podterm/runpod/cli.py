"""RunPod interaction via the runpodctl CLI.

All RunPod control-plane operations (datacenters, GPUs, templates, pods, SSH)
shell out to the `runpodctl` binary. The only sanctioned non-CLI calls live in
podterm/runpod/api.py (read-only telemetry + machine logs).
"""

from __future__ import annotations

import json
import subprocess

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

# Flags whose *value* (the following arg) is secret-bearing and must never be
# echoed into an exception message, traceback, or log. The serialized template
# env passed via --env carries the per-run EVENTD_TOKEN; --password and any
# token/secret-named flag are masked defensively.
_SENSITIVE_FLAGS = frozenset({"--env", "--password", "--secret", "--token", "--api-key"})
_REDACTED = "<redacted>"


def _redact_args(args: tuple[str, ...] | list[str]) -> str:
    """Render command args for display, masking values of sensitive flags.

    The value immediately following a sensitive flag is replaced with
    ``<redacted>`` so secrets (e.g. the EVENTD_TOKEN inside ``--env <json>``)
    never reach logs or error bodies.
    """
    parts: list[str] = []
    mask_next = False
    for arg in args:
        if mask_next:
            parts.append(_REDACTED)
            mask_next = False
            continue
        parts.append(arg)
        if arg in _SENSITIVE_FLAGS:
            mask_next = True
    return " ".join(parts)


def _sensitive_values(args: tuple[str, ...] | list[str]) -> list[str]:
    """Collect secret substrings to scrub from any text runpodctl echoes back.

    For each sensitive flag, the following arg is itself a secret; when that arg
    is a serialized JSON env (``--env '{"EVENTD_TOKEN": "…", …}'``) its *values*
    are also scrubbed individually, so a token echoed bare (not as the whole
    JSON blob) is still masked. Longest-first so larger blobs are replaced
    before their fragments.
    """
    values: list[str] = []
    flag_seen = False
    for arg in args:
        if flag_seen:
            flag_seen = False
            if not arg:
                continue
            values.append(arg)
            try:
                parsed = json.loads(arg)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                values.extend(str(v) for v in parsed.values() if str(v))
            continue
        if arg in _SENSITIVE_FLAGS:
            flag_seen = True
    # Replace longer secrets first to avoid leaving a fragment of a larger one.
    return sorted(set(values), key=len, reverse=True)


def _scrub(text: str, secrets: list[str]) -> str:
    """Remove any known secret substrings from text (runpodctl may echo input)."""
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, _REDACTED)
    return text


class RpcError(RuntimeError):
    """A runpodctl invocation failed.

    Subclasses ``RuntimeError`` so existing ``except Exception`` / ``except
    RuntimeError`` call sites keep working. The string form is built solely
    from the *redacted* command, return code, and *scrubbed* stderr — it never
    includes raw secret-bearing argument values (``--env`` with the per-run
    ``EVENTD_TOKEN``, ``--password``, …), so credentials cannot leak into logs,
    tracebacks, or HTTP error bodies.
    """

    def __init__(self, command: str, returncode: int | None, stderr: str = "") -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = (stderr or "").strip()
        rc = "?" if returncode is None else str(returncode)
        msg = f"runpodctl {command} failed (exit {rc})"
        if self.stderr:
            msg += f": {self.stderr}"
        super().__init__(msg)


def _rpc(*args: str, timeout: int = 30) -> str:
    """Run a runpodctl command and return stdout.

    Raises :class:`RpcError` on any failure (non-zero exit, timeout, missing
    binary). Error messages are built from a redacted view of the arguments so
    secret-bearing flag values never reach logs or HTTP error bodies.
    """
    command = _redact_args(args)
    try:
        result = subprocess.run(
            ["runpodctl", *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RpcError(command, None, "runpodctl executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RpcError(command, None, f"timed out after {timeout}s") from exc
    if result.returncode != 0:
        # Scrub any secret arg values runpodctl may have echoed back into stderr.
        stderr = _scrub(result.stderr, _sensitive_values(args))
        raise RpcError(command, result.returncode, stderr)
    return result.stdout


def _rpc_json(*args: str, **kwargs: object) -> dict | list:
    raw = _rpc(*args, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Report position only (`exc` stringifies to "<msg>: line L column C
        # (char N)" — no document bytes) and a redacted command; never echo raw
        # args, which may carry secrets.
        raise RpcError(
            _redact_args(args), None, f"invalid JSON in runpodctl output: {exc}"
        ) from exc


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


def api_get_pod(pod_id: str, include_machine: bool = False) -> dict | None:
    try:
        args = ["pod", "get", pod_id]
        if include_machine:
            args.append("--include-machine")
        return _rpc_json(*args)
    except Exception:
        return None


def detect_redis_server() -> str | None:
    """Find a running pod with 'redis' in the image and return ip:port."""
    for pod in api_list_pods():
        if pod.get("desiredStatus") != "RUNNING":
            continue
        if "redis" not in (pod.get("imageName") or "").lower():
            continue
        details = api_get_pod(pod["id"])
        if not details:
            continue
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


def api_get_ssh_info(pod_id: str) -> dict | None:
    """Get SSH connection info for a pod."""
    try:
        return _rpc_json("ssh", "info", pod_id)
    except Exception:
        return None
