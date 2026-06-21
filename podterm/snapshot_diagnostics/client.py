from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

from podterm.config import EVENTD_PORT

from .state import _CACHE_DIR


def base_url(pod_id: str) -> str:
    override = os.environ.get("PODTERM_EVENTS_URL")
    return override or f"https://{pod_id}-{EVENTD_PORT}.proxy.runpod.net"


def request(url: str, token: str, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": "podterm/2.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:
        return 0, b""


def download_snapshot(pod_id: str, token: str, payload: dict) -> Path:
    step = int(payload.get("step", 0))
    dest_dir = _CACHE_DIR / pod_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"step{step}.pt"
    status, body = request(f"{base_url(pod_id)}/snapshot?step={step}", token, timeout=120)
    if status != 200 or not body:
        raise RuntimeError(f"snapshot download failed (status={status}, bytes={len(body)})")
    want = payload.get("sha256")
    if want and hashlib.sha256(body).hexdigest() != want:
        raise RuntimeError("snapshot sha256 mismatch — refusing to diagnose a corrupt checkpoint")
    dest.write_bytes(body)
    return dest
