"""Off-pod snapshot diagnostics orchestrator.

On each `snapshot` event from a training run, pull the checkpoint from the pod's event daemon
(`/snapshot` endpoint), run the diagnostics suite locally — off the training GPU — persist the
result to the DB, and fan it out over SSE. A successful run waits (in bootstrap.sh) for the
final snapshot's ack before the pod stops, so we call `/snapshot/ack` once the final one lands.

The diagnostics computation lives in `podterm.diagnostics` but runs as a subprocess under
gpt-golf's torch env (it imports the model architecture, `train_gpt`/`config_gpt`, from the
gpt-golf checkout — a bare state_dict doesn't define the model).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

from podterm import db
from podterm.config import (
    DEFAULT_DATA_VARIANT,
    EVENTD_PORT,
    diag_cache_val_shard,
    diag_caps,
    diag_device,
    diag_python_cmd,
    diagnostics_enabled,
    gpt_golf_dir,
)
from podterm.sse import hub

log = logging.getLogger("podterm.snapshots")

_PODTERM_ROOT = Path(__file__).resolve().parents[1]  # repo root: holds the importable `podterm` pkg
_CACHE_DIR = Path.cwd() / ".cache" / "snapshots"
_SUBPROCESS_TIMEOUT = float(os.environ.get("DIAG_TIMEOUT", "900"))
_VAL_SHARD_TIMEOUT = float(os.environ.get("DIAG_VAL_SHARD_TIMEOUT", "600"))
_BPB_TOL = float(os.environ.get("DIAG_BPB_TOL", "0.02"))  # rel-err for off-pod vs on-pod BPB agreement

# Per-pod serialization + coalescing: a slow diagnostics run must not let snapshots pile up, so
# while one runs we keep only the newest pending snapshot per pod and process that next.
_locks: dict[str, asyncio.Lock] = {}
_pending: dict[str, dict] = {}
_device_cache: str | None = None

# Val-shard caching: serialize the (potential) download across worker threads and remember which
# variants are already materialized, so repeat snapshots don't re-shell-out once a variant is local.
_val_shard_lock = threading.Lock()
_val_shard_ready: set[str] = set()


# -- HTTP to the pod daemon (same auth/UA shape as PodPoller._request) -------


def _base_url(pod_id: str) -> str:
    override = os.environ.get("PODTERM_EVENTS_URL")  # local-test hook: a single concrete daemon
    return override or f"https://{pod_id}-{EVENTD_PORT}.proxy.runpod.net"


def _request(url: str, token: str, timeout: float) -> tuple[int, bytes]:
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


# -- Subprocess env ----------------------------------------------------------


def _gpt_golf_env(**extra: str) -> dict[str, str]:
    """Base env for subprocesses launched with cwd=gpt_golf_dir(). Drops PodTerm's own virtualenv
    vars so `uv run` cleanly resolves gpt-golf's project env instead of warning about (and ignoring)
    a mismatched VIRTUAL_ENV inherited from the PodTerm process."""
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHONHOME"}}
    env.update(extra)
    return env


# -- Device probe ------------------------------------------------------------


def _resolve_device() -> str:
    """'auto' → probe gpt-golf's torch once for CUDA; else use the configured device verbatim."""
    global _device_cache
    setting = diag_device()
    if setting != "auto":
        return setting
    if _device_cache is None:
        try:
            r = subprocess.run(
                [*diag_python_cmd(), "-c", "import torch; print(torch.cuda.is_available())"],
                cwd=str(gpt_golf_dir()), env=_gpt_golf_env(),
                capture_output=True, text=True, timeout=180,
            )
            _device_cache = "cuda:0" if r.stdout.strip().endswith("True") else "cpu"
        except Exception:
            _device_cache = "cpu"
        log.info("diagnostics device resolved to %s", _device_cache)
    return _device_cache


# -- Orchestration -----------------------------------------------------------


async def handle_snapshot(pod_id: str, payload: dict) -> None:
    """Entry point from the event pipeline; safe to fire-and-forget per snapshot event."""
    if not diagnostics_enabled():
        return
    _pending[pod_id] = payload  # newest wins
    lock = _locks.setdefault(pod_id, asyncio.Lock())
    if lock.locked():
        return  # an in-flight worker will pick up the latest pending snapshot
    async with lock:
        while pod_id in _pending:
            p = _pending.pop(pod_id)
            try:
                await asyncio.to_thread(_process, pod_id, p)
            except Exception:
                log.exception("snapshot diagnostics failed pod=%s step=%s", pod_id, p.get("step"))


def _process(pod_id: str, payload: dict) -> None:
    """Blocking: download → (ack final) → run diagnostics → persist → SSE. Runs in a worker thread."""
    step = int(payload.get("step", 0))
    run = db.get_run(pod_id) or {}
    cfg = {}
    if run.get("config_json"):
        try:
            cfg = json.loads(run["config_json"])
        except (ValueError, TypeError):
            cfg = {}
    token = cfg.get("eventd_token", "")
    variant = run.get("data_variant") or cfg.get("data_variant") or DEFAULT_DATA_VARIANT

    ckpt = _download(pod_id, token, payload)
    if payload.get("final"):
        # Release the pod the instant its last checkpoint is safely off-pod (downloaded + sha-verified
        # in _download). bootstrap.sh's teardown is polling for this ack and breaks out as soon as it
        # lands; diagnostics run off-pod, so the pod needn't be held alive for them.
        _ack_final(pod_id, token, step)

    diag, status = _run_diagnostics(ckpt, step, variant)
    _reconcile_bpb(pod_id, step, diag)

    db.add_diagnostics(pod_id, step, status, json.dumps(diag))
    hub.send(pod_id, "diagnostic", {
        "step": step, "status": status, "final": bool(payload.get("final")),
        "health": diag.get("health"),
        "sections": [s.get("name") for s in diag.get("sections", [])],
    })
    log.info("diagnostics done pod=%s step=%s status=%s", pod_id, step, status)


def _reconcile_bpb(pod_id: str, step: int, diag: dict) -> None:
    """Cross-check the off-pod BPB (computed in the subprocess, meta.checks.bpb) against the run's
    on-pod eval BPB — the only place both numbers coexist (the subprocess can't see podterm's DB).
    A mismatch means val-shard/tokenizer/seq-len drift, not a model bug. Writes the result back into
    diag['meta']['checks']['bpb_match'] so it's persisted + visible. Best-effort."""
    checks = diag.get("meta", {}).get("checks")
    if not isinstance(checks, dict) or not isinstance(checks.get("bpb"), (int, float)):
        return
    try:
        pod_bpb = db.get_eval_bpb_near(pod_id, step)
    except Exception:
        log.exception("bpb reconcile lookup failed pod=%s step=%s", pod_id, step)
        return
    off = float(checks["bpb"])
    if pod_bpb is None:
        checks["bpb_match"] = {"pod_bpb": None, "off_pod_bpb": off, "match": None}
        return
    rel = abs(off - pod_bpb) / max(abs(pod_bpb), 1e-9)
    checks["bpb_match"] = {"pod_bpb": pod_bpb, "off_pod_bpb": off,
                          "rel_err": rel, "match": rel <= _BPB_TOL}
    if rel > _BPB_TOL:
        log.warning("off-pod BPB %.4f disagrees with pod %.4f (rel %.1f%%) pod=%s step=%s",
                    off, pod_bpb, rel * 100, pod_id, step)


def _ack_final(pod_id: str, token: str, step: int) -> None:
    """Tell the pod its final snapshot is downloaded — 'download complete, you can stop now'.

    Writes the teardown-handshake sentinel via the daemon's /snapshot/ack endpoint, which lets
    bootstrap.sh's cleanup() stop waiting (up to 120s) and tear the pod down immediately."""
    status, _ = _request(f"{_base_url(pod_id)}/snapshot/ack?step={step}", token, timeout=15)
    if status == 200:
        log.info("final snapshot acked pod=%s step=%s — pod released", pod_id, step)
    else:
        log.warning("final snapshot ack failed pod=%s step=%s status=%s", pod_id, step, status)


def _download(pod_id: str, token: str, payload: dict) -> Path:
    step = int(payload.get("step", 0))
    dest_dir = _CACHE_DIR / pod_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"step{step}.pt"
    status, body = _request(f"{_base_url(pod_id)}/snapshot?step={step}", token, timeout=120)
    if status != 200 or not body:
        raise RuntimeError(f"snapshot download failed (status={status}, bytes={len(body)})")
    want = payload.get("sha256")
    if want and hashlib.sha256(body).hexdigest() != want:
        raise RuntimeError("snapshot sha256 mismatch — refusing to diagnose a corrupt checkpoint")
    dest.write_bytes(body)
    return dest


def _ensure_val_shard(variant: str) -> bool:
    """Make sure the val shard + tokenizer for `variant` are cached locally (in gpt-golf's data dir,
    where `train_gpt.load_validation_tokens()` looks), so the diagnostics token stages have data to
    read. Delegates to gpt-golf's own downloader with `--train-shards 0` — that materializes just the
    val split + tokenizer and no-ops on files already present, so the first snapshot pays the download
    and the rest are a cheap stat-only pass. Returns False on failure; the caller falls back to a
    weights-only run rather than letting the subprocess crash on a missing shard.

    Serialized across worker threads (concurrent pods may share a variant) and memoized per process.
    """
    with _val_shard_lock:
        if variant in _val_shard_ready:
            return True
        try:
            proc = subprocess.run(
                [*diag_python_cmd(), "data/cached_challenge_fineweb.py",
                 "--variant", variant, "--train-shards", "0"],
                cwd=str(gpt_golf_dir()), env=_gpt_golf_env(),
                capture_output=True, text=True, timeout=_VAL_SHARD_TIMEOUT,
            )
        except Exception:
            log.exception("val shard cache failed variant=%s", variant)
            return False
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            log.warning("val shard download failed variant=%s rc=%s: %s", variant, proc.returncode, tail)
            return False
        _val_shard_ready.add(variant)
        log.info("val shard cached locally variant=%s", variant)
        return True


def _run_diagnostics(ckpt: Path, step: int, variant: str) -> tuple[dict, str]:
    out = ckpt.with_suffix(".diag.json")
    device = _resolve_device()
    no_tokens = device == "cpu" or os.environ.get("DIAG_NO_TOKENS", "").strip().lower() in {"1", "true", "yes"}
    # Importing train_gpt (even for a weights-only run) loads the tokenizer at import time, and the
    # token stages additionally need the val shard. gpt-golf's downloader brings both together, so
    # ensure it for *every* run — not just GPU/token ones — or the subprocess can't even import.
    if diag_cache_val_shard() and not _ensure_val_shard(variant) and not no_tokens:
        # No val shard — degrade the token stages to weights-only. (If the tokenizer is missing too,
        # train_gpt won't import and the subprocess error is recorded below.)
        log.warning("falling back to weights-only diagnostics (no val shard) variant=%s step=%s", variant, step)
        no_tokens = True
    cmd = [*diag_python_cmd(), "-m", "podterm.diagnostics", str(ckpt), "--out", str(out), "--step", str(step)]
    if no_tokens:
        cmd.append("--no-tokens")
    env = _gpt_golf_env(
        PYTHONPATH=os.pathsep.join(filter(None, [str(_PODTERM_ROOT), os.environ.get("PYTHONPATH", "")])),
        DEVICE=device,
        DATA_VARIANT=variant,
        **diag_caps(),
    )
    proc = subprocess.run(
        cmd, cwd=str(gpt_golf_dir()), env=env,
        capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
    )
    try:
        diag = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        # No parseable output — record the failure (with a tail of stderr) rather than dropping it.
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return {"step": step, "error": f"{type(e).__name__}: {e}", "stderr": tail, "sections": []}, "error"
    # The subprocess writes a value-based verdict (runner.py → health.compute) into diag["status"].
    # Fall back to execution-only status for older docs that predate the health layer.
    return diag, diag.get("status") or _overall_status(diag)


def _overall_status(diag: dict) -> str:
    """Execution-only status (fallback): worst of the per-section stage states."""
    states = {s.get("status") for s in diag.get("sections", [])}
    if "error" in states:
        return "error"
    if "partial" in states:
        return "partial"
    return "ok"
