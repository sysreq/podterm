from __future__ import annotations

import asyncio
import json
import logging

from podterm import db
from podterm.config import DEFAULT_DATA_VARIANT, diagnostics_enabled

from . import state
from .client import base_url, download_snapshot, request
from .runner import run_diagnostics

log = logging.getLogger("podterm.snapshots")


async def handle_snapshot(pod_id: str, payload: dict) -> list[dict]:
    """Run diagnostics for the pending snapshot(s) on this pod.

    Per-pod processing is serialized + coalesced to the newest pending snapshot.
    Returns one `diagnostic` SSE event payload per snapshot processed (empty when
    nothing was processed / diagnostics are disabled / coalesced into another
    holder of the lock). The caller sends them on the event loop — the worker
    thread must not touch asyncio queues.
    """
    if not diagnostics_enabled():
        return []
    state._pending[pod_id] = payload
    lock = state._locks.setdefault(pod_id, asyncio.Lock())
    if lock.locked():
        return []
    events: list[dict] = []
    async with lock:
        while pod_id in state._pending:
            p = state._pending.pop(pod_id)
            try:
                events.append(await asyncio.to_thread(process_snapshot, pod_id, p))
            except Exception:
                log.exception("snapshot diagnostics failed pod=%s step=%s", pod_id, p.get("step"))
    return events


def process_snapshot(pod_id: str, payload: dict) -> dict:
    """Download + run diagnostics + persist; return the `diagnostic` event dict.

    Runs in a worker thread (via asyncio.to_thread): it must NOT touch the
    asyncio SSE queues. The event payload is returned to the event loop, which
    does the fan-out.
    """
    step = int(payload.get("step", 0))
    run = db.get_run(pod_id) or {}
    cfg = parse_config(run.get("config_json"))
    token = cfg.get("eventd_token", "")
    variant = run.get("data_variant") or cfg.get("data_variant") or DEFAULT_DATA_VARIANT

    ckpt = download_snapshot(pod_id, token, payload)
    if payload.get("final"):
        ack_final(pod_id, token, step)

    diag, status = run_diagnostics(ckpt, step, variant)
    reconcile_bpb(pod_id, step, diag)

    db.add_diagnostics(pod_id, step, status, json.dumps(diag))
    log.info("diagnostics done pod=%s step=%s status=%s", pod_id, step, status)
    return {
        "step": step, "status": status, "final": bool(payload.get("final")),
        "health": diag.get("health"),
        "sections": [s.get("name") for s in diag.get("sections", [])],
    }


def parse_config(config_json: str | None) -> dict:
    if not config_json:
        return {}
    try:
        return json.loads(config_json)
    except (ValueError, TypeError):
        return {}


def reconcile_bpb(pod_id: str, step: int, diag: dict) -> None:
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
                              "rel_err": rel, "match": rel <= state._BPB_TOL}
    if rel > state._BPB_TOL:
        log.warning("off-pod BPB %.4f disagrees with pod %.4f (rel %.1f%%) pod=%s step=%s",
                    off, pod_bpb, rel * 100, pod_id, step)


def ack_final(pod_id: str, token: str, step: int) -> None:
    status, _ = request(f"{base_url(pod_id)}/snapshot/ack?step={step}", token, timeout=15)
    if status == 200:
        log.info("final snapshot acked pod=%s step=%s — pod released", pod_id, step)
    else:
        log.warning("final snapshot ack failed pod=%s step=%s status=%s", pod_id, step, status)
