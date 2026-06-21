from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from podterm.config import (
    diag_cache_val_shard,
    diag_caps,
    diag_device,
    diag_python_cmd,
    gpt_golf_dir,
)

from . import state

log = logging.getLogger("podterm.snapshots")


def gpt_golf_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHONHOME"}}
    env.update(extra)
    return env


def resolve_device() -> str:
    setting = diag_device()
    if setting != "auto":
        return setting
    if state._device_cache is None:
        try:
            r = subprocess.run(
                [*diag_python_cmd(), "-c", "import torch; print(torch.cuda.is_available())"],
                cwd=str(gpt_golf_dir()), env=gpt_golf_env(),
                capture_output=True, text=True, timeout=180,
            )
            state._device_cache = "cuda:0" if r.stdout.strip().endswith("True") else "cpu"
        except Exception:
            state._device_cache = "cpu"
        log.info("diagnostics device resolved to %s", state._device_cache)
    return state._device_cache


def ensure_val_shard(variant: str) -> bool:
    with state._val_shard_lock:
        if variant in state._val_shard_ready:
            return True
        try:
            proc = subprocess.run(
                [*diag_python_cmd(), "data/cached_challenge_fineweb.py",
                 "--variant", variant, "--train-shards", "0"],
                cwd=str(gpt_golf_dir()), env=gpt_golf_env(),
                capture_output=True, text=True, timeout=state._VAL_SHARD_TIMEOUT,
            )
        except Exception:
            log.exception("val shard cache failed variant=%s", variant)
            return False
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            log.warning("val shard download failed variant=%s rc=%s: %s", variant, proc.returncode, tail)
            return False
        state._val_shard_ready.add(variant)
        log.info("val shard cached locally variant=%s", variant)
        return True


def run_diagnostics(ckpt: Path, step: int, variant: str) -> tuple[dict, str]:
    out = ckpt.with_suffix(".diag.json")
    device = resolve_device()
    no_tokens = device == "cpu" or os.environ.get("DIAG_NO_TOKENS", "").strip().lower() in {"1", "true", "yes"}
    if diag_cache_val_shard() and not ensure_val_shard(variant) and not no_tokens:
        log.warning("falling back to weights-only diagnostics (no val shard) variant=%s step=%s", variant, step)
        no_tokens = True
    cmd = [*diag_python_cmd(), "-m", "podterm.diagnostics", str(ckpt), "--out", str(out), "--step", str(step)]
    if no_tokens:
        cmd.append("--no-tokens")
    env = gpt_golf_env(
        PYTHONPATH=os.pathsep.join(filter(None, [str(state._PODTERM_ROOT), os.environ.get("PYTHONPATH", "")])),
        DEVICE=device,
        DATA_VARIANT=variant,
        **diag_caps(),
    )
    proc = subprocess.run(
        cmd, cwd=str(gpt_golf_dir()), env=env,
        capture_output=True, text=True, timeout=state._SUBPROCESS_TIMEOUT,
    )
    try:
        diag = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return {"step": step, "error": f"{type(e).__name__}: {e}", "stderr": tail, "sections": []}, "error"
    return diag, diag.get("status") or overall_status(diag)


def overall_status(diag: dict) -> str:
    states = {s.get("status") for s in diag.get("sections", [])}
    if "error" in states:
        return "error"
    if "partial" in states:
        return "partial"
    return "ok"
