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

# Least-privilege allowlist for the diagnostics subprocess. We copy ONLY what the launcher
# (uv run / python) and the suite genuinely need — never the RunPod API key, the Clerk console
# cookie, or GitHub tokens. Anything the caller wants set explicitly comes through **extra.
#   - launcher/runtime: PATH, HOME, locale, temp dir, CUDA libs/device selection
#   - uv/XDG caches so `uv run --no-sync` finds its environment without re-resolving
#   - diagnostics/model knobs read by config_gpt._with_env() / config_gpt.settings() if set
#     (DEVICE, DATA_VARIANT, DATA_PATH, TOKENIZER_PATH, VOCAB, SEED, DIAG_*, GPT_GOLF_DIR, ...)
#   - HF_*/HUGGINGFACE_* + DATA_*/MATCHED_FINEWEB_*: ensure_val_shard() shells out to gpt-golf's
#     HF downloader, which honours the HF cache/token and a custom dataset repo/revision. These are
#     dataset-access knobs the subprocess legitimately needs (same trust boundary as gpt-golf), not
#     the platform secrets finding 15 calls out to withhold.
# VIRTUAL_ENV is deliberately omitted so podterm's venv doesn't shadow gpt-golf's uv environment.
_ENV_ALLOW_NAMES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ", "TERM",
    "LANG", "LC_ALL", "LC_CTYPE",
    "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "NVIDIA_VISIBLE_DEVICES",
    "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONUTF8",
    "DEVICE", "DATA_VARIANT", "DATA_PATH", "TOKENIZER_PATH", "VOCAB", "SEED",
    "DATA_REPO_ID", "DATA_VERSION", "DATA_REMOTE_ROOT_PREFIX",
    "SNAPSHOT_DIR", "DATA_READY_SENTINEL",
})
# PYTHONHOME / VIRTUAL_ENV are intentionally NOT allowed (would shadow gpt-golf's stdlib/venv).
_ENV_ALLOW_PREFIXES = ("DIAG_", "GPT_GOLF", "UV_", "XDG_", "HF_", "HUGGINGFACE", "MATCHED_FINEWEB_")


def _env_allowed(name: str) -> bool:
    return name in _ENV_ALLOW_NAMES or name.startswith(_ENV_ALLOW_PREFIXES)


def gpt_golf_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if _env_allowed(k)}
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
    # Tie this request to the doc the subprocess writes, so a leftover .diag.json from a previous
    # (possibly failed) run can never be mistaken for this run's fresh result.
    request_id = f"{step}:{ckpt.name}"
    # Remove any stale output up front; the subprocess writes atomically (temp + os.replace), and we
    # only read back when it exits 0. Either way, a prior file must not survive into the read.
    try:
        out.unlink(missing_ok=True)
    except OSError as e:
        log.warning("could not remove stale diag output %s: %s", out, e)

    device = resolve_device()
    no_tokens = device == "cpu" or os.environ.get("DIAG_NO_TOKENS", "").strip().lower() in {"1", "true", "yes"}
    if diag_cache_val_shard() and not ensure_val_shard(variant) and not no_tokens:
        log.warning("falling back to weights-only diagnostics (no val shard) variant=%s step=%s", variant, step)
        no_tokens = True
    cmd = [*diag_python_cmd(), "-m", "podterm.diagnostics", str(ckpt),
           "--out", str(out), "--step", str(step), "--request-id", request_id]
    if no_tokens:
        cmd.append("--no-tokens")
    env = gpt_golf_env(
        PYTHONPATH=os.pathsep.join(filter(None, [str(state._PODTERM_ROOT), os.environ.get("PYTHONPATH", "")])),
        DEVICE=device,
        DATA_VARIANT=variant,
        **diag_caps(),
    )
    try:
        proc = subprocess.run(
            cmd, cwd=str(gpt_golf_dir()), env=env,
            capture_output=True, text=True, timeout=state._SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        tail = ((e.stderr or e.stdout or "") or "")[-2000:]
        log.warning("diagnostics subprocess timed out step=%s: %s", step, tail[-200:])
        return {"step": step, "error": f"timeout after {state._SUBPROCESS_TIMEOUT}s",
                "stderr": tail, "sections": []}, "error"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        log.warning("diagnostics subprocess failed step=%s rc=%s: %s", step, proc.returncode, tail[-200:])
        return {"step": step, "error": f"subprocess exit {proc.returncode}",
                "stderr": tail, "sections": []}, "error"

    try:
        diag = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return {"step": step, "error": f"{type(e).__name__}: {e}", "stderr": tail, "sections": []}, "error"

    # Reject a doc that doesn't carry our request id — guards against a leftover/foreign file the
    # subprocess somehow didn't overwrite (e.g. wrong --out, partial write outside our control).
    # A non-dict JSON value (list/scalar) is foreign by definition → treat as a mismatch, not a crash.
    meta = diag.get("meta") if isinstance(diag, dict) else None
    got = meta.get("request_id") if isinstance(meta, dict) else None
    if got != request_id:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        log.warning("diagnostics doc request_id mismatch step=%s got=%r want=%r", step, got, request_id)
        return {"step": step, "error": f"request_id mismatch (got {got!r}, want {request_id!r})",
                "stderr": tail, "sections": []}, "error"

    return diag, diag.get("status") or overall_status(diag)


def overall_status(diag: dict) -> str:
    states = {s.get("status") for s in diag.get("sections", [])}
    if "error" in states:
        return "error"
    if "partial" in states:
        return "partial"
    return "ok"
