"""Constants, defaults, and environment builder helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path


log = logging.getLogger("podterm.config")

_FALSY = {"0", "false", "no", "off", ""}


def env_int(
    name: str,
    default: int,
    environ: Mapping[str, str] | None = None,
    *,
    minimum: int | None = None,
) -> int:
    """Parse an int env var, falling back to ``default`` on any bad/out-of-range value.

    Never raises: a non-numeric value, or one below ``minimum`` (when given), logs a
    warning and yields ``default`` so a stray env value can't crash import or starve a
    loop. Pass ``minimum=1`` for intervals/delays that must stay positive (a value of 0
    would otherwise create a busy zero-delay loop).
    """
    env = environ or os.environ
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        log.warning("invalid int for %s=%r; using default %r", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        log.warning("%s=%r below minimum %d; using default %r", name, value, minimum, default)
        return default
    return value


def env_float(
    name: str,
    default: float,
    environ: Mapping[str, str] | None = None,
    *,
    minimum: float | None = None,
) -> float:
    """Parse a float env var, falling back to ``default`` on any bad/out-of-range value.

    Never raises (see :func:`env_int`). Pass ``minimum`` (e.g. a small positive epsilon)
    for refresh/poll intervals so a 0 or negative value can't create a zero-delay loop.
    """
    env = environ or os.environ
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        log.warning("invalid float for %s=%r; using default %r", name, raw, default)
        return default
    if value != value:  # NaN
        log.warning("%s=%r is NaN; using default %r", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        log.warning("%s=%r below minimum %r; using default %r", name, value, minimum, default)
        return default
    return value


def boot_progress_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the boot/image-pull progress path runs (hapi polling + pull events).

    On by default. Set PODTERM_BOOT_PROGRESS to 0/false/no/off to fully disable
    it — a demo kill-switch for when the console cookie has gone stale and the
    boot panel would otherwise sit empty.
    """
    env = environ or os.environ
    return env.get("PODTERM_BOOT_PROGRESS", "1").strip().lower() not in _FALSY


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from a local .env into os.environ (no overwrite).

    Deliberately tiny — no python-dotenv dependency. Used for local secrets
    like RUNPOD_CONSOLE_CLIENT_COOKIE that must never be committed (.env is
    gitignored). Existing env vars win, so the shell can still override.
    """
    env_path = Path(path) if path else Path.cwd() / ".env"
    try:
        text = env_path.read_text()
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

POD_PREFIX = "gg"
DEFAULT_GPU = "RTX 6000 Ada"
DEFAULT_IMAGE = "ghcr.io/sysreq/gpt-caddy-single-train:latest"
DEFAULT_CONTAINER_DISK_GB = 50
DEFAULT_VOLUME_DISK_GB = 50
DEFAULT_CLOUD_TYPE = "SECURE"
DEFAULT_DATACENTER = "US-WA-1"
DEFAULT_BRANCH = "main"
DEFAULT_TIME_BUDGET = 600
DEFAULT_PREP_SHARDS = 10
DEFAULT_EVAL_TOKENS = 20971520
DEFAULT_DATA_REPO_ID = "sysrekt/parameter-golf"
DEFAULT_DATA_VERSION = "main"
DEFAULT_DATA_VARIANT = "sp1024"
DEFAULT_TORCH_COMPILE_DEBUG_DIR = "/workspace"
DEFAULT_TORCH_LOGS = "graph_breaks,graph_code"

BASELINE_GPU = "NVIDIA H100 80GB HBM3"
BASELINE_GPU_COUNT = 8

TEMPLATE_NAME = "gpt-caddy-single-train"

# Port the on-pod event daemon (gpt-golf scripts/pod_eventd.py) listens on,
# exposed as /http in the template → https://{pod_id}-{port}.proxy.runpod.net
EVENTD_PORT = 8765

# ---------------------------------------------------------------------------
# Off-pod diagnostics (model-health snapshots)
# ---------------------------------------------------------------------------

# Default per-run snapshot cadence injected into the pod env (config_gpt.TRAIN.snapshot_every).
DEFAULT_SNAPSHOT_EVERY = 1000


def diagnostics_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether GPT Caddy downloads snapshots and runs diagnostics. On unless DIAG_ENABLED is falsy."""
    env = environ or os.environ
    return env.get("DIAG_ENABLED", "1").strip().lower() not in _FALSY


def gpt_golf_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The sibling gpt-golf checkout — diagnostics import its model (train_gpt/config_gpt)."""
    env = environ or os.environ
    return Path(env.get("GPT_GOLF_DIR", Path.cwd().parent / "gpt-golf")).expanduser()


def diag_python_cmd(environ: Mapping[str, str] | None = None) -> list[str]:
    """Launcher prefix for the diagnostics subprocess, run with cwd=gpt_golf_dir so it picks up
    gpt-golf's torch env. Override via DIAG_PYTHON (e.g. an explicit interpreter path)."""
    import shlex
    env = environ or os.environ
    return shlex.split(env.get("DIAG_PYTHON", "uv run --no-sync python"))


def diag_device(environ: Mapping[str, str] | None = None) -> str:
    """DEVICE for the diagnostics run: 'auto' (probe), 'cpu', or an explicit cuda device."""
    env = environ or os.environ
    return env.get("DIAG_DEVICE", "auto").strip() or "auto"


def diag_cache_val_shard(environ: Mapping[str, str] | None = None) -> bool:
    """Whether token-requiring diagnostics first ensure the val shard is cached locally (in gpt-golf's
    data dir) by invoking gpt-golf's downloader. On unless DIAG_CACHE_VAL_SHARD is falsy — turn it off
    if the val shard is provisioned out-of-band and you don't want GPT Caddy reaching out to HuggingFace."""
    env = environ or os.environ
    return env.get("DIAG_CACHE_VAL_SHARD", "1").strip().lower() not in _FALSY


def diag_caps(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """DIAG_* knobs that bound the forward sweep cost (read by the diagnostics config loader).

    Defaults are deliberately small (a couple of batches, sampling off) so a per-interval run is
    cheap; override any of them in the env for a fuller sweep.
    """
    env = environ or os.environ
    # Forwarded as env strings to the diagnostics subprocess, but validated here so a bad
    # value (non-numeric / negative batch count) falls back to the documented default
    # instead of blowing up the subprocess's own parsing. minimum=0 keeps counts >= 0
    # (0 = off for sampling); batch counts are also allowed to be 0.
    return {
        "DIAG_MAX_BATCHES": str(env_int("DIAG_MAX_BATCHES", 2, env, minimum=0)),
        "DIAG_ENTROPY_BATCHES": str(env_int("DIAG_ENTROPY_BATCHES", 1, env, minimum=0)),
        "DIAG_SAMPLE_TOKENS": str(env_int("DIAG_SAMPLE_TOKENS", 0, env, minimum=0)),
    }

# ---------------------------------------------------------------------------
# Debug environment helpers
# ---------------------------------------------------------------------------


def default_compile_debug_enabled(
    last: dict | None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    if last is not None and "compile_debug" in last:
        return bool(last["compile_debug"])
    env = environ or os.environ
    return env.get("TORCH_COMPILE_DEBUG") == "1" or env.get("INDUCTOR_POST_FUSION_SVG") == "1"


def default_graph_logs_enabled(
    last: dict | None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    if last is not None and "graph_logs" in last:
        return bool(last["graph_logs"])
    env = environ or os.environ
    return bool(env.get("TORCH_LOGS"))


def build_optional_debug_env(
    cfg: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    source_env = environ or os.environ

    compile_debug = cfg.get("compile_debug")
    if compile_debug is None:
        for key in ("TORCH_COMPILE_DEBUG", "TORCH_COMPILE_DEBUG_DIR", "INDUCTOR_POST_FUSION_SVG"):
            if source_env.get(key):
                env[key] = source_env[key]
    elif compile_debug:
        env["TORCH_COMPILE_DEBUG"] = "1"
        env["TORCH_COMPILE_DEBUG_DIR"] = str(
            cfg.get("compile_debug_dir", DEFAULT_TORCH_COMPILE_DEBUG_DIR),
        )
        env["INDUCTOR_POST_FUSION_SVG"] = "1"

    graph_logs = cfg.get("graph_logs")
    if graph_logs is None:
        if source_env.get("TORCH_LOGS"):
            env["TORCH_LOGS"] = source_env["TORCH_LOGS"]
    elif graph_logs:
        env["TORCH_LOGS"] = str(cfg.get("torch_logs", DEFAULT_TORCH_LOGS))

    return env
