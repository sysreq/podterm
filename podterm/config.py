"""Constants, defaults, and environment builder helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


_FALSY = {"0", "false", "no", "off", ""}


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
DEFAULT_IMAGE = "ghcr.io/sysreq/gpt-golf-train:latest"
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

TEMPLATE_NAME = "gpt-golf-train"

# Port the on-pod event daemon (gpt-golf scripts/pod_eventd.py) listens on,
# exposed as /http in the template → https://{pod_id}-{port}.proxy.runpod.net
EVENTD_PORT = 8765

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
