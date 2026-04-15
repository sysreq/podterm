"""Constants, defaults, and environment builder helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

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
