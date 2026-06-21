from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

_PODTERM_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = Path.cwd() / ".cache" / "snapshots"
_SUBPROCESS_TIMEOUT = float(os.environ.get("DIAG_TIMEOUT", "900"))
_VAL_SHARD_TIMEOUT = float(os.environ.get("DIAG_VAL_SHARD_TIMEOUT", "600"))
_BPB_TOL = float(os.environ.get("DIAG_BPB_TOL", "0.02"))

_locks: dict[str, asyncio.Lock] = {}
_pending: dict[str, dict] = {}
_device_cache: str | None = None
_val_shard_lock = threading.Lock()
_val_shard_ready: set[str] = set()
