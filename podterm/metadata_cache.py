"""Cached launch-form metadata.

The launch dialog needs RunPod datacenters/GPU availability and the HF dataset
manifest. Those are slow enough to make the first click feel sticky, but they
change rarely, so keep a warmed in-process snapshot and refresh periodically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from time import monotonic

from podterm.helpers import build_variant_choices, fetch_manifest
from podterm.runpod import get_datacenters

log = logging.getLogger("podterm.metadata")


def _refresh_interval() -> float:
    return float(os.environ.get("PODTERM_METADATA_REFRESH_SECONDS", "120"))


@dataclass
class MetadataSnapshot:
    datacenters: list[dict] = field(default_factory=list)
    variants: dict = field(default_factory=lambda: {"options": [], "lookup": {}})
    refreshed_at: float = 0.0


class MetadataCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = MetadataSnapshot()

    def refresh(self) -> None:
        datacenters = get_datacenters()
        manifest = fetch_manifest()
        opts, lookup = build_variant_choices(manifest)
        variants = {"options": [{"label": o[0], "id": o[1]} for o in opts], "lookup": lookup}
        with self._lock:
            self._snapshot = MetadataSnapshot(
                datacenters=datacenters,
                variants=variants,
                refreshed_at=monotonic(),
            )

    async def refresh_once(self) -> None:
        try:
            await asyncio.to_thread(self.refresh)
        except Exception:
            log.exception("metadata cache refresh failed")

    async def refresh_loop(self, initial_delay: bool = False) -> None:
        if initial_delay:
            await asyncio.sleep(_refresh_interval())
        while True:
            await self.refresh_once()
            await asyncio.sleep(_refresh_interval())

    def datacenters(self) -> list[dict]:
        with self._lock:
            return list(self._snapshot.datacenters)

    def variants(self) -> dict:
        with self._lock:
            return {
                "options": list(self._snapshot.variants.get("options", [])),
                "lookup": dict(self._snapshot.variants.get("lookup", {})),
            }

    def gpus(self, datacenter_id: str) -> list[dict]:
        with self._lock:
            datacenters = list(self._snapshot.datacenters)
        for dc in datacenters:
            if dc.get("id") == datacenter_id:
                return [
                    {"label": f"{g['displayName']} ({g['stockStatus']})", "id": g["gpuId"]}
                    for g in (dc.get("gpuAvailability") or [])
                ]
        return []

    def ready(self) -> bool:
        with self._lock:
            return bool(self._snapshot.datacenters and self._snapshot.variants.get("options"))


metadata_cache = MetadataCache()
