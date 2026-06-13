"""Launch-form metadata routes (datacenters, GPUs, branches, variants, last config)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from podterm.helpers import build_variant_choices, fetch_manifest, get_git_branches
from podterm.pods import manager
from podterm.runpod import get_available_gpus, get_datacenters

router = APIRouter()


@router.get("/api/datacenters")
async def datacenters():
    return await asyncio.to_thread(get_datacenters)


@router.get("/api/gpus/{datacenter_id}")
async def gpus(datacenter_id: str):
    result = await asyncio.to_thread(get_available_gpus, datacenter_id)
    return [{"label": g[0], "id": g[1]} for g in result]


@router.get("/api/branches")
async def branches():
    return await asyncio.to_thread(get_git_branches)


@router.get("/api/variants")
async def variants():
    def _fetch():
        manifest = fetch_manifest()
        opts, lookup = build_variant_choices(manifest)
        return {"options": [{"label": o[0], "id": o[1]} for o in opts], "lookup": lookup}
    return await asyncio.to_thread(_fetch)


@router.get("/api/last-config")
async def get_last_config():
    return manager.last_config or {}
