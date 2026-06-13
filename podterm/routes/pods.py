"""Pod lifecycle routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from podterm.pods import LaunchConfig, manager
from podterm.runpod import detect_redis_server

router = APIRouter()


@router.get("/api/pods")
async def list_pods():
    return await asyncio.to_thread(manager.list_pods)


@router.get("/api/redis-server")
async def redis_server():
    addr = await asyncio.to_thread(detect_redis_server)
    return {"address": addr}


@router.post("/api/pods/launch")
async def launch_pod(cfg: LaunchConfig):
    return await asyncio.to_thread(manager.launch, cfg)


@router.post("/api/pods/{pod_id}/stop")
async def stop_pod(pod_id: str):
    return await asyncio.to_thread(manager.stop, pod_id)
