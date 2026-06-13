"""SSE streaming route."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from podterm.sse import hub

router = APIRouter()


@router.get("/api/stream/{pod_id}")
async def stream_pod(pod_id: str):
    client_queue = hub.subscribe(pod_id)
    return StreamingResponse(
        hub.stream(pod_id, client_queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
