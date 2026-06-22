"""FastAPI app — wires the lifespan, static files, and route modules together.

All logic lives in the service modules: sse.py (fan-out), pipeline.py (event
drain + telemetry), pods.py (pod lifecycle), runpod/ (RunPod API), db.py
(persistence). This module is glue only.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from podterm import db
from podterm.metadata_cache import metadata_cache
from podterm.pipeline import pipeline
from podterm.pods import manager
from podterm.routes import routers

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start drain loop + telemetry poller
    drain_task = asyncio.create_task(pipeline.drain_loop())
    telemetry_task = asyncio.create_task(pipeline.telemetry_loop())
    try:
        await asyncio.wait_for(metadata_cache.refresh_once(), timeout=20)
    except TimeoutError:
        # Don't block app startup forever if RunPod/HF metadata is slow; routes still have fallbacks.
        pass
    metadata_task = asyncio.create_task(metadata_cache.refresh_loop(initial_delay=True))

    # Auto-open browser after a short delay
    async def _open_browser():
        await asyncio.sleep(1.0)
        try:
            subprocess.Popen(
                ["explorer.exe", "http://127.0.0.1:8000"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            try:
                import webbrowser
                webbrowser.open("http://127.0.0.1:8000")
            except Exception:
                pass
    asyncio.create_task(_open_browser())

    yield

    # Ordered shutdown: stop producers → drain/await consumers → close db.
    # 1. Stop poller threads first (signals + joins them) so no new events land.
    manager.stop_all()
    # 2. Cancel the recurring tasks, then await each so their `finally` blocks
    #    run (the drain loop flushes remaining metrics + awaits snapshot jobs).
    for task in (drain_task, telemetry_task, metadata_task):
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(
            drain_task, telemetry_task, metadata_task, return_exceptions=True
        )
    # 3. Close the database last, after every consumer has finished.
    db.close()


app = FastAPI(title="PodTerm", lifespan=lifespan)

# Restrictive CSP that works whether Plotly is CDN-loaded or self-hosted under
# /static/vendor/ ('self' covers the vendored copy).
_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.plot.ly; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)
_SECURITY_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Header-only middleware: leaves the (possibly streaming) body untouched so
    # SSE responses on /api/stream/... keep streaming to EventSource clients.
    response = await call_next(request)
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


for router in routers:
    app.include_router(router)

# Mount static after the explicit / route so it doesn't shadow it
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
