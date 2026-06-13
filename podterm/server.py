"""FastAPI app — wires the lifespan, static files, and route modules together.

All logic lives in the service modules: sse.py (fan-out), pipeline.py (event
drain + telemetry), pods.py (pod lifecycle), runpod/ (RunPod API), db.py
(persistence). This module is glue only.
"""

from __future__ import annotations

import asyncio
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from podterm import db
from podterm.pipeline import pipeline
from podterm.pods import manager
from podterm.routes import routers

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start drain loop + telemetry poller
    drain_task = asyncio.create_task(pipeline.drain_loop())
    telemetry_task = asyncio.create_task(pipeline.telemetry_loop())

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

    # Cleanup
    drain_task.cancel()
    telemetry_task.cancel()
    manager.stop_all()
    db.close()


app = FastAPI(title="PodTerm", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


for router in routers:
    app.include_router(router)

# Mount static after the explicit / route so it doesn't shadow it
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
