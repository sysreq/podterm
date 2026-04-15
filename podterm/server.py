"""FastAPI backend — REST API, SSE streaming, drain loop."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from podterm import db
from podterm.config import (
    DEFAULT_CLOUD_TYPE,
    DEFAULT_DATA_REPO_ID,
    DEFAULT_DATA_VARIANT,
    DEFAULT_DATA_VERSION,
    DEFAULT_IMAGE,
    POD_PREFIX,
    build_optional_debug_env,
)
from podterm.helpers import (
    build_variant_choices,
    fetch_manifest,
    get_git_branches,
    get_local_pubkey,
)
from podterm.parser import (
    CommitInfo,
    MemoryInfo,
    ModelInfo,
    PhaseMarker,
    RunSummary,
    StepMetric,
    parse_line,
)
from podterm.runpod import (
    api_create_pod,
    api_terminate_pod,
    create_or_update_template,
    get_available_gpus,
    get_datacenters,
    get_gpt_golf_pods,
    get_network_volume,
)
from podterm.ssh import LogQueue, SshTailThread

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

log_queue: LogQueue = queue.Queue()
ssh_threads: dict[str, SshTailThread] = {}
sse_subscribers: dict[str, list[asyncio.Queue]] = {}  # pod_id → client queues
last_launch_config: dict | None = None

# Per-run state accumulators
run_memory: dict[str, MemoryInfo] = {}
run_summary: dict[str, RunSummary] = {}
run_exit_code: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Drain loop — reads log_queue, parses, writes to DB, fans out SSE
# ---------------------------------------------------------------------------

async def drain_loop() -> None:
    """Asyncio background task: drain SSH log queue, parse, persist, fan-out to SSE."""
    metrics_buffer: dict[str, list[StepMetric]] = {}
    last_flush = time.monotonic()

    while True:
        drained = 0
        while drained < 500:
            try:
                pod_id, line = log_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1

            # Fan out raw log line to SSE subscribers
            _sse_send(pod_id, "log", {"line": line})

            # Parse
            event = parse_line(line)
            if event is None:
                continue

            if isinstance(event, StepMetric):
                metrics_buffer.setdefault(pod_id, []).append(event)
                _sse_send(pod_id, "metric", {
                    "step": event.step, "total_steps": event.total_steps,
                    "train_loss": event.train_loss, "step_avg_ms": event.step_avg_ms,
                    "val_loss": event.val_loss, "val_bpb": event.val_bpb,
                    "train_time_ms": event.train_time_ms,
                })
            elif isinstance(event, MemoryInfo):
                run_memory[pod_id] = event
                _sse_send(pod_id, "memory", asdict(event))
                db.update_run(pod_id, peak_memory_mib=event.peak_mib, reserved_memory_mib=event.reserved_mib)
            elif isinstance(event, RunSummary):
                run_summary[pod_id] = event
                _sse_send(pod_id, "summary", asdict(event))
            elif isinstance(event, ModelInfo):
                _sse_send(pod_id, "info", {"model_params": event.params})
                db.update_run(pod_id, model_params=event.params)
            elif isinstance(event, CommitInfo):
                _sse_send(pod_id, "info", {"commit_hash": event.hash, "commit_msg": event.message})
                db.update_run(pod_id, commit_hash=event.hash, commit_msg=event.message)
            elif isinstance(event, PhaseMarker):
                _sse_send(pod_id, "phase", {"phase": event.phase, "exit_code": event.exit_code})
                if event.exit_code is not None:
                    run_exit_code[pod_id] = event.exit_code
                if "Starting Training" in event.phase:
                    # Clear metrics buffer on restart
                    metrics_buffer.pop(pod_id, None)
                elif "Training finished" in event.phase:
                    _finalize_run(pod_id)

        # Batch-flush metrics to DB every 5 seconds
        now = time.monotonic()
        if now - last_flush > 5 and metrics_buffer:
            for pid, metrics in metrics_buffer.items():
                try:
                    db.add_metrics_batch(pid, metrics)
                except Exception:
                    pass
            metrics_buffer.clear()
            last_flush = now

        await asyncio.sleep(0.05)


def _sse_send(pod_id: str, event_type: str, data: dict) -> None:
    """Fan out an SSE event to all subscribers for a pod."""
    clients = sse_subscribers.get(pod_id)
    if not clients:
        return
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead: list[asyncio.Queue] = []
    for q in clients:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        clients.remove(q)


def _finalize_run(pod_id: str) -> None:
    try:
        summary = run_summary.pop(pod_id, None)
        memory = run_memory.pop(pod_id, None)
        exit_code = run_exit_code.pop(pod_id, None)
        db.finish_run(pod_id, summary=summary, memory=memory, exit_code=exit_code)
    except Exception:
        pass


def _connect_pod(pod_id: str, pod_name: str, cost_per_hr=None) -> None:
    """Start SSH streaming for a pod if not already connected."""
    if pod_id in ssh_threads:
        return
    try:
        db.create_run(pod_id, pod_name, {"gpu": "", "cost_per_hr": cost_per_hr})
    except Exception:
        pass
    thread = SshTailThread(pod_id, pod_name, log_queue)
    ssh_threads[pod_id] = thread
    thread.start()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start drain loop
    task = asyncio.create_task(drain_loop())

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
    task.cancel()
    for t in ssh_threads.values():
        t.stop()
    db.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PodTerm", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# Mount static after the explicit / route so it doesn't shadow it
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pod management
# ---------------------------------------------------------------------------

@app.get("/api/pods")
async def list_pods():
    def _fetch():
        try:
            pods = get_gpt_golf_pods()
        except Exception:
            pods = []
        # Auto-connect to running pods
        for p in pods:
            pid = p.get("id", "")
            if p.get("desiredStatus") == "RUNNING" and pid not in ssh_threads:
                _connect_pod(pid, p.get("name", pid), p.get("costPerHr"))
        return pods
    pods = await asyncio.to_thread(_fetch)
    return pods


class LaunchConfig(BaseModel):
    branch: str
    name: str | None = None
    datacenter: str
    gpu: str
    gpu_count: int = 1
    train_script: str = "train_gpt.py"
    profile_steps: int = 0
    compile_debug: bool = False
    graph_logs: bool = False
    time_budget: int = 600
    prep_shards: int = 10
    data_repo_id: str = DEFAULT_DATA_REPO_ID
    data_version: str = DEFAULT_DATA_VERSION
    data_variant: str = DEFAULT_DATA_VARIANT
    data_path: str = ""
    tokenizer_path: str = ""
    vocab_size: str = ""


@app.post("/api/pods/launch")
async def launch_pod(cfg: LaunchConfig):
    global last_launch_config
    cfg_dict = cfg.model_dump()
    last_launch_config = cfg_dict

    def _launch():
        suffix = cfg.name or cfg.branch
        pod_name = f"{POD_PREFIX}-{suffix}-{time.strftime('%m%d-%H%M%S')}"

        env = {
            "BRANCH": cfg.branch,
            "PREP_SHARDS": str(cfg.prep_shards),
            "TRAIN_SCRIPT": cfg.train_script,
            "NPROC": str(cfg.gpu_count),
            "NCCL_IB_DISABLE": "1",
            "HF_HUB_CACHE": "/workspace/.cache/huggingface",
            "DATA_REPO_ID": cfg.data_repo_id,
            "DATA_VERSION": cfg.data_version,
            "DATA_PATH": cfg.data_path or "./data/datasets/fineweb10B_sp1024",
            "DATA_VARIANT": cfg.data_variant,
            "TOKENIZER_PATH": cfg.tokenizer_path or "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": cfg.vocab_size or "1024",
            "MAX_WALLCLOCK_SECONDS": str(cfg.time_budget),
            "TRAIN_LOG_EVERY": "250",
            "VAL_LOSS_EVERY": "1000",
            "GITHUB_TOKEN": "{{ RUNPOD_SECRET_gh_gpt-golf_token }}",
            "HF_TOKEN": "{{ RUNPOD_SECRET_hf_gpt-golf_token }}",
        }
        if cfg.profile_steps > 0:
            env["GPT_GOLF_PROFILE"] = str(cfg.profile_steps)
        env.update(build_optional_debug_env(cfg_dict))

        pubkey = get_local_pubkey()
        if pubkey:
            env["PUBLIC_KEY"] = pubkey

        network_vol = get_network_volume(cfg.datacenter)
        tpl = create_or_update_template(DEFAULT_IMAGE, env)
        pod = api_create_pod(
            pod_name, cfg.gpu, tpl,
            DEFAULT_CLOUD_TYPE, cfg.datacenter, network_vol,
            gpu_count=cfg.gpu_count,
        )
        pod_id = pod["id"]
        cost = pod.get("costPerHr", 0)
        cfg_dict["cost_per_hr"] = cost

        db.create_run(pod_id, pod_name, cfg_dict)

        thread = SshTailThread(pod_id, pod_name, log_queue)
        ssh_threads[pod_id] = thread
        thread.start()

        return {"pod_id": pod_id, "name": pod_name, "cost_per_hr": cost}

    return await asyncio.to_thread(_launch)


@app.post("/api/pods/{pod_id}/stop")
async def stop_pod(pod_id: str):
    def _stop():
        api_terminate_pod(pod_id)
        thread = ssh_threads.pop(pod_id, None)
        if thread:
            thread.stop()
        _finalize_run(pod_id)
        return {"status": "terminated", "pod_id": pod_id}
    return await asyncio.to_thread(_stop)


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

@app.get("/api/stream/{pod_id}")
async def stream_pod(pod_id: str):
    client_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
    sse_subscribers.setdefault(pod_id, []).append(client_queue)

    async def event_generator():
        try:
            # Send initial keepalive
            yield ": connected\n\n"
            while True:
                msg = await client_queue.get()
                yield msg
        except asyncio.CancelledError:
            pass
        finally:
            clients = sse_subscribers.get(pod_id, [])
            if client_queue in clients:
                clients.remove(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Launch form data endpoints
# ---------------------------------------------------------------------------

@app.get("/api/datacenters")
async def datacenters():
    return await asyncio.to_thread(get_datacenters)


@app.get("/api/gpus/{datacenter_id}")
async def gpus(datacenter_id: str):
    result = await asyncio.to_thread(get_available_gpus, datacenter_id)
    return [{"label": g[0], "id": g[1]} for g in result]


@app.get("/api/branches")
async def branches():
    return await asyncio.to_thread(get_git_branches)


@app.get("/api/variants")
async def variants():
    def _fetch():
        manifest = fetch_manifest()
        opts, lookup = build_variant_choices(manifest)
        return {"options": [{"label": o[0], "id": o[1]} for o in opts], "lookup": lookup}
    return await asyncio.to_thread(_fetch)


@app.get("/api/last-config")
async def get_last_config():
    return last_launch_config or {}


# ---------------------------------------------------------------------------
# History + comparison
# ---------------------------------------------------------------------------

@app.get("/api/runs")
async def list_runs(branch: str | None = None, gpu: str | None = None, limit: int = 100):
    return db.list_runs(limit=limit, branch=branch, gpu=gpu)


@app.post("/api/compare")
async def compare_runs(body: dict):
    run_ids = body.get("run_ids", [])
    if len(run_ids) < 2:
        return {"error": "Select at least 2 runs"}
    metrics = db.get_metrics_multi(run_ids)
    runs = {rid: db.get_run(rid) for rid in run_ids}
    return {"metrics": metrics, "runs": runs}


@app.get("/api/runs/filters")
async def run_filters():
    return {
        "branches": db.get_distinct_branches(),
        "gpus": db.get_distinct_gpus(),
    }
