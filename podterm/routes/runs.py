"""Run history + comparison routes."""

from __future__ import annotations

from fastapi import APIRouter

from podterm import db

router = APIRouter()


@router.get("/api/runs")
async def list_runs(branch: str | None = None, gpu: str | None = None, limit: int = 100):
    return db.list_runs(limit=limit, branch=branch, gpu=gpu)


@router.get("/api/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str):
    return db.get_metrics(run_id)


@router.get("/api/runs/{run_id}/diagnostics")
async def get_run_diagnostics(run_id: str):
    """Model-health diagnostics time series (one entry per snapshot step)."""
    return db.get_diagnostics(run_id)


@router.post("/api/compare")
async def compare_runs(body: dict):
    run_ids = body.get("run_ids", [])
    if len(run_ids) < 2:
        return {"error": "Select at least 2 runs"}
    metrics = db.get_metrics_multi(run_ids)
    runs = {rid: db.get_run(rid) for rid in run_ids}
    return {"metrics": metrics, "runs": runs}


@router.get("/api/runs/filters")
async def run_filters():
    return {
        "branches": db.get_distinct_branches(),
        "gpus": db.get_distinct_gpus(),
    }
