"""Run history + comparison routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from podterm import db
from podterm.diagnostics.compare import diff_docs

router = APIRouter()


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


@router.get("/api/runs")
async def list_runs(
    branch: str | None = None,
    gpu: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    return db.list_runs(limit=limit, branch=branch, gpu=gpu)


@router.get("/api/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str):
    return db.get_metrics(run_id)


@router.get("/api/runs/{run_id}/diagnostics")
async def get_run_diagnostics(run_id: str):
    """Model-health diagnostics time series (one entry per snapshot step)."""
    return db.get_diagnostics(run_id)


@router.post("/api/compare")
async def compare_runs(body: CompareRequest):
    run_ids = body.run_ids
    if len(run_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 runs")
    metrics = db.get_metrics_multi(run_ids)
    runs = {rid: db.get_run(rid) for rid in run_ids}
    return {"metrics": metrics, "runs": runs, "diagnostics": _compare_diagnostics(run_ids)}


def _compare_diagnostics(run_ids: list[str]) -> dict:
    """Per-run latest health verdict + a structured diff of each run's latest snapshot against the
    first selected run (the baseline). Powers the Compare tab's Model Health Diff."""
    latest: dict[str, dict | None] = {}
    for rid in run_ids:
        hist = db.get_diagnostics(rid)
        latest[rid] = hist[-1]["diag"] if hist and hist[-1].get("diag") else None
    health = {rid: (doc.get("health") if doc else None) for rid, doc in latest.items()}
    base_id = run_ids[0]
    diff = {}
    if latest[base_id]:
        for rid in run_ids[1:]:
            if latest[rid]:
                diff[rid] = diff_docs(latest[base_id], latest[rid])
    return {"base": base_id, "health": health, "diff": diff}


@router.get("/api/runs/filters")
async def run_filters():
    return {
        "branches": db.get_distinct_branches(),
        "gpus": db.get_distinct_gpus(),
    }
