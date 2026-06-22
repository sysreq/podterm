"""Run history + comparison routes."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from podterm import db
from podterm.diagnostics.compare import diff_docs

router = APIRouter()

# Keys whose values are credentials and must never leave the server. The
# per-launch ``eventd_token`` (the bearer for the pod daemon proxy) is the
# concrete leak we guard against; the pattern is broad so future secrets are
# masked by default rather than by opt-in.
#
# We match a secret word only when it is a whole ``_``/``-``-delimited segment
# of the key (or at the start/end) so plain-config fields the UI reads —
# ``batch_tokens``, ``tokenizer_path`` — are NOT masked while ``eventd_token`` /
# ``api_key`` are. The delimiter requirement also stops a literal secret word
# like ``key`` from matching inside ``keyword`` or ``tokenizer``.
_SECRET_WORDS = r"token|secret|password|passwd|cookie|api[_-]?key|apikey|key|bearer|credential"
_SECRET_KEY_RE = re.compile(
    rf"(?:^|[_-])(?:{_SECRET_WORDS})(?:[_-]|$)",
    re.IGNORECASE,
)
_REDACTED = "••• redacted •••"


def _is_secret_key(key: object) -> bool:
    return isinstance(key, str) and _SECRET_KEY_RE.search(key) is not None


def _redact(value: Any) -> Any:
    """Return a copy of ``value`` with any secret-keyed entries masked.

    Recurses into nested dicts and lists; scalar values are returned unchanged.
    Pure and stateless — the input is never mutated.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _is_secret_key(k) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _sanitize_run(row: dict | None) -> dict | None:
    """Return a copy of a run row with secrets stripped from ``config_json``.

    ``config_json`` is a JSON string holding the launch config, which includes
    the ``eventd_token`` (the pod-daemon bearer). We parse it, redact
    secret-keyed values recursively, and re-serialize so the response shape
    stays compatible with the frontend (which reads ``config_json``). Any
    secret-keyed top-level columns are masked too as defense in depth.
    """
    if row is None:
        return None
    sanitized = _redact(dict(row))
    raw = sanitized.get("config_json")
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            # Unparseable config_json: drop it rather than risk leaking a token
            # embedded in a malformed string.
            sanitized["config_json"] = None
        else:
            sanitized["config_json"] = json.dumps(_redact(parsed))
    return sanitized


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


@router.get("/api/runs")
async def list_runs(
    branch: str | None = None,
    gpu: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    return [_sanitize_run(row) for row in db.list_runs(limit=limit, branch=branch, gpu=gpu)]


@router.get("/api/runs/filters")
async def run_filters():
    return {
        "branches": db.get_distinct_branches(),
        "gpus": db.get_distinct_gpus(),
    }


@router.get("/api/runs/{run_id}")
async def get_run_row(run_id: str):
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _sanitize_run(row)


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
    runs = {rid: _sanitize_run(db.get_run(rid)) for rid in run_ids}
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
