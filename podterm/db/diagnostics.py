"""Off-pod model-health diagnostics persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .connection import get_conn


def add_diagnostics(run_id: str, step: int, status: str, diag_json: str) -> None:
    """Store one snapshot's diagnostics result keyed by (run, step)."""
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO run_diagnostics (run_id, step, created_at, status, diag_json)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, step, datetime.now(timezone.utc).isoformat(), status, diag_json),
    )
    conn.commit()


def get_diagnostics(run_id: str) -> list[dict]:
    """All diagnostics snapshots for a run, oldest step first."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT step, created_at, status, diag_json FROM run_diagnostics WHERE run_id = ? ORDER BY step",
        (run_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["diag"] = json.loads(d.pop("diag_json")) if d.get("diag_json") else None
        except (ValueError, TypeError):
            d["diag"] = None
        out.append(d)
    return out
