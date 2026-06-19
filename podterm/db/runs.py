"""Run row CRUD and run-level queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from podterm.models import MemoryInfo, RunSummary

from .connection import get_conn
from .schema import RUN_UPDATE_COLUMNS


def create_run(run_id: str, pod_name: str, config: dict | None = None) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO runs (run_id, pod_name, started_at, config_json,
           branch, gpu_type, gpu_count, datacenter, data_variant, vocab_size, cost_per_hr)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            pod_name,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(config) if config else None,
            (config or {}).get("branch"),
            (config or {}).get("gpu"),
            (config or {}).get("gpu_count", 1),
            (config or {}).get("datacenter"),
            (config or {}).get("data_variant"),
            int((config or {}).get("vocab_size", 0) or 0) or None,
            (config or {}).get("cost_per_hr"),
        ),
    )
    conn.commit()


def update_run(run_id: str, **fields: object) -> None:
    """Update known run-row fields."""
    if not fields:
        return
    unknown = set(fields) - RUN_UPDATE_COLUMNS
    if unknown:
        raise ValueError(f"unknown run field(s): {', '.join(sorted(unknown))}")
    conn = get_conn()
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [run_id]
    conn.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", vals)
    conn.commit()


def finish_run(
    run_id: str,
    summary: RunSummary | None = None,
    memory: MemoryInfo | None = None,
    exit_code: int | None = None,
) -> None:
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()

    row = conn.execute("SELECT started_at, cost_per_hr FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    duration = None
    total_cost = None
    if row and row["started_at"]:
        try:
            started = datetime.fromisoformat(row["started_at"])
            duration = int((datetime.now(timezone.utc) - started).total_seconds())
            if row["cost_per_hr"]:
                total_cost = round(row["cost_per_hr"] * duration / 3600, 4)
        except (ValueError, TypeError):
            pass

    step_row = conn.execute(
        "SELECT MAX(step) as max_step FROM metrics WHERE run_id = ?", (run_id,)
    ).fetchone()
    total_steps = step_row["max_step"] if step_row else None

    conn.execute(
        """UPDATE runs SET
            finished_at = ?, duration_seconds = ?, exit_code = ?, total_cost = ?, total_steps = ?,
            best_val_bpb = COALESCE(?, best_val_bpb),
            peak_memory_mib = COALESCE(?, peak_memory_mib),
            reserved_memory_mib = COALESCE(?, reserved_memory_mib)
           WHERE run_id = ?""",
        (
            now,
            duration,
            exit_code,
            total_cost,
            total_steps,
            summary.best_val_bpb if summary else None,
            memory.peak_mib if memory else None,
            memory.reserved_mib if memory else None,
            run_id,
        ),
    )
    conn.commit()


def get_run(run_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 100, branch: str | None = None, gpu: str | None = None) -> list[dict]:
    conn = get_conn()
    query = "SELECT * FROM runs WHERE 1=1"
    params: list[object] = []
    if branch:
        query += " AND branch = ?"
        params.append(branch)
    if gpu:
        query += " AND gpu_type = ?"
        params.append(gpu)
    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_distinct_branches() -> list[str]:
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT branch FROM runs WHERE branch IS NOT NULL ORDER BY branch").fetchall()
    return [r["branch"] for r in rows]


def get_distinct_gpus() -> list[str]:
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT gpu_type FROM runs WHERE gpu_type IS NOT NULL ORDER BY gpu_type").fetchall()
    return [r["gpu_type"] for r in rows]
