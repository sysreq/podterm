"""SQLite storage layer for run history and metrics."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from podterm.parser import MemoryInfo, RunSummary, StepMetric

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    pod_name            TEXT NOT NULL,
    branch              TEXT,
    commit_hash         TEXT,
    commit_msg          TEXT,
    gpu_type            TEXT,
    gpu_count           INTEGER DEFAULT 1,
    datacenter          TEXT,
    data_variant        TEXT,
    vocab_size          INTEGER,
    model_params        INTEGER,
    cost_per_hr         REAL,
    total_cost          REAL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    duration_seconds    INTEGER,
    exit_code           INTEGER,
    best_val_bpb        REAL,
    peak_memory_mib     INTEGER,
    reserved_memory_mib INTEGER,
    total_steps         INTEGER,
    config_json         TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    total_steps   INTEGER,
    train_loss    REAL,
    val_loss      REAL,
    val_bpb       REAL,
    train_time_ms INTEGER,
    step_avg_ms   REAL,
    PRIMARY KEY (run_id, step)
);

CREATE INDEX IF NOT EXISTS idx_runs_branch  ON runs(branch);
CREATE INDEX IF NOT EXISTS idx_runs_gpu     ON runs(gpu_type);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
"""

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

_DB_DIR = os.path.expanduser("~/.local/share/podterm")
_DB_PATH = os.path.join(_DB_DIR, "runs.db")

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(_DB_DIR, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript(_SCHEMA)
    return _conn


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


def create_run(run_id: str, pod_name: str, config: dict | None = None) -> None:
    conn = _get_conn()
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
    """Update arbitrary fields on a run row."""
    conn = _get_conn()
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
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()

    # Calculate duration
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

    # Count total steps
    step_row = conn.execute(
        "SELECT MAX(step) as max_step FROM metrics WHERE run_id = ?", (run_id,)
    ).fetchone()
    total_steps = step_row["max_step"] if step_row else None

    conn.execute(
        """UPDATE runs SET
            finished_at = ?, duration_seconds = ?, exit_code = ?, total_cost = ?, total_steps = ?,
            best_val_bpb = ?, peak_memory_mib = ?, reserved_memory_mib = ?
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def add_metric(run_id: str, m: StepMetric) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO metrics
           (run_id, step, total_steps, train_loss, val_loss, val_bpb, train_time_ms, step_avg_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, m.step, m.total_steps, m.train_loss or None, m.val_loss, m.val_bpb, m.train_time_ms, m.step_avg_ms),
    )
    conn.commit()


def add_metrics_batch(run_id: str, metrics: list[StepMetric]) -> None:
    if not metrics:
        return
    conn = _get_conn()
    conn.executemany(
        """INSERT OR REPLACE INTO metrics
           (run_id, step, total_steps, train_loss, val_loss, val_bpb, train_time_ms, step_avg_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (run_id, m.step, m.total_steps, m.train_loss or None, m.val_loss, m.val_bpb, m.train_time_ms, m.step_avg_ms)
            for m in metrics
        ],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def get_run(run_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 100, branch: str | None = None, gpu: str | None = None) -> list[dict]:
    conn = _get_conn()
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


def get_metrics(run_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM metrics WHERE run_id = ? ORDER BY step", (run_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_metrics_multi(run_ids: list[str]) -> dict[str, list[dict]]:
    """Fetch metrics for multiple runs (for comparison overlay)."""
    result: dict[str, list[dict]] = {}
    for rid in run_ids:
        result[rid] = get_metrics(rid)
    return result


def get_distinct_branches() -> list[str]:
    conn = _get_conn()
    rows = conn.execute("SELECT DISTINCT branch FROM runs WHERE branch IS NOT NULL ORDER BY branch").fetchall()
    return [r["branch"] for r in rows]


def get_distinct_gpus() -> list[str]:
    conn = _get_conn()
    rows = conn.execute("SELECT DISTINCT gpu_type FROM runs WHERE gpu_type IS NOT NULL ORDER BY gpu_type").fetchall()
    return [r["gpu_type"] for r in rows]
