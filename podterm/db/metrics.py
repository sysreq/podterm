"""Step metric persistence and metric queries."""

from __future__ import annotations

from podterm.models import StepMetric

from .connection import get_conn


def add_metric(run_id: str, m: StepMetric) -> None:
    conn = get_conn()
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
    conn = get_conn()
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


def get_metrics(run_id: str) -> list[dict]:
    conn = get_conn()
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


def get_eval_bpb_near(run_id: str, step: int) -> float | None:
    """The val_bpb recorded closest to `step`; None if no eval exists yet."""
    conn = get_conn()
    row = conn.execute(
        "SELECT val_bpb FROM metrics WHERE run_id = ? AND val_bpb IS NOT NULL "
        "ORDER BY ABS(step - ?) LIMIT 1",
        (run_id, step),
    ).fetchone()
    return row["val_bpb"] if row else None
