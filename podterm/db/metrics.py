"""Step metric persistence and metric queries."""

from __future__ import annotations

from podterm.models import StepMetric

from .connection import with_conn

# Upsert that MERGES partial metric events for the same (run_id, step): a later
# event carrying only some fields must not erase values recorded by an earlier
# event. COALESCE(excluded.col, metrics.col) keeps the existing value whenever
# the incoming column is NULL.
_UPSERT_METRIC = """
    INSERT INTO metrics
        (run_id, step, total_steps, train_loss, val_loss, val_bpb, train_time_ms, step_avg_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(run_id, step) DO UPDATE SET
        total_steps   = COALESCE(excluded.total_steps, metrics.total_steps),
        train_loss    = COALESCE(excluded.train_loss, metrics.train_loss),
        val_loss      = COALESCE(excluded.val_loss, metrics.val_loss),
        val_bpb       = COALESCE(excluded.val_bpb, metrics.val_bpb),
        train_time_ms = COALESCE(excluded.train_time_ms, metrics.train_time_ms),
        step_avg_ms   = COALESCE(excluded.step_avg_ms, metrics.step_avg_ms)
"""


def _metric_values(run_id: str, m: StepMetric) -> tuple[object, ...]:
    # NB: explicit None-checks, not truthiness — a valid 0.0 must round-trip as
    # 0.0, not be coerced to SQL NULL (which would then be read as "missing").
    return (
        run_id,
        m.step,
        m.total_steps,
        m.train_loss if m.train_loss is not None else None,
        m.val_loss,
        m.val_bpb,
        m.train_time_ms,
        m.step_avg_ms,
    )


def add_metric(run_id: str, m: StepMetric) -> None:
    with with_conn() as conn:
        conn.execute(_UPSERT_METRIC, _metric_values(run_id, m))
        conn.commit()


def add_metrics_batch(run_id: str, metrics: list[StepMetric]) -> None:
    if not metrics:
        return
    with with_conn() as conn:
        conn.executemany(_UPSERT_METRIC, [_metric_values(run_id, m) for m in metrics])
        conn.commit()


def get_metrics(run_id: str) -> list[dict]:
    with with_conn() as conn:
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
    with with_conn() as conn:
        row = conn.execute(
            "SELECT val_bpb FROM metrics WHERE run_id = ? AND val_bpb IS NOT NULL "
            "ORDER BY ABS(step - ?) LIMIT 1",
            (run_id, step),
        ).fetchone()
    return row["val_bpb"] if row else None
