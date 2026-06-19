"""SQLite storage facade for run history, metrics, and diagnostics."""

from __future__ import annotations

from .connection import close, get_conn
from .diagnostics import add_diagnostics, get_diagnostics
from .metrics import add_metric, add_metrics_batch, get_eval_bpb_near, get_metrics, get_metrics_multi
from .runs import (
    create_run,
    finish_run,
    get_distinct_branches,
    get_distinct_gpus,
    get_run,
    list_runs,
    update_run,
)

__all__ = [
    "add_diagnostics",
    "add_metric",
    "add_metrics_batch",
    "close",
    "create_run",
    "finish_run",
    "get_conn",
    "get_diagnostics",
    "get_distinct_branches",
    "get_distinct_gpus",
    "get_eval_bpb_near",
    "get_metrics",
    "get_metrics_multi",
    "get_run",
    "list_runs",
    "update_run",
]
