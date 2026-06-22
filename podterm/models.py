"""Structured event types shared by the live pipeline and the DB layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepMetric:
    step: int
    total_steps: int
    train_time_ms: int
    step_avg_ms: float
    train_loss: float | None = None
    val_loss: float | None = None
    val_bpb: float | None = None


@dataclass
class MemoryInfo:
    peak_mib: int
    reserved_mib: int


@dataclass
class RunSummary:
    # NB: the log's "final val_loss" field actually reports val_bpb, not val_loss
    final_val_bpb: float
    best_val_bpb: float
