"""Real-time log line parser — extracts training metrics from SSH log output."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Parsed event types
# ---------------------------------------------------------------------------


@dataclass
class StepMetric:
    step: int
    total_steps: int
    train_loss: float
    train_time_ms: int
    step_avg_ms: float
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


@dataclass
class ModelInfo:
    params: int  # "13,183,904" → 13183904


@dataclass
class CommitInfo:
    hash: str
    message: str


@dataclass
class PhaseMarker:
    phase: str  # e.g. "Starting Training", "Training finished"
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Regex patterns (derived from actual training log output)
# ---------------------------------------------------------------------------

# Train step: step:250/20000 train_loss:4.6215 train_time:44960ms step_avg:180.02ms
RE_TRAIN = re.compile(
    r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)\s+train_time:(\d+)ms\s+step_avg:([\d.]+)ms"
)

# Val step: step:3328/20000 val_loss:3.2058 val_bpb:1.3804 train_time:600258ms step_avg:180.37ms
# Note: val lines have val_loss INSTEAD OF train_loss, not after it
RE_VAL = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:(\d+)ms\s+step_avg:([\d.]+)ms"
)

# peak memory: 12942 MiB reserved: 18960 MiB
RE_MEMORY = re.compile(r"peak memory:\s*(\d+)\s*MiB\s+reserved:\s*(\d+)\s*MiB")

# final val_loss:1.3804 best_val_bpb:1.3804
RE_FINAL = re.compile(r"final\s+val_loss:([\d.]+)\s+best_val_bpb:([\d.]+)")

# model params: 13,183,904
RE_MODEL = re.compile(r"model params:\s*([\d,]+)")

# ==> Commit: ae7e269 (New Baseline. End of Day 13.)
RE_COMMIT = re.compile(r"==> Commit:\s*(\w+)\s+\((.+)\)")

# ==> Starting Training (train_gpt.py nproc=1)...
# ==> Training finished (exit 0).
RE_PHASE = re.compile(r"==> (.+?)(?:\s+\(exit (\d+)\))?\.{0,3}\s*$")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_line(line: str) -> StepMetric | MemoryInfo | RunSummary | ModelInfo | CommitInfo | PhaseMarker | None:
    """Parse a single log line into a typed event, or None for non-metric lines.

    Tries patterns in frequency order: step lines are most common.
    """
    line = line.strip()
    if not line:
        return None

    # Validation step (check before train — val lines also start with "step:")
    m = RE_VAL.match(line)
    if m:
        return StepMetric(
            step=int(m[1]),
            total_steps=int(m[2]),
            train_loss=0.0,  # not reported on val lines
            train_time_ms=int(m[5]),
            step_avg_ms=float(m[6]),
            val_loss=float(m[3]),
            val_bpb=float(m[4]),
        )

    # Training step
    m = RE_TRAIN.match(line)
    if m:
        return StepMetric(
            step=int(m[1]),
            total_steps=int(m[2]),
            train_loss=float(m[3]),
            train_time_ms=int(m[4]),
            step_avg_ms=float(m[5]),
        )

    # Memory info
    m = RE_MEMORY.search(line)
    if m:
        return MemoryInfo(peak_mib=int(m[1]), reserved_mib=int(m[2]))

    # Final summary
    m = RE_FINAL.search(line)
    if m:
        return RunSummary(final_val_bpb=float(m[1]), best_val_bpb=float(m[2]))

    # Model params
    m = RE_MODEL.search(line)
    if m:
        return ModelInfo(params=int(m[1].replace(",", "")))

    # Commit info
    m = RE_COMMIT.search(line)
    if m:
        return CommitInfo(hash=m[1], message=m[2])

    # Phase markers (==> ...)
    if line.startswith("==>"):
        m = RE_PHASE.search(line)
        if m:
            exit_code = int(m[2]) if m[2] is not None else None
            return PhaseMarker(phase=m[1], exit_code=exit_code)

    return None
