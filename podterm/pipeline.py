"""Event pipeline — drains the pod event queue, persists to the DB, fans out SSE.

The phase strings "Starting Training" and "Training finished" are load-bearing
here (and in the UI): they reset the metrics buffer on restart and finalize the
run row, respectively.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from dataclasses import asdict

from podterm import db, snapshots
from podterm.events import LogQueue
from podterm.models import MemoryInfo, RunSummary, StepMetric
from podterm.runpod import api_get_telemetry
from podterm.sse import SSEHub, hub

TELEMETRY_POLL_SEC = 5.0


class EventPipeline:
    """Owns the cross-thread event queue and the async drain + telemetry loops."""

    def __init__(self, sse: SSEHub) -> None:
        self.sse = sse
        self.queue: LogQueue = queue.Queue()
        # Per-run state accumulators (pod_id → latest value), drained on finalize.
        self.run_memory: dict[str, MemoryInfo] = {}
        self.run_summary: dict[str, RunSummary] = {}
        self.run_exit_code: dict[str, int] = {}

    # -- lifecycle ----------------------------------------------------------

    def finalize_run(self, pod_id: str) -> None:
        try:
            summary = self.run_summary.pop(pod_id, None)
            memory = self.run_memory.pop(pod_id, None)
            exit_code = self.run_exit_code.pop(pod_id, None)
            db.finish_run(pod_id, summary=summary, memory=memory, exit_code=exit_code)
        except Exception:
            pass

    # -- background loops ---------------------------------------------------

    async def drain_loop(self) -> None:
        """Asyncio background task: drain the pod event queue, persist, fan-out to SSE."""
        metrics_buffer: dict[str, list[StepMetric]] = {}
        last_flush = time.monotonic()

        while True:
            drained = 0
            while drained < 500:
                try:
                    pod_id, kind, payload = self.queue.get_nowait()
                except queue.Empty:
                    break
                drained += 1

                if kind == "log":
                    self.sse.send(pod_id, "log", payload)
                    continue

                t = payload.get("t")
                if t == "metric":
                    event = StepMetric(
                        step=payload.get("step", 0), total_steps=payload.get("total_steps", 0),
                        train_loss=payload.get("train_loss") or 0.0,
                        train_time_ms=payload.get("train_time_ms", 0),
                        step_avg_ms=payload.get("step_avg_ms", 0.0),
                        val_loss=payload.get("val_loss"), val_bpb=payload.get("val_bpb"),
                    )
                    metrics_buffer.setdefault(pod_id, []).append(event)
                    self.sse.send(pod_id, "metric", {
                        "step": event.step, "total_steps": event.total_steps,
                        "train_loss": event.train_loss, "step_avg_ms": event.step_avg_ms,
                        "val_loss": event.val_loss, "val_bpb": event.val_bpb,
                        "train_time_ms": event.train_time_ms,
                    })
                elif t == "memory":
                    event = MemoryInfo(peak_mib=payload.get("peak_mib", 0), reserved_mib=payload.get("reserved_mib", 0))
                    self.run_memory[pod_id] = event
                    self.sse.send(pod_id, "memory", asdict(event))
                    db.update_run(pod_id, peak_memory_mib=event.peak_mib, reserved_memory_mib=event.reserved_mib)
                elif t == "summary":
                    event = RunSummary(
                        final_val_bpb=payload.get("final_val_bpb", 0.0),
                        best_val_bpb=payload.get("best_val_bpb", 0.0),
                    )
                    self.run_summary[pod_id] = event
                    self.sse.send(pod_id, "summary", {**asdict(event), "final_val_loss": payload.get("final_val_loss")})
                elif t == "model":
                    self.sse.send(pod_id, "info", {"model_params": payload.get("model_params")})
                    db.update_run(pod_id, model_params=payload.get("model_params"))
                elif t == "commit":
                    self.sse.send(pod_id, "info", {"commit_hash": payload.get("commit_hash"), "commit_msg": payload.get("commit_msg")})
                    db.update_run(pod_id, commit_hash=payload.get("commit_hash"), commit_msg=payload.get("commit_msg"))
                elif t == "gpu":
                    self.sse.send(pod_id, "info", {"gpu_type": payload.get("gpu_type")})
                    db.update_run(pod_id, gpu_type=payload.get("gpu_type"), gpu_count=payload.get("gpu_count", 1))
                elif t == "phase":
                    phase = str(payload.get("phase", ""))
                    exit_code = payload.get("exit_code")
                    self.sse.send(pod_id, "phase", {"phase": phase, "exit_code": exit_code})
                    if exit_code is not None:
                        self.run_exit_code[pod_id] = exit_code
                    if "Starting Training" in phase:
                        # Clear metrics buffer on restart
                        metrics_buffer.pop(pod_id, None)
                    elif "Training finished" in phase:
                        self.finalize_run(pod_id)
                elif t == "snapshot":
                    # Model snapshot ready on the pod — pull it and run diagnostics off-pod.
                    self.sse.send(pod_id, "snapshot", {"step": payload.get("step"), "final": payload.get("final")})
                    asyncio.create_task(snapshots.handle_snapshot(pod_id, payload))
                elif t == "pull":
                    # Boot/image-pull progress — transient, SSE only (no DB row)
                    self.sse.send(pod_id, "pull", payload)
                elif t == "pod_gone":
                    self.finalize_run(pod_id)
                elif t == "raw":
                    self.sse.send(pod_id, "log", {"line": payload.get("line", "")})

            # Batch-flush metrics to DB every 5 seconds
            now = time.monotonic()
            if now - last_flush > 5 and metrics_buffer:
                for pid, metrics in metrics_buffer.items():
                    try:
                        db.add_metrics_batch(pid, metrics)
                    except Exception:
                        pass
                metrics_buffer.clear()
                last_flush = now

            await asyncio.sleep(0.05)

    async def telemetry_loop(self) -> None:
        """Poll utilization for all pods and emit SSE `telemetry` events.

        Only polls while at least one browser is subscribed — no API traffic
        when nobody is watching. SSEHub.send drops events for unwatched pods.
        """
        log = logging.getLogger("podterm.telemetry")
        while True:
            try:
                if self.sse.has_subscribers():
                    telemetry = await asyncio.to_thread(api_get_telemetry)
                    for pod_id, t in telemetry.items():
                        self.sse.send(pod_id, "telemetry", t)
            except Exception:
                log.exception("telemetry tick failed")  # transient failures just skip a tick
            await asyncio.sleep(TELEMETRY_POLL_SEC)


pipeline = EventPipeline(hub)
