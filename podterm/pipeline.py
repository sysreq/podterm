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

# Cap concurrent off-pod diagnostics across all pods so simultaneous snapshots
# can't exhaust local CPU/RAM/GPU. Kept small and constant on purpose.
SNAPSHOT_CONCURRENCY = 2

# Bound the number of times a per-pod metrics batch is retried after a failed
# DB write before it's dropped, so a permanently-failing write can't grow the
# buffer without limit.
MAX_METRIC_FLUSH_RETRIES = 10


def _first_payload_value(payload: dict, *keys: str) -> object:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


class EventPipeline:
    """Owns the cross-thread event queue and the async drain + telemetry loops."""

    def __init__(self, sse: SSEHub) -> None:
        self.sse = sse
        self.queue: LogQueue = queue.Queue()
        # Per-run state accumulators (pod_id → latest value), drained on finalize.
        self.run_memory: dict[str, MemoryInfo] = {}
        self.run_summary: dict[str, RunSummary] = {}
        self.run_exit_code: dict[str, int] = {}
        self.log = logging.getLogger("podterm.pipeline")
        # In-flight snapshot-diagnostic tasks, tracked so they're not GC'd and
        # so shutdown can await them (see drain_loop's finally block).
        self._snapshot_tasks: set[asyncio.Task] = set()
        # Bound concurrent diagnostics across pods. Created lazily on first use
        # so the bound semaphore binds to the running loop, not import time.
        self._snapshot_sem: asyncio.Semaphore | None = None
        # Per-pod failed-flush retry counters (pod_id → consecutive failures).
        self._metric_flush_failures: dict[str, int] = {}

    # -- lifecycle ----------------------------------------------------------

    def finalize_run(self, pod_id: str) -> None:
        try:
            summary = self.run_summary.pop(pod_id, None)
            memory = self.run_memory.pop(pod_id, None)
            exit_code = self.run_exit_code.pop(pod_id, None)
            db.finish_run(pod_id, summary=summary, memory=memory, exit_code=exit_code)
        except Exception:
            self.log.exception("failed to finalize run pod=%s", pod_id)

    # -- event handlers ----------------------------------------------------

    def _handle_metric(self, pod_id: str, payload: dict, metrics_buffer: dict[str, list[StepMetric]]) -> None:
        # Preserve a valid 0.0 rather than coercing it to a missing-sentinel:
        # `payload.get("train_loss") or 0.0` would discard a genuine zero loss
        # and conflate it with the key being absent. Only default when the key
        # is truly missing (the DB layer stores 0.0 as 0.0).
        train_loss = payload.get("train_loss")
        if train_loss is None:
            train_loss = 0.0
        event = StepMetric(
            step=payload.get("step", 0), total_steps=payload.get("total_steps", 0),
            train_loss=train_loss,
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

    def _handle_memory(self, pod_id: str, payload: dict) -> None:
        event = MemoryInfo(peak_mib=payload.get("peak_mib", 0), reserved_mib=payload.get("reserved_mib", 0))
        self.run_memory[pod_id] = event
        self.sse.send(pod_id, "memory", asdict(event))
        db.update_run(pod_id, peak_memory_mib=event.peak_mib, reserved_memory_mib=event.reserved_mib)

    def _handle_summary(self, pod_id: str, payload: dict) -> None:
        event = RunSummary(
            final_val_bpb=payload.get("final_val_bpb", 0.0),
            best_val_bpb=payload.get("best_val_bpb", 0.0),
        )
        self.run_summary[pod_id] = event
        self.sse.send(pod_id, "summary", {**asdict(event), "final_val_loss": payload.get("final_val_loss")})

    def _handle_info_update(self, pod_id: str, event: str, info: dict, **run_fields: object) -> None:
        self.sse.send(pod_id, event, info)
        if run_fields:
            db.update_run(pod_id, **run_fields)

    def _handle_model(self, pod_id: str, payload: dict) -> None:
        model_params = payload.get("model_params")
        self._handle_info_update(pod_id, "info", {"model_params": model_params}, model_params=model_params)

    def _handle_config(self, pod_id: str, payload: dict) -> None:
        seed = _first_payload_value(payload, "seed")
        seq_len = _first_payload_value(payload, "seq_len", "sequence_length")
        batch_tokens = _first_payload_value(payload, "batch_tokens", "batch_size", "batch")
        info = {
            "seed": seed,
            "seq_len": seq_len,
            "batch_tokens": batch_tokens,
            "grad_accum": payload.get("grad_accum"),
        }
        self._handle_info_update(pod_id, "info", info, seed=seed, seq_len=seq_len, batch_tokens=batch_tokens)

    def _handle_commit(self, pod_id: str, payload: dict) -> None:
        info = {"commit_hash": payload.get("commit_hash"), "commit_msg": payload.get("commit_msg")}
        self._handle_info_update(pod_id, "info", info, **info)

    def _handle_gpu(self, pod_id: str, payload: dict) -> None:
        info = {
            "gpu_type": payload.get("gpu_type"),
            "driver_version": payload.get("driver_version"),
            "cuda_version": payload.get("cuda_version"),
        }
        self._handle_info_update(
            pod_id,
            "info",
            info,
            gpu_type=payload.get("gpu_type"),
            gpu_count=payload.get("gpu_count", 1),
            driver_version=payload.get("driver_version"),
            cuda_version=payload.get("cuda_version"),
        )

    def _handle_phase(self, pod_id: str, payload: dict, metrics_buffer: dict[str, list[StepMetric]]) -> None:
        phase = str(payload.get("phase", ""))
        exit_code = payload.get("exit_code")
        self.sse.send(pod_id, "phase", {"phase": phase, "exit_code": exit_code})
        if exit_code is not None:
            self.run_exit_code[pod_id] = exit_code
        if "Starting Training" in phase:
            # Reset on restart: drop the stale buffer AND its retry counter so the
            # new run gets a fresh flush-retry budget rather than inheriting the
            # previous run's failure count on the same pod_id.
            metrics_buffer.pop(pod_id, None)
            self._metric_flush_failures.pop(pod_id, None)
            return
        if "Training finished" in phase:
            self.finalize_run(pod_id)

    def _handle_snapshot(self, pod_id: str, payload: dict) -> None:
        self.sse.send(pod_id, "snapshot", {"step": payload.get("step"), "final": payload.get("final")})
        task = asyncio.create_task(self._run_snapshot(pod_id, payload))
        # Track the task so it isn't garbage-collected mid-flight and so shutdown
        # can await it; the done-callback discards it and surfaces exceptions.
        self._snapshot_tasks.add(task)
        task.add_done_callback(self._on_snapshot_done)

    def _on_snapshot_done(self, task: asyncio.Task) -> None:
        self._snapshot_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.log.error("snapshot task failed", exc_info=exc)

    async def _run_snapshot(self, pod_id: str, payload: dict) -> None:
        """Run snapshot diagnostics off-loop, then fan out the SSE on the loop.

        The worker thread (inside handle_snapshot) must not touch the asyncio
        SSE queues; it returns the diagnostic event payload(s) and we send them
        here, back on the event loop — one `diagnostic` per processed snapshot.
        A semaphore bounds concurrent diagnostics across pods so simultaneous
        snapshots can't exhaust local CPU/RAM/GPU.
        """
        if self._snapshot_sem is None:
            self._snapshot_sem = asyncio.Semaphore(SNAPSHOT_CONCURRENCY)
        async with self._snapshot_sem:
            events = await snapshots.handle_snapshot(pod_id, payload)
        for event in events:
            self.sse.send(pod_id, "diagnostic", event)

    def _handle_pull(self, pod_id: str, payload: dict) -> None:
        self.sse.send(pod_id, "pull", payload)

    def _handle_pod_gone(self, pod_id: str, payload: dict) -> None:
        self.finalize_run(pod_id)

    def _handle_raw(self, pod_id: str, payload: dict) -> None:
        self.sse.send(pod_id, "log", {"line": payload.get("line", "")})

    _EVENT_HANDLERS = {
        "memory": "_handle_memory",
        "summary": "_handle_summary",
        "model": "_handle_model",
        "config": "_handle_config",
        "commit": "_handle_commit",
        "gpu": "_handle_gpu",
        "snapshot": "_handle_snapshot",
        "pull": "_handle_pull",
        "pod_gone": "_handle_pod_gone",
        "raw": "_handle_raw",
    }

    def _handle_event(self, pod_id: str, payload: dict, metrics_buffer: dict[str, list[StepMetric]]) -> None:
        t = payload.get("t")
        if t == "metric":
            self._handle_metric(pod_id, payload, metrics_buffer)
            return
        if t == "phase":
            self._handle_phase(pod_id, payload, metrics_buffer)
            return

        handler_name = self._EVENT_HANDLERS.get(t)
        if handler_name is None:
            return
        getattr(self, handler_name)(pod_id, payload)

    def _handle_queue_item(
        self,
        pod_id: str,
        kind: str,
        payload: dict,
        metrics_buffer: dict[str, list[StepMetric]],
    ) -> None:
        if kind == "log":
            self.sse.send(pod_id, "log", payload)
            return
        self._handle_event(pod_id, payload, metrics_buffer)

    def _flush_metrics(self, metrics_buffer: dict[str, list[StepMetric]]) -> None:
        # Remove a per-pod batch only once its write commits; a failed write
        # (transient SQLite lock, serialization error) keeps the batch buffered
        # for a bounded number of retries so we don't silently lose metrics, but
        # also can't grow the buffer without limit on a permanent failure.
        for pid in list(metrics_buffer.keys()):
            metrics = metrics_buffer[pid]
            try:
                db.add_metrics_batch(pid, metrics)
            except Exception:
                failures = self._metric_flush_failures.get(pid, 0) + 1
                self._metric_flush_failures[pid] = failures
                if failures >= MAX_METRIC_FLUSH_RETRIES:
                    self.log.exception(
                        "dropping metrics batch after %s failed flush attempts pod=%s count=%s",
                        failures, pid, len(metrics),
                    )
                    del metrics_buffer[pid]
                    self._metric_flush_failures.pop(pid, None)
                else:
                    self.log.warning(
                        "failed to persist metrics batch pod=%s count=%s (attempt %s/%s) — retrying",
                        pid, len(metrics), failures, MAX_METRIC_FLUSH_RETRIES,
                    )
                continue
            # Commit succeeded: drop the batch and reset its failure counter.
            del metrics_buffer[pid]
            self._metric_flush_failures.pop(pid, None)

    # -- background loops ---------------------------------------------------

    async def drain_loop(self) -> None:
        """Asyncio background task: drain the pod event queue, persist, fan-out to SSE."""
        metrics_buffer: dict[str, list[StepMetric]] = {}
        last_flush = time.monotonic()

        try:
            while True:
                drained = 0
                while drained < 500:
                    try:
                        pod_id, kind, payload = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    drained += 1

                    # Error boundary per queue item: one malformed event, bad
                    # numeric conversion, or DB hiccup must not kill the sole
                    # drain task that drives all metric persistence + live updates.
                    try:
                        self._handle_queue_item(pod_id, kind, payload, metrics_buffer)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        event_type = payload.get("t") if isinstance(payload, dict) else None
                        self.log.exception(
                            "event processing failed",
                            extra={"pod_id": pod_id, "event_type": event_type},
                        )

                # Batch-flush metrics to DB every 5 seconds
                now = time.monotonic()
                if now - last_flush > 5 and metrics_buffer:
                    self._flush_metrics(metrics_buffer)
                    last_flush = now

                await asyncio.sleep(0.05)
        finally:
            # On task cancellation at shutdown, persist any buffered metrics and
            # let in-flight diagnostics finish/clean up before we exit. The flush
            # is synchronous, so it always completes here. Awaiting the snapshot
            # tasks only takes effect if the canceller awaits this task after
            # cancelling it (see server.py shutdown ordering); we use the
            # shield-and-reawait pattern so the cancellation thrown into our
            # `await` doesn't abandon the in-flight diagnostics half-done.
            if metrics_buffer:
                self._flush_metrics(metrics_buffer)
            await self._drain_snapshot_tasks()

    async def _drain_snapshot_tasks(self) -> None:
        """Wait for in-flight snapshot tasks to finish, surviving cancellation.

        A plain `await asyncio.shield(gather(...))` inside an already-cancelled
        task re-raises CancelledError at the await point, abandoning the still-
        running tasks. Re-awaiting the same shielded future swallows the injected
        cancellation until the underlying gather actually completes.
        """
        if not self._snapshot_tasks:
            return
        inner = asyncio.gather(*self._snapshot_tasks, return_exceptions=True)
        shielded = asyncio.shield(inner)
        while True:
            try:
                await shielded
                return
            except asyncio.CancelledError:
                if inner.done():
                    return
                # Cancellation was aimed at us, not the diagnostics: keep waiting.
                continue

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
