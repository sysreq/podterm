import asyncio

from podterm import pipeline as pipeline_mod
from podterm.pipeline import EventPipeline


class FakeSSE:
    def __init__(self):
        self.events = []

    def send(self, pod_id, event, payload):
        self.events.append((pod_id, event, payload))

    def has_subscribers(self):
        return False


def test_metric_event_buffers_and_fans_out():
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    metrics = {}

    pipe._handle_event("pod-1", {
        "t": "metric",
        "step": 7,
        "total_steps": 20,
        "train_loss": 1.5,
        "train_time_ms": 1234,
        "step_avg_ms": 10.5,
        "val_bpb": 0.9,
    }, metrics)

    assert metrics["pod-1"][0].step == 7
    assert sse.events == [("pod-1", "metric", {
        "step": 7,
        "total_steps": 20,
        "train_loss": 1.5,
        "step_avg_ms": 10.5,
        "val_loss": None,
        "val_bpb": 0.9,
        "train_time_ms": 1234,
    })]


def test_phase_event_resets_buffer_and_finalizes(monkeypatch):
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    metrics = {"pod-1": [object()]}
    finished = []
    monkeypatch.setattr(
        pipeline_mod.db,
        "finish_run",
        lambda pod_id, **kwargs: finished.append((pod_id, kwargs.get("exit_code"))),
    )

    pipe._handle_event("pod-1", {"t": "phase", "phase": "Starting Training"}, metrics)
    assert "pod-1" not in metrics

    pipe._handle_event("pod-1", {
        "t": "phase",
        "phase": "Training finished",
        "exit_code": 0,
    }, metrics)

    assert finished == [("pod-1", 0)]
    assert "pod-1" not in pipe.run_exit_code


def test_config_event_normalizes_repro_aliases(monkeypatch):
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    updates = []
    monkeypatch.setattr(
        pipeline_mod.db,
        "update_run",
        lambda pod_id, **fields: updates.append((pod_id, fields)),
    )

    pipe._handle_event("pod-1", {
        "t": "config",
        "seed": 123,
        "sequence_length": 1024,
        "batch_size": 8192,
    }, {})

    assert sse.events == [("pod-1", "info", {
        "seed": 123,
        "seq_len": 1024,
        "batch_tokens": 8192,
        "grad_accum": None,
    })]
    assert updates == [("pod-1", {"seed": 123, "seq_len": 1024, "batch_tokens": 8192})]


def test_info_event_handlers_update_run_fields(monkeypatch):
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    updates = []
    monkeypatch.setattr(
        pipeline_mod.db,
        "update_run",
        lambda pod_id, **fields: updates.append((pod_id, fields)),
    )

    pipe._handle_event("pod-1", {"t": "model", "model_params": 1234}, {})
    pipe._handle_event("pod-1", {
        "t": "commit",
        "commit_hash": "abc",
        "commit_msg": "train",
    }, {})
    pipe._handle_event("pod-1", {
        "t": "gpu",
        "gpu_type": "A100",
        "gpu_count": 2,
        "driver_version": "550",
        "cuda_version": "12.4",
    }, {})

    assert sse.events == [
        ("pod-1", "info", {"model_params": 1234}),
        ("pod-1", "info", {"commit_hash": "abc", "commit_msg": "train"}),
        ("pod-1", "info", {
            "gpu_type": "A100",
            "driver_version": "550",
            "cuda_version": "12.4",
        }),
    ]
    assert updates == [
        ("pod-1", {"model_params": 1234}),
        ("pod-1", {"commit_hash": "abc", "commit_msg": "train"}),
        ("pod-1", {
            "gpu_type": "A100",
            "gpu_count": 2,
            "driver_version": "550",
            "cuda_version": "12.4",
        }),
    ]


def test_passthrough_and_lifecycle_events(monkeypatch):
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    finished = []
    monkeypatch.setattr(
        pipeline_mod.db,
        "finish_run",
        lambda pod_id, **kwargs: finished.append((pod_id, kwargs.get("exit_code"))),
    )

    pipe._handle_event("pod-1", {"t": "pull", "done": False}, {})
    pipe._handle_event("pod-1", {"t": "raw", "line": "hello"}, {})
    pipe.run_exit_code["pod-1"] = 7
    pipe._handle_event("pod-1", {"t": "pod_gone"}, {})

    assert sse.events == [
        ("pod-1", "pull", {"t": "pull", "done": False}),
        ("pod-1", "log", {"line": "hello"}),
    ]
    assert finished == [("pod-1", 7)]


def test_snapshot_event_fans_out_and_tracks_task(monkeypatch):
    # F8/F9: the `snapshot` SSE goes out immediately; diagnostics run in a tracked
    # task that fans out the `diagnostic` SSE *on the loop* once handle_snapshot
    # returns the event payload — the worker thread never calls hub.send.
    sse = FakeSSE()
    pipe = EventPipeline(sse)

    async def fake_handle_snapshot(pod_id, payload):
        return [{"step": payload["step"], "status": "ok", "final": True, "health": None, "sections": []}]

    monkeypatch.setattr(pipeline_mod.snapshots, "handle_snapshot", fake_handle_snapshot)

    async def scenario():
        pipe._handle_event("pod-1", {"t": "snapshot", "step": 10, "final": True}, {})
        # The snapshot task must be tracked (not fire-and-forget) so it isn't
        # GC'd and shutdown can await it.
        assert len(pipe._snapshot_tasks) == 1
        await asyncio.gather(*pipe._snapshot_tasks)

    asyncio.run(scenario())

    assert ("pod-1", "snapshot", {"step": 10, "final": True}) in sse.events
    assert ("pod-1", "diagnostic", {
        "step": 10, "status": "ok", "final": True, "health": None, "sections": [],
    }) in sse.events
    # Done-callback discards the finished task from the tracking set.
    assert pipe._snapshot_tasks == set()


def test_snapshot_empty_result_skips_diagnostic_sse(monkeypatch):
    # When handle_snapshot returns no events (diagnostics disabled / coalesced
    # away because another task holds the lock), no `diagnostic` SSE is emitted.
    sse = FakeSSE()
    pipe = EventPipeline(sse)

    async def fake_handle_snapshot(pod_id, payload):
        return []

    monkeypatch.setattr(pipeline_mod.snapshots, "handle_snapshot", fake_handle_snapshot)

    async def scenario():
        pipe._handle_event("pod-1", {"t": "snapshot", "step": 5, "final": False}, {})
        await asyncio.gather(*pipe._snapshot_tasks)

    asyncio.run(scenario())

    assert ("pod-1", "snapshot", {"step": 5, "final": False}) in sse.events
    assert not any(event == "diagnostic" for _, event, _ in sse.events)


def test_snapshot_fans_out_every_coalesced_diagnostic(monkeypatch):
    # F8/F9: when several pending snapshots are processed under one lock,
    # every processed snapshot fans out its own `diagnostic` SSE (matching the
    # old per-snapshot hub.send), not just the last one.
    sse = FakeSSE()
    pipe = EventPipeline(sse)

    async def fake_handle_snapshot(pod_id, payload):
        return [
            {"step": 10, "status": "ok", "final": False, "health": None, "sections": []},
            {"step": 20, "status": "warn", "final": True, "health": None, "sections": []},
        ]

    monkeypatch.setattr(pipeline_mod.snapshots, "handle_snapshot", fake_handle_snapshot)

    async def scenario():
        pipe._handle_event("pod-1", {"t": "snapshot", "step": 20, "final": True}, {})
        await asyncio.gather(*pipe._snapshot_tasks)

    asyncio.run(scenario())

    diag_steps = [p["step"] for _, ev, p in sse.events if ev == "diagnostic"]
    assert diag_steps == [10, 20]


def test_metric_train_loss_zero_is_preserved():
    # F7: a valid 0.0 train_loss must survive (not be coerced to a sentinel),
    # and must be distinguishable from a genuinely-missing key (which defaults).
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    metrics = {}

    pipe._handle_event("pod-1", {"t": "metric", "step": 1, "train_loss": 0.0}, metrics)
    assert metrics["pod-1"][0].train_loss == 0.0
    assert metrics["pod-1"][0].train_loss is not None

    pipe._handle_event("pod-1", {"t": "metric", "step": 2}, metrics)
    assert metrics["pod-1"][1].train_loss == 0.0


def test_flush_metrics_keeps_batch_buffered_on_failure(monkeypatch):
    # F4: a failed db.add_metrics_batch must NOT discard the batch — it stays
    # buffered for bounded retry rather than being silently lost.
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    calls = []

    def boom(pid, metrics):
        calls.append(pid)
        raise RuntimeError("locked")

    monkeypatch.setattr(pipeline_mod.db, "add_metrics_batch", boom)

    buffer = {"pod-1": [object()]}
    pipe._flush_metrics(buffer)

    # Batch is retained for retry, failure counter incremented.
    assert "pod-1" in buffer
    assert pipe._metric_flush_failures["pod-1"] == 1
    assert calls == ["pod-1"]


def test_flush_metrics_drops_batch_after_retry_cap(monkeypatch):
    # F4: bounded retry — after MAX_METRIC_FLUSH_RETRIES the batch is dropped so
    # a permanent failure can't grow the buffer without limit.
    sse = FakeSSE()
    pipe = EventPipeline(sse)

    def boom(pid, metrics):
        raise RuntimeError("locked")

    monkeypatch.setattr(pipeline_mod.db, "add_metrics_batch", boom)

    buffer = {"pod-1": [object()]}
    for _ in range(pipeline_mod.MAX_METRIC_FLUSH_RETRIES):
        pipe._flush_metrics(buffer)

    assert "pod-1" not in buffer
    assert "pod-1" not in pipe._metric_flush_failures


def test_starting_training_resets_flush_failure_counter():
    # A run restart ("Starting Training") must drop the stale per-pod retry
    # counter so the new run gets a fresh flush-retry budget on the same pod_id.
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    pipe._metric_flush_failures["pod-1"] = 7
    metrics = {"pod-1": [object()]}

    pipe._handle_event("pod-1", {"t": "phase", "phase": "Starting Training"}, metrics)

    assert "pod-1" not in metrics
    assert "pod-1" not in pipe._metric_flush_failures


def test_flush_metrics_clears_batch_only_after_commit(monkeypatch):
    # F4: a successful write removes the batch and resets the failure counter.
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    written = []
    monkeypatch.setattr(pipeline_mod.db, "add_metrics_batch", lambda pid, m: written.append(pid))

    buffer = {"pod-1": [object()], "pod-2": [object()]}
    pipe._flush_metrics(buffer)

    assert buffer == {}
    assert sorted(written) == ["pod-1", "pod-2"]


def test_drain_loop_survives_a_bad_event(monkeypatch):
    # F3: a handler raising on one event must not kill the drain loop; later
    # events still process.
    sse = FakeSSE()
    pipe = EventPipeline(sse)

    real_handle_metric = pipe._handle_metric

    def flaky_metric(pod_id, payload, metrics_buffer):
        if payload.get("step") == 1:
            raise ValueError("bad event")
        return real_handle_metric(pod_id, payload, metrics_buffer)

    monkeypatch.setattr(pipe, "_handle_metric", flaky_metric)

    pipe.queue.put(("pod-1", "event", {"t": "metric", "step": 1, "train_loss": 1.0}))
    pipe.queue.put(("pod-1", "event", {"t": "metric", "step": 2, "train_loss": 2.0}))

    async def scenario():
        task = asyncio.create_task(pipe.drain_loop())
        # Give the loop time to drain both items (it sleeps 0.05s per tick).
        for _ in range(20):
            await asyncio.sleep(0.02)
            if any(p.get("step") == 2 for _, ev, p in sse.events if ev == "metric"):
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    steps = [p["step"] for _, ev, p in sse.events if ev == "metric"]
    # The bad event (step 1) was swallowed; the good event (step 2) still fanned out.
    assert 2 in steps
