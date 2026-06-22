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


def test_snapshot_event_fans_out_and_starts_task(monkeypatch):
    sse = FakeSSE()
    pipe = EventPipeline(sse)
    tasks = []

    def fake_create_task(coro):
        tasks.append(coro)
        coro.close()

    monkeypatch.setattr(pipeline_mod.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(pipeline_mod.snapshots, "handle_snapshot", lambda pod_id, payload: _noop())

    pipe._handle_event("pod-1", {"t": "snapshot", "step": 10, "final": True}, {})

    assert sse.events == [("pod-1", "snapshot", {"step": 10, "final": True})]
    assert len(tasks) == 1


async def _noop():
    return None
