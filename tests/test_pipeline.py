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
