import json
import queue

import podterm.runpod.cli as cli_mod
from podterm.eventing import poller as poller_mod
from podterm.eventing import startup as startup_mod
from podterm.eventing.poller import PodPoller
from podterm.runpod.api import (
    PodLookupResult,
    PodLookupState,
    _classify_lookup_error,
    api_get_pod_lookup,
)


class StopAfterWaits:
    def __init__(self, max_waits: int) -> None:
        self.max_waits = max_waits
        self.waits = 0

    def is_set(self) -> bool:
        return self.waits >= self.max_waits

    def wait(self, timeout: float) -> bool:
        self.waits += 1
        return self.is_set()


def make_poller() -> PodPoller:
    poller = PodPoller("pod-1", "test-pod", "token", queue.Queue(), base_url="http://events.test")
    poller._boot_enabled = False
    return poller


def queued(poller: PodPoller):
    items = []
    while not poller.log_queue.empty():
        items.append(poller.log_queue.get_nowait())
    return items


def test_wait_for_running_returns_on_running(monkeypatch):
    poller = make_poller()
    monkeypatch.setattr(
        startup_mod,
        "api_get_pod",
        lambda pod_id: {"desiredStatus": "RUNNING"},
    )

    assert poller._wait_for_running() is True
    assert queued(poller) == []


def test_wait_for_running_logs_terminal_status(monkeypatch):
    poller = make_poller()
    monkeypatch.setattr(
        startup_mod,
        "api_get_pod",
        lambda pod_id: {"desiredStatus": "EXITED"},
    )

    assert poller._wait_for_running() is False
    assert queued(poller) == [("pod-1", "log", {"line": "Pod EXITED."})]


def test_wait_for_daemon_returns_log_size_on_health():
    poller = make_poller()
    poller._request = lambda path, timeout: (200, {}, b'{"log_size": 42}')

    assert poller._wait_for_daemon() == 42
    assert queued(poller) == [
        ("pod-1", "log", {"line": "Event daemon connected  |  http://events.test"})
    ]


def test_wait_for_daemon_returns_on_auth_failure():
    poller = make_poller()
    poller._request = lambda path, timeout: (401, {}, b"")

    assert poller._wait_for_daemon() is None
    assert queued(poller) == [
        ("pod-1", "log", {"line": "Event daemon auth failed (token mismatch)."})
    ]


def test_events_loop_emits_batch_and_advances_cursor():
    poller = make_poller()
    calls = []

    def request(path: str, timeout: float):
        calls.append(path)
        if len(calls) == 1:
            return (
                200,
                {},
                json.dumps({"events": [{"t": "metric", "step": 1}], "next": 3}).encode(),
            )
        return 401, {}, b""

    poller._request = request

    poller._events_loop()

    assert calls == [
        "/events?since=0&wait=25",
        "/events?since=3&wait=25",
    ]
    assert queued(poller) == [
        ("pod-1", "event", {"t": "metric", "step": 1}),
        ("pod-1", "log", {"line": "Event daemon auth failed (token mismatch)."}),
    ]


def test_events_loop_requires_two_gone_confirmations():
    poller = make_poller()
    poller._stop_event = StopAfterWaits(max_waits=10)
    poller._request = lambda path, timeout: (0, None, b"")
    poller._check_pod_gone = lambda: True

    poller._events_loop()

    assert queued(poller) == [
        ("pod-1", "event", {"t": "pod_gone"}),
        ("pod-1", "log", {"line": "Pod stopped."}),
    ]


def test_events_loop_suppresses_pod_gone_after_finished():
    poller = make_poller()
    poller._stop_event = StopAfterWaits(max_waits=10)
    calls = 0

    def request(path: str, timeout: float):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (
                200,
                {},
                json.dumps(
                    {"events": [{"t": "phase", "phase": "Training finished"}], "next": 1}
                ).encode(),
            )
        return 0, None, b""

    poller._request = request
    poller._check_pod_gone = lambda: True

    poller._events_loop()

    assert queued(poller) == [
        ("pod-1", "event", {"t": "phase", "phase": "Training finished"}),
        ("pod-1", "log", {"line": "Pod stopped."}),
    ]


def test_check_pod_gone_not_found_is_gone(monkeypatch):
    poller = make_poller()
    monkeypatch.setattr(
        poller_mod,
        "api_get_pod_lookup",
        lambda pod_id: PodLookupResult(PodLookupState.NOT_FOUND),
    )
    assert poller._check_pod_gone() is True


def test_check_pod_gone_unavailable_is_not_gone(monkeypatch):
    poller = make_poller()
    monkeypatch.setattr(
        poller_mod,
        "api_get_pod_lookup",
        lambda pod_id: PodLookupResult(PodLookupState.UNAVAILABLE),
    )
    # A transient CLI/network outage must never be read as "pod gone".
    assert poller._check_pod_gone() is False


def test_check_pod_gone_running_pod_is_not_gone(monkeypatch):
    poller = make_poller()
    monkeypatch.setattr(
        poller_mod,
        "api_get_pod_lookup",
        lambda pod_id: PodLookupResult(PodLookupState.FOUND, {"desiredStatus": "RUNNING"}),
    )
    assert poller._check_pod_gone() is False


def test_check_pod_gone_terminal_status_is_gone(monkeypatch):
    poller = make_poller()
    monkeypatch.setattr(
        poller_mod,
        "api_get_pod_lookup",
        lambda pod_id: PodLookupResult(PodLookupState.FOUND, {"desiredStatus": "EXITED"}),
    )
    assert poller._check_pod_gone() is True


def test_events_loop_unavailable_lookup_never_finalizes(monkeypatch):
    """An `unavailable` lookup must not emit pod_gone, even repeatedly."""
    poller = make_poller()
    poller._stop_event = StopAfterWaits(max_waits=10)
    poller._request = lambda path, timeout: (0, None, b"")
    monkeypatch.setattr(
        poller_mod,
        "api_get_pod_lookup",
        lambda pod_id: PodLookupResult(PodLookupState.UNAVAILABLE),
    )

    poller._events_loop()

    # No pod_gone and no "Pod stopped." — the loop just kept retrying until stop.
    assert queued(poller) == []


def test_events_loop_repeated_not_found_finalizes(monkeypatch):
    """Repeated definitive not_found lookups emit pod_gone (after N confirmations)."""
    poller = make_poller()
    poller._stop_event = StopAfterWaits(max_waits=10)
    poller._request = lambda path, timeout: (0, None, b"")
    monkeypatch.setattr(
        poller_mod,
        "api_get_pod_lookup",
        lambda pod_id: PodLookupResult(PodLookupState.NOT_FOUND),
    )

    poller._events_loop()

    assert queued(poller) == [
        ("pod-1", "event", {"t": "pod_gone"}),
        ("pod-1", "log", {"line": "Pod stopped."}),
    ]


def test_classify_absent_error_is_not_found():
    assert _classify_lookup_error(RuntimeError("pod abc not found")) is PodLookupState.NOT_FOUND
    assert _classify_lookup_error(RuntimeError("no such pod")) is PodLookupState.NOT_FOUND


def test_classify_transient_error_is_unavailable():
    # Transient signals win even when the text also says "not found".
    assert _classify_lookup_error(RuntimeError("503 Service Unavailable")) is PodLookupState.UNAVAILABLE
    assert _classify_lookup_error(RuntimeError("connection timed out")) is PodLookupState.UNAVAILABLE
    assert _classify_lookup_error(RuntimeError("502 Bad Gateway: not found")) is PodLookupState.UNAVAILABLE
    # Unknown shape defaults to the safe (transient) classification.
    assert _classify_lookup_error(RuntimeError("weird gibberish")) is PodLookupState.UNAVAILABLE


def test_lookup_found_on_dict(monkeypatch):
    # api_get_pod_lookup imports _rpc_json from the cli module at call time.
    monkeypatch.setattr(cli_mod, "_rpc_json", lambda *a, **k: {"id": "p1", "desiredStatus": "RUNNING"})
    result = api_get_pod_lookup("p1")
    assert result.state is PodLookupState.FOUND
    assert result.pod == {"id": "p1", "desiredStatus": "RUNNING"}


def test_lookup_found_when_cli_returns_list(monkeypatch):
    # _rpc_json is typed dict | list; a single-element list must still be FOUND.
    monkeypatch.setattr(cli_mod, "_rpc_json", lambda *a, **k: [{"id": "p1"}])
    result = api_get_pod_lookup("p1")
    assert result.state is PodLookupState.FOUND
    assert result.pod == {"id": "p1"}


def test_lookup_not_found_on_empty_result(monkeypatch):
    monkeypatch.setattr(cli_mod, "_rpc_json", lambda *a, **k: None)
    assert api_get_pod_lookup("p1").state is PodLookupState.NOT_FOUND
    monkeypatch.setattr(cli_mod, "_rpc_json", lambda *a, **k: [])
    assert api_get_pod_lookup("p1").state is PodLookupState.NOT_FOUND


def test_lookup_unavailable_on_transient_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(cli_mod, "_rpc_json", boom)
    assert api_get_pod_lookup("p1").state is PodLookupState.UNAVAILABLE


def test_log_loop_drops_buffer_when_log_resets():
    poller = make_poller()
    poller._stop_event = StopAfterWaits(max_waits=2)
    responses = iter(
        [
            (200, {"X-Log-Offset": "4"}, b"held"),
            (200, {"X-Log-Offset": "0"}, b"new\n"),
        ]
    )
    poller._request = lambda path, timeout: next(responses)

    poller._log_loop(offset=0)

    assert queued(poller) == [("pod-1", "log", {"line": "new"})]
