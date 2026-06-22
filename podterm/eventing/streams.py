from __future__ import annotations

from collections.abc import Callable

from .util import parse_json_body

EVENT_WAIT_SEC = 25
EVENT_TIMEOUT_SEC = 40.0
EVENT_BACKOFF_INITIAL_SEC = 2.0
EVENT_BACKOFF_MAX_SEC = 30.0
LOG_POLL_SEC = 2.0
LOG_TIMEOUT_SEC = 15.0
LOG_LIMIT_BYTES = 262144
LOG_TAIL_BYTES = 65536

Request = Callable[[str, float], tuple[int, object, bytes]]
EmitLog = Callable[[str], None]
EmitEvent = Callable[[dict], None]
PodGoneCheck = Callable[[], bool]


class EventStream:
    """Long-poll structured event batches and stop when the pod is gone."""

    def __init__(
        self,
        pod_id: str,
        request: Request,
        emit_log: EmitLog,
        emit_event: EmitEvent,
        stop_event,
        pod_gone: PodGoneCheck,
    ) -> None:
        self.pod_id = pod_id
        self.request = request
        self.emit_log = emit_log
        self.emit_event = emit_event
        self.stop_event = stop_event
        self.pod_gone = pod_gone

    def run(self) -> None:
        cur = 0
        finished = False
        gone_checks = 0
        backoff = EVENT_BACKOFF_INITIAL_SEC
        while not self.stop_event.is_set():
            data, should_stop = self._fetch_batch(cur)
            if should_stop:
                return
            if data is not None:
                gone_checks = 0
                backoff = EVENT_BACKOFF_INITIAL_SEC
                cur, batch_finished = self._emit_batch(data, cur)
                finished = finished or batch_finished
                continue

            gone_checks, should_stop = self._record_gone_check(gone_checks, finished)
            if should_stop:
                return
            self.stop_event.wait(backoff)
            backoff = min(backoff * 2, EVENT_BACKOFF_MAX_SEC)

    def _fetch_batch(self, cur: int) -> tuple[dict | None, bool]:
        status, _, body = self.request(f"/events?since={cur}&wait={EVENT_WAIT_SEC}", EVENT_TIMEOUT_SEC)
        match status:
            case 200:
                return parse_json_body(body), False
            case 401:
                self.emit_log("Event daemon auth failed (token mismatch).")
                return None, True
            case _:
                return None, False

    def _emit_batch(self, data: dict, cur: int) -> tuple[int, bool]:
        finished = False
        for ev in data.get("events", []):
            if ev.get("t") == "phase" and "Training finished" in str(ev.get("phase", "")):
                finished = True
            self.emit_event(ev)
        return int(data.get("next", cur)), finished

    def _record_gone_check(self, gone_checks: int, finished: bool) -> tuple[int, bool]:
        gone_checks = gone_checks + 1 if self.pod_gone() else 0
        if gone_checks < 2:
            return gone_checks, False
        if not finished:
            self.emit_event({"t": "pod_gone"})
        self.emit_log("Pod stopped.")
        return gone_checks, True


class LogTailer:
    """Poll raw log bytes and emit complete decoded lines."""

    def __init__(self, request: Request, emit_log: EmitLog, stop_event) -> None:
        self.request = request
        self.emit_log = emit_log
        self.stop_event = stop_event

    def run(self, offset: int) -> None:
        buf = b""
        while not self.stop_event.is_set():
            status, headers, body = self.request(
                f"/log?offset={offset}&limit={LOG_LIMIT_BYTES}",
                LOG_TIMEOUT_SEC,
            )
            if status == 200:
                offset, buf = self._consume(offset, buf, headers, body)
            self.stop_event.wait(LOG_POLL_SEC)

    def _consume(self, offset: int, buf: bytes, headers: object, body: bytes) -> tuple[int, bytes]:
        new_offset = int(headers.get("X-Log-Offset", offset + len(body)))
        buf = b"" if new_offset < offset else buf
        if body:
            buf += body
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                self.emit_log(raw.decode("utf-8", errors="replace").rstrip("\r"))
        return new_offset, buf
