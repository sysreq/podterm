"""SSE fan-out hub — per-pod subscriber queues + event broadcast."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator


class SSEHub:
    """Fans structured events out to all browser clients subscribed to a pod."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[asyncio.Queue]] = {}  # pod_id → client queues

    def has_subscribers(self) -> bool:
        return any(self.subscribers.values())

    def send(self, pod_id: str, event_type: str, data: dict) -> None:
        """Fan out an SSE event to all subscribers for a pod."""
        clients = self.subscribers.get(pod_id)
        if not clients:
            return
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        dead: list[asyncio.Queue] = []
        for q in clients:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            clients.remove(q)
            # Drain and send sentinel so the generator exits and EventSource reconnects
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
            try:
                q.put_nowait(None)
            except Exception:
                pass

    def subscribe(self, pod_id: str) -> asyncio.Queue:
        """Register a new client queue for a pod and return it."""
        client_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subscribers.setdefault(pod_id, []).append(client_queue)
        return client_queue

    def _unsubscribe(self, pod_id: str, client_queue: asyncio.Queue) -> None:
        clients = self.subscribers.get(pod_id, [])
        if client_queue in clients:
            clients.remove(client_queue)

    async def stream(self, pod_id: str, client_queue: asyncio.Queue) -> AsyncIterator[str]:
        """Yield SSE messages for a subscribed client until it disconnects."""
        try:
            # Send initial keepalive
            yield ": connected\n\n"
            while True:
                msg = await client_queue.get()
                if msg is None:
                    return  # Sentinel: server ejected us (queue overflow)
                yield msg
        except asyncio.CancelledError:
            pass
        finally:
            self._unsubscribe(pod_id, client_queue)


hub = SSEHub()
