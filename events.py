"""NINA event-websocket subscriber and in-memory event store.

The orchestrator needs to react to NINA events (exposure complete, autofocus
done, safety unsafe, sequence error) within ~1s, without burning tokens on
poll-storms. NINA exposes a WebSocket at /v2/api/event-websocket; we run a
background task that pipes incoming events into a bounded in-memory buffer,
and the agent calls poll_events_since(cursor) to drain new events.

This design keeps the MCP tool layer synchronous (no streaming required)
while still being event-driven from the agent's point of view.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any, AsyncIterator, Optional

import aiohttp


logger = logging.getLogger(__name__)


class EventStore:
    """Bounded ring buffer with a monotonic cursor.

    `cursor` reflects the *total number of events ever appended*, so a caller
    that polled with cursor=N earlier will only get events appended after
    their previous poll. When the buffer overflows, oldest events are dropped
    but the cursor keeps counting — clients may miss events but will not see
    duplicates.
    """

    def __init__(self, max_size: int = 1000):
        self._buf: deque[tuple[int, dict[str, Any]]] = deque(maxlen=max_size)
        self._next_id = 0  # monotonic, equals total appended count

    def append(self, event: dict[str, Any]) -> None:
        self._buf.append((self._next_id, event))
        self._next_id += 1

    def since(
        self,
        cursor: Optional[int],
        *,
        max_events: int = 100,
    ) -> dict[str, Any]:
        """Return events with id >= cursor (or all if cursor is None).

        The returned next_cursor is what the caller should pass on the next
        poll to avoid duplicates.
        """
        start = 0 if cursor is None else cursor
        out: list[dict[str, Any]] = []
        last_id = start
        for eid, ev in self._buf:
            if eid < start:
                continue
            out.append(ev)
            last_id = eid + 1
            if len(out) >= max_events:
                break
        if not out:
            last_id = max(start, self._next_id)
        return {"events": out, "next_cursor": last_id}


async def run_subscriber(
    store: EventStore,
    messages: AsyncIterator[Any],
) -> None:
    """Drain an async iterator of messages into the store.

    Non-dict messages are dropped silently so a misbehaving WebSocket frame
    can't kill the loop.
    """
    async for msg in messages:
        if isinstance(msg, dict):
            store.append(msg)


async def _aiohttp_ws_messages(url: str) -> AsyncIterator[dict[str, Any]]:
    """Production WebSocket source — yields parsed JSON events from NINA."""
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        yield json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON websocket frame, skipping: %r", msg.data[:200])
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break


async def start_background_subscriber(
    store: EventStore,
    host: str,
    port: int,
) -> asyncio.Task:
    """Launch the websocket subscriber as a background task.

    Reconnects on failure with exponential backoff so a NINA restart doesn't
    permanently break the event flow.
    """
    url = f"ws://{host}:{port}/v2/api/event-websocket"

    async def _loop() -> None:
        backoff = 1.0
        while True:
            try:
                logger.info("event-websocket: connecting to %s", url)
                await run_subscriber(store, _aiohttp_ws_messages(url))
                backoff = 1.0  # reset on clean disconnect
            except Exception as e:
                logger.warning("event-websocket subscriber error: %s — retrying in %.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    return asyncio.create_task(_loop(), name="nina-event-subscriber")


async def tool_poll_events_since(
    store: EventStore,
    *,
    cursor: Optional[int] = None,
    max_events: int = 100,
) -> dict[str, Any]:
    """MCP-tool entry point: return events since cursor in the standard envelope."""
    try:
        result = store.since(cursor, max_events=max_events)
        return {
            "Success": True,
            "Message": f"Returned {len(result['events'])} event(s); next_cursor={result['next_cursor']}",
            "Details": {
                "Events": result["events"],
                "NextCursor": result["next_cursor"],
            },
            "Type": "NINA_EVENTS",
        }
    except Exception as e:
        return {
            "Success": False,
            "Error": str(e),
            "ErrorType": "EventStoreError",
            "ErrorDetails": {},
            "StatusCode": 500,
            "Type": "NINA_EVENTS",
        }
