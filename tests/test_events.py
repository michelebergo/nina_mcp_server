"""Tests for events — NINA event-websocket subscriber and event store.

The store is a bounded in-memory buffer with a monotonic cursor. MCP tools
that need event-driven behavior call poll_events_since(cursor); a background
asyncio task feeds the store from NINA's /v2/api/event-websocket.
"""

import asyncio
import pytest

from events import EventStore, run_subscriber, tool_poll_events_since


class TestEventStore:
    def test_new_store_is_empty(self):
        s = EventStore()
        result = s.since(0)
        assert result["events"] == []
        assert result["next_cursor"] == 0

    def test_append_increments_cursor(self):
        s = EventStore()
        s.append({"Event": "SLEW_START", "Time": "..."})
        result = s.since(0)
        assert len(result["events"]) == 1
        assert result["next_cursor"] == 1

    def test_since_returns_only_events_after_cursor(self):
        s = EventStore()
        for i in range(5):
            s.append({"Event": f"E{i}"})
        result = s.since(2)
        assert [e["Event"] for e in result["events"]] == ["E2", "E3", "E4"]
        assert result["next_cursor"] == 5

    def test_since_none_returns_all(self):
        """A caller with no cursor (first poll) gets everything buffered."""
        s = EventStore()
        s.append({"Event": "A"})
        s.append({"Event": "B"})
        result = s.since(None)
        assert [e["Event"] for e in result["events"]] == ["A", "B"]
        assert result["next_cursor"] == 2

    def test_since_at_current_cursor_returns_empty(self):
        s = EventStore()
        s.append({"Event": "A"})
        result = s.since(1)
        assert result["events"] == []
        assert result["next_cursor"] == 1

    def test_max_events_caps_response(self):
        s = EventStore()
        for i in range(50):
            s.append({"Event": f"E{i}"})
        result = s.since(0, max_events=10)
        assert len(result["events"]) == 10
        assert result["next_cursor"] == 10  # caller polls again from 10

    def test_buffer_overflow_drops_oldest(self):
        """Bounded buffer — once max_size is reached, oldest events fall off.
        Cursor MUST continue monotonically increasing."""
        s = EventStore(max_size=3)
        for i in range(5):
            s.append({"Event": f"E{i}"})
        # E0 and E1 should have been dropped; E2, E3, E4 remain
        result = s.since(0)
        events = [e["Event"] for e in result["events"]]
        assert events == ["E2", "E3", "E4"]
        # Cursor reflects total events seen (5), not buffer size
        assert result["next_cursor"] == 5

    def test_cursor_never_goes_backward(self):
        s = EventStore()
        for i in range(3):
            s.append({"Event": f"E{i}"})
        first = s.since(0)
        # Caller now has next_cursor=3
        s.append({"Event": "E3"})
        second = s.since(first["next_cursor"])
        assert second["next_cursor"] == 4
        assert [e["Event"] for e in second["events"]] == ["E3"]


class TestSubscriberLoop:
    """run_subscriber pipes events from an async iterator into the store.
    The real iterator is an aiohttp WebSocket; tests inject a fake one."""

    @pytest.mark.asyncio
    async def test_pipes_events_into_store(self):
        store = EventStore()

        async def fake_ws_messages():
            yield {"Event": "SLEW_START"}
            yield {"Event": "EXPOSURE_COMPLETE", "ImagePath": "/tmp/x.fits"}

        await run_subscriber(store, fake_ws_messages())

        result = store.since(0)
        assert [e["Event"] for e in result["events"]] == ["SLEW_START", "EXPOSURE_COMPLETE"]

    @pytest.mark.asyncio
    async def test_subscriber_swallows_non_dict_messages(self):
        """If the websocket sends garbage (e.g. None or strings), keep going."""
        store = EventStore()

        async def fake_ws_messages():
            yield {"Event": "OK"}
            yield "garbage"
            yield None
            yield {"Event": "ALSO_OK"}

        await run_subscriber(store, fake_ws_messages())

        events = [e["Event"] for e in store.since(0)["events"]]
        assert events == ["OK", "ALSO_OK"]


class TestPollToolEnvelope:
    @pytest.mark.asyncio
    async def test_first_poll_returns_all_and_next_cursor(self):
        store = EventStore()
        store.append({"Event": "A"})
        store.append({"Event": "B"})

        env = await tool_poll_events_since(store, cursor=None)
        assert env["Success"] is True
        assert env["Type"] == "NINA_EVENTS"
        assert env["Details"]["NextCursor"] == 2
        assert [e["Event"] for e in env["Details"]["Events"]] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_idempotent_when_no_new_events(self):
        """Polling at the current cursor returns empty events but same cursor."""
        store = EventStore()
        store.append({"Event": "A"})

        first = await tool_poll_events_since(store, cursor=None)
        assert first["Details"]["NextCursor"] == 1
        second = await tool_poll_events_since(store, cursor=first["Details"]["NextCursor"])
        assert second["Details"]["Events"] == []
        assert second["Details"]["NextCursor"] == 1

    @pytest.mark.asyncio
    async def test_max_events_threaded_through(self):
        store = EventStore()
        for i in range(20):
            store.append({"Event": f"E{i}"})
        env = await tool_poll_events_since(store, cursor=0, max_events=5)
        assert len(env["Details"]["Events"]) == 5
        assert env["Details"]["NextCursor"] == 5
