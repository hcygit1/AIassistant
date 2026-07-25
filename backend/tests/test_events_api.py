from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.dependencies import (
    get_agent_manager,
    get_event_bus,
    get_heartbeat_runner,
)
from api.events import agent_events, router


class OneEventQueue:
    def __init__(self, event: dict) -> None:
        self._event = event

    async def get(self):
        if self._event is not None:
            event = self._event
            self._event = None
            return event
        raise asyncio.CancelledError


class EventsApiDependencyTests(unittest.TestCase):
    def test_sse_uses_overridden_event_bus_and_preserves_protocol(
        self,
    ) -> None:
        event_bus = Mock()
        queue = OneEventQueue(
            {"type": "lifecycle", "message": "完成"}
        )
        event_bus.subscribe.return_value = queue
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_event_bus] = lambda: event_bus
        client = TestClient(app)

        response = client.get("/api/agents/main/events")
        operation = app.openapi()["paths"][
            "/api/agents/{agent_id}/events"
        ]["get"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.text,
            'data: {"type": "lifecycle", "message": "完成"}\n\n',
        )
        self.assertTrue(
            response.headers["content-type"].startswith(
                "text/event-stream"
            )
        )
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        event_bus.subscribe.assert_called_once_with("main")
        event_bus.unsubscribe.assert_called_once_with("main", queue)
        self.assertEqual(
            {item["name"] for item in operation.get("parameters", [])},
            {"agent_id"},
        )

    def test_status_uses_overridden_runtime_dependencies(self) -> None:
        manager = Mock()
        manager.get_state.return_value = SimpleNamespace(
            total_turns=3,
            total_input_tokens=10,
            total_output_tokens=5,
            compaction_count=1,
            thinking=True,
            verbose=False,
            reasoning=False,
            last_active=123.0,
        )
        heartbeat = Mock(active_agents={"main"})
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_agent_manager] = lambda: manager
        app.dependency_overrides[get_heartbeat_runner] = lambda: heartbeat
        client = TestClient(app)

        response = client.get("/api/agents/main/status")
        operation = app.openapi()["paths"][
            "/api/agents/{agent_id}/status"
        ]["get"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_turns"], 3)
        self.assertEqual(response.json()["heartbeat_active"], True)
        manager.get_state.assert_called_once_with("main")
        self.assertEqual(
            {item["name"] for item in operation.get("parameters", [])},
            {"agent_id"},
        )


class EventsStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_emits_keepalive_and_unsubscribes_on_close(
        self,
    ) -> None:
        event_bus = Mock()
        queue: asyncio.Queue = asyncio.Queue()
        event_bus.subscribe.return_value = queue

        async def timeout(awaitable, *, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        with patch("api.events.asyncio.wait_for", new=timeout):
            response = await agent_events(
                "main",
                event_bus=event_bus,
            )
            chunk = await anext(response.body_iterator)
            await response.body_iterator.aclose()

        self.assertEqual(chunk, ": keepalive\n\n")
        event_bus.unsubscribe.assert_called_once_with("main", queue)


if __name__ == "__main__":
    unittest.main()
