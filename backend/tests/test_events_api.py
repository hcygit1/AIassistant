from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.dependencies import get_agent_manager, get_heartbeat_runner
from api.events import router


class EventsApiDependencyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
