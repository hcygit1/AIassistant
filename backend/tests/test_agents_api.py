from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.agents import router
from api.dependencies import get_agent_manager, get_heartbeat_runner


class AgentsApiDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_manager = Mock()
        self.agent_manager.register_agent = AsyncMock()
        self.heartbeat_runner = Mock()
        self.heartbeat_runner.add_agent = AsyncMock()
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_agent_manager] = lambda: self.agent_manager
        app.dependency_overrides[get_heartbeat_runner] = (
            lambda: self.heartbeat_runner
        )
        self.app = app
        self.client = TestClient(app)

    def test_create_uses_overridden_runtime_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_dir = Path(tmpdir) / "worker"

            def ensure_workspace(*_args, **_kwargs):
                (agent_dir / "workspace").mkdir(parents=True)

            with (
                patch("api.agents.list_agents", return_value=[]),
                patch("api.agents.resolve_agent_dir", return_value=agent_dir),
                patch("runtime.workspace.ensure_agent_workspace", ensure_workspace),
                patch("api.agents.add_agent_to_config") as add_config,
                patch("tools.skills_scanner.write_skills_snapshot") as snapshot,
            ):
                response = self.client.post(
                    "/api/agents",
                    json={
                        "id": "worker",
                        "name": "Worker",
                        "description": "Background work",
                    },
                )

        self.assertEqual(response.status_code, 200)
        add_config.assert_called_once()
        snapshot.assert_called_once_with("worker")
        self.agent_manager.register_agent.assert_awaited_once_with("worker")
        self.heartbeat_runner.add_agent.assert_awaited_once_with("worker")

    def test_dependencies_do_not_add_openapi_parameters(self) -> None:
        operation = self.app.openapi()["paths"]["/api/agents"]["post"]

        self.assertEqual(operation.get("parameters", []), [])

    def test_delete_uses_overridden_heartbeat_runner(self) -> None:
        config = {
            "agents": {
                "list": [{"id": "main"}, {"id": "worker"}],
            }
        }
        with (
            patch("api.agents.get_raw_config", return_value=config),
            patch("api.agents.save_config") as save_config,
        ):
            response = self.client.delete(
                "/api/agents/worker?delete_files=false"
            )

        self.assertEqual(response.status_code, 200)
        save_config.assert_called_once()
        self.heartbeat_runner.update_config.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
