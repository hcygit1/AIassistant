from __future__ import annotations

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

from api.config_api import router
from api.dependencies import get_agent_manager, get_heartbeat_runner
from llm.models_config import ModelRef


class ConfigApiDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_manager = Mock()
        self.heartbeat_runner = Mock()
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_agent_manager] = (
            lambda: self.agent_manager
        )
        app.dependency_overrides[get_heartbeat_runner] = (
            lambda: self.heartbeat_runner
        )
        self.app = app
        self.client = TestClient(app)

    def test_current_model_uses_overridden_agent_manager(self) -> None:
        current = ModelRef(provider="fake", model="runtime")
        self.agent_manager.get_current_model_ref.return_value = current
        model_definition = SimpleNamespace(
            reasoning=True,
            input=["text"],
            context_window=64000,
            max_tokens=4096,
        )

        with (
            patch(
                "llm.model_selection.get_model_display_name",
                return_value="Runtime Model",
            ),
            patch(
                "llm.models_config.models_config.get_model",
                return_value=model_definition,
            ),
            patch(
                "llm.models_config.models_config.resolve_api_protocol",
                return_value="openai-completions",
            ),
        ):
            response = self.client.get("/api/models/current/main")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ref"], "fake/runtime")
        self.agent_manager.get_current_model_ref.assert_called_once_with(
            "main"
        )

    def test_tool_policy_uses_overridden_heartbeat_runner(self) -> None:
        config = {
            "agents": {
                "list": [
                    {
                        "id": "main",
                        "tools": {"deny": ["exec"]},
                    }
                ]
            }
        }

        with (
            patch("api.config_api.get_raw_config", return_value=config),
            patch("api.config_api.get_config", return_value=config),
            patch("api.config_api.save_config"),
            patch(
                "system_messages.heartbeat.heartbeat_runner.update_config"
            ),
        ):
            response = self.client.put(
                "/api/config/tools-policy",
                json={
                    "agent_id": "main",
                    "tool_name": "exec",
                    "allowed": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.heartbeat_runner.update_config.assert_called_once_with()

    def test_dependencies_do_not_add_openapi_parameters(self) -> None:
        paths = self.app.openapi()["paths"]

        self.assertEqual(
            {
                item["name"]
                for item in paths["/api/models/current/{agent_id}"][
                    "get"
                ].get("parameters", [])
            },
            {"agent_id"},
        )
        self.assertEqual(
            {
                item["name"]
                for item in paths["/api/models/switch/{agent_id}"][
                    "post"
                ].get("parameters", [])
            },
            {"agent_id"},
        )


if __name__ == "__main__":
    unittest.main()
