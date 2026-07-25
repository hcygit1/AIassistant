from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class SessionsApiDependencyTests(unittest.TestCase):
    def test_default_provider_resolves_compatibility_global_lazily(self) -> None:
        from api.dependencies import get_session_manager

        replacement = Mock()
        with patch(
            "sessions.session_manager.session_manager",
            replacement,
        ):
            resolved = get_session_manager()

        self.assertIs(resolved, replacement)

    def setUp(self) -> None:
        from api.dependencies import get_session_manager
        from api.sessions import router

        self.manager = Mock()
        self.manager.resolve_main_session_id.return_value = "main-main"
        self.manager.load_session.return_value = {
            "created_at": 1.0,
            "updated_at": 2.0,
            "messages": [],
        }
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_session_manager] = lambda: self.manager
        self.app = app
        self.client = TestClient(app)

    def test_main_session_uses_overridden_session_manager(self) -> None:
        response = self.client.get("/api/agents/main/session")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "main-main")
        self.manager.load_session.assert_called_once_with(
            "main-main",
            "main",
        )

    def test_dependency_does_not_add_openapi_query_parameter(self) -> None:
        operation = self.app.openapi()["paths"][
            "/api/agents/{agent_id}/session"
        ]["get"]
        parameter_names = {
            parameter["name"] for parameter in operation.get("parameters", [])
        }

        self.assertEqual(parameter_names, {"agent_id"})


if __name__ == "__main__":
    unittest.main()
