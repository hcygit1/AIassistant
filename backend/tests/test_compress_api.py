from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.compress import router
from api.dependencies import get_agent_manager, get_session_manager


class CompressApiDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_manager = Mock()
        self.session_manager.load_session.return_value = {
            "messages": [{"role": "user", "content": str(index)} for index in range(4)]
        }
        self.agent_manager = Mock()
        self.agent_manager.compress_session = AsyncMock(
            return_value={"archived_count": 2},
        )
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_session_manager] = (
            lambda: self.session_manager
        )
        app.dependency_overrides[get_agent_manager] = lambda: self.agent_manager
        self.app = app
        self.client = TestClient(app)

    def _compress(self):
        return self.client.post(
            "/api/agents/main/sessions/main-main/compress"
        )

    def test_success_uses_overridden_managers(self) -> None:
        response = self._compress()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"archived_count": 2})
        self.agent_manager.compress_session.assert_awaited_once_with(
            "main-main",
            "main",
        )

    def test_missing_session_returns_404(self) -> None:
        self.session_manager.load_session.return_value = None

        response = self._compress()

        self.assertEqual(response.status_code, 404)
        self.agent_manager.compress_session.assert_not_awaited()

    def test_short_session_returns_400(self) -> None:
        self.session_manager.load_session.return_value = {"messages": []}

        response = self._compress()

        self.assertEqual(response.status_code, 400)
        self.agent_manager.compress_session.assert_not_awaited()

    def test_agent_failure_returns_500(self) -> None:
        self.agent_manager.compress_session.side_effect = RuntimeError("failed")

        response = self._compress()

        self.assertEqual(response.status_code, 500)
        self.assertIn("failed", response.json()["detail"])

    def test_business_error_returns_400(self) -> None:
        self.agent_manager.compress_session.return_value = {
            "error": "nothing to compact"
        }

        response = self._compress()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "nothing to compact")

    def test_dependencies_do_not_add_openapi_parameters(self) -> None:
        operation = self.app.openapi()["paths"][
            "/api/agents/{agent_id}/sessions/{session_id}/compress"
        ]["post"]
        parameter_names = {
            parameter["name"] for parameter in operation.get("parameters", [])
        }

        self.assertEqual(parameter_names, {"agent_id", "session_id"})


if __name__ == "__main__":
    unittest.main()
