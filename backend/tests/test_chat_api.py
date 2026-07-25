from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.chat import router as chat_router
from api.dependencies import (
    get_session_manager,
    get_user_turn_service,
)
from turns.events import TurnEvent


class ChatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(chat_router, prefix="/api")
        self.app = app
        self.client = TestClient(app)

    def test_submit_uses_overridden_runtime_dependencies(self) -> None:
        manager = Mock()
        manager.resolve_main_session_id.return_value = "override-main"
        service = Mock()
        service.submit = AsyncMock(
            return_value={
                "turn_id": "turn-override",
                "position": 1,
                "status": "queued",
                "session_id": "override-main",
            }
        )
        self.app.dependency_overrides[get_session_manager] = lambda: manager
        self.app.dependency_overrides[get_user_turn_service] = lambda: service

        response = self.client.post(
            "/api/chat/submit",
            json={"message": "hello", "agent_id": "main"},
        )

        self.assertEqual(response.status_code, 202)
        service.submit.assert_awaited_once_with(
            "hello",
            "main",
            "override-main",
        )

    def test_chat_dependencies_do_not_add_openapi_parameters(self) -> None:
        operation = self.app.openapi()["paths"][
            "/api/chat/pending-turn"
        ]["get"]
        parameter_names = {
            parameter["name"] for parameter in operation.get("parameters", [])
        }

        self.assertEqual(parameter_names, {"session_id", "agent_id"})

    def test_submit_resolves_main_session_and_returns_202(self) -> None:
        submit_mock = AsyncMock(
            return_value={
                "turn_id": "turn-1",
                "position": 1,
                "status": "queued",
                "session_id": "main-main",
            }
        )
        with (
            patch("sessions.session_manager.session_manager.resolve_main_session_id", return_value="main-main"),
            patch("turns.service.user_turn_service.submit", submit_mock),
        ):
            response = self.client.post(
                "/api/chat/submit",
                json={"message": "hello", "agent_id": "main", "session_id": ""},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["turn_id"], "turn-1")
        submit_mock.assert_awaited_once_with("hello", "main", "main-main")

    def test_pending_turn_uses_resolved_main_session(self) -> None:
        pending_mock = AsyncMock(
            return_value={
                "turn_id": "turn-2",
                "status": "queued",
                "position": 2,
                "session_id": "main-main",
                "agent_id": "main",
            }
        )
        with (
            patch("sessions.session_manager.session_manager.resolve_main_session_id", return_value="main-main"),
            patch("turns.service.user_turn_service.pending", pending_mock),
        ):
            response = self.client.get("/api/chat/pending-turn?agent_id=main&session_id=")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["turn_id"], "turn-2")
        pending_mock.assert_awaited_once_with("main", "main-main")

    def test_abort_passes_turn_id_and_user_flag_to_service(self) -> None:
        abort_mock = AsyncMock(return_value={"aborted": True})
        with (
            patch("sessions.session_manager.session_manager.resolve_main_session_id", return_value="main-main"),
            patch("turns.service.user_turn_service.abort", abort_mock),
        ):
            response = self.client.post(
                "/api/chat/abort",
                json={
                    "agent_id": "main",
                    "session_id": "",
                    "turn_id": "turn-3",
                    "user_initiated": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"aborted": True})
        abort_mock.assert_awaited_once_with(
            "main",
            "main-main",
            turn_id="turn-3",
            user_initiated=False,
        )

    def test_turn_status_proxies_to_service(self) -> None:
        status_mock = AsyncMock(
            return_value={
                "turn_id": "turn-4",
                "status": "running",
                "position": 0,
                "session_id": "main-main",
                "agent_id": "main",
                "error": None,
            }
        )
        with patch("turns.service.user_turn_service.status", status_mock):
            response = self.client.get("/api/chat/turn/turn-4/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")
        status_mock.assert_awaited_once_with("turn-4")

    def test_turn_stream_encodes_structured_events_as_sse(self) -> None:
        async def fake_stream(turn_id: str):
            self.assertEqual(turn_id, "turn-5")
            yield TurnEvent.from_payload(
                {
                    "type": "token",
                    "content": "你好",
                }
            )

        with patch("turns.service.user_turn_service.stream", fake_stream):
            response = self.client.get("/api/chat/turn/turn-5/stream")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.text,
            'event: token\ndata: {"type": "token", "content": "你好"}\n\n',
        )


if __name__ == "__main__":
    unittest.main()
