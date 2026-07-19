from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app
from runtime.agent import agent_manager
from system_messages.heartbeat import heartbeat_runner


class AppCorsSettingsTests(unittest.TestCase):
    def test_cors_settings_include_optional_origin_regex(self) -> None:
        resolver = getattr(backend_app, "resolve_cors_settings", None)
        self.assertIsNotNone(resolver, "resolve_cors_settings should be defined")

        origins, origin_regex = resolver(
            {
                "PIPIXIA_CORS_ORIGINS": (
                    "https://app.example.test, http://localhost:4100"
                ),
                "PIPIXIA_CORS_ORIGIN_REGEX": r"  ^https?://[^/]+:4100$  ",
            }
        )

        self.assertEqual(
            origins,
            ["https://app.example.test", "http://localhost:4100"],
        )
        self.assertEqual(origin_regex, r"^https?://[^/]+:4100$")


class AppLifespanTests(unittest.IsolatedAsyncioTestCase):
    def _patch_lifespan_dependencies(
        self,
    ) -> tuple[
        ExitStack,
        AsyncMock,
        AsyncMock,
        AsyncMock,
        AsyncMock,
        AsyncMock,
        Mock,
    ]:
        stack = ExitStack()
        initialize_mock = AsyncMock()
        close_mock = AsyncMock()
        wait_mock = AsyncMock()
        heartbeat_start_mock = AsyncMock()
        heartbeat_stop_mock = AsyncMock()
        fail_unrecoverable_mock = Mock(return_value=0)

        for target in (
            patch.object(backend_app, "load_config"),
            patch.object(backend_app, "_setup_logging_from_config"),
            patch.object(backend_app, "scan_all_skills"),
            patch.object(backend_app, "list_agents", return_value=[]),
            patch.object(backend_app.skills_watcher, "start"),
            patch.object(backend_app.skills_watcher, "stop"),
            patch.object(agent_manager, "initialize", new=initialize_mock),
            patch.object(agent_manager, "close", new=close_mock),
            patch.object(
                heartbeat_runner,
                "start",
                new=heartbeat_start_mock,
            ),
            patch.object(
                heartbeat_runner,
                "stop",
                new=heartbeat_stop_mock,
            ),
            patch(
                "sessions.session_work_delivery.session_work_delivery.fail_unrecoverable_pending",
                new=fail_unrecoverable_mock,
            ),
            patch(
                "sessions.session_work_delivery.session_work_delivery.recover_pending_work",
                return_value=0,
            ),
            patch("subagents.subagent_archive.start_subagent_archive"),
            patch("subagents.subagent_archive.stop_subagent_archive"),
            patch("subagents.subagent_resume.resume_subagent_runs", new=AsyncMock()),
            patch("config.get_config", return_value={"cron": {"enabled": False}}),
            patch.object(
                agent_manager,
                "wait_for_pending_tasks",
                new=wait_mock,
            ),
        ):
            stack.enter_context(target)

        return (
            stack,
            initialize_mock,
            close_mock,
            wait_mock,
            heartbeat_start_mock,
            heartbeat_stop_mock,
            fail_unrecoverable_mock,
        )

    async def test_lifespan_closes_agent_manager_on_shutdown(self) -> None:
        application = FastAPI()
        stack, initialize_mock, close_mock, wait_mock, _, _, fail_unrecoverable_mock = (
            self._patch_lifespan_dependencies()
        )

        with stack:
            async with backend_app.lifespan(application):
                pass

        initialize_mock.assert_awaited_once()
        fail_unrecoverable_mock.assert_called_once_with()
        close_mock.assert_awaited_once_with(timeout=30)
        wait_mock.assert_not_awaited()

    async def test_lifespan_closes_agent_manager_when_application_raises(self) -> None:
        application = FastAPI()
        stack, _, close_mock, wait_mock, _, _, _ = self._patch_lifespan_dependencies()

        with stack:
            with self.assertRaisesRegex(RuntimeError, "application failed"):
                async with backend_app.lifespan(application):
                    raise RuntimeError("application failed")

        close_mock.assert_awaited_once_with(timeout=30)
        wait_mock.assert_not_awaited()

    async def test_lifespan_stops_heartbeat_before_closing_agent_manager(self) -> None:
        application = FastAPI()
        stack, _, close_mock, _, _, heartbeat_stop_mock, _ = (
            self._patch_lifespan_dependencies()
        )
        shutdown_order: list[str] = []

        async def _record_heartbeat_stop() -> None:
            shutdown_order.append("heartbeat")

        async def _record_agent_close(*, timeout: float) -> None:
            self.assertEqual(timeout, 30)
            shutdown_order.append("agent")

        heartbeat_stop_mock.side_effect = _record_heartbeat_stop
        close_mock.side_effect = _record_agent_close

        with stack:
            async with backend_app.lifespan(application):
                pass

        self.assertEqual(shutdown_order, ["heartbeat", "agent"])

    async def test_lifespan_cleans_up_when_startup_fails(self) -> None:
        application = FastAPI()
        (
            stack,
            initialize_mock,
            close_mock,
            _,
            heartbeat_start_mock,
            heartbeat_stop_mock,
            _,
        ) = self._patch_lifespan_dependencies()
        heartbeat_start_mock.side_effect = RuntimeError("heartbeat failed")

        with stack:
            with self.assertRaisesRegex(RuntimeError, "heartbeat failed"):
                async with backend_app.lifespan(application):
                    self.fail("application should not start")

        initialize_mock.assert_awaited_once()
        heartbeat_stop_mock.assert_awaited_once()
        close_mock.assert_awaited_once_with(timeout=30)


if __name__ == "__main__":
    unittest.main()
