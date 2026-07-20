from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from system_messages.heartbeat import HeartbeatRunner


class HeartbeatRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_waits_for_cancelled_tasks_to_finish(self) -> None:
        runner = HeartbeatRunner()
        runner._running = True
        cleanup_finished = asyncio.Event()

        async def _task_with_async_cleanup() -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                cleanup_finished.set()
                raise

        task = asyncio.create_task(_task_with_async_cleanup())
        await asyncio.sleep(0)
        runner._tasks["main"] = task

        await runner.stop()

        self.assertTrue(cleanup_finished.is_set())
        self.assertTrue(task.done())
        self.assertEqual(runner.active_agents, [])

    async def test_run_heartbeat_uses_injected_delivery(self) -> None:
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"
        work_delivery = Mock()
        runner = HeartbeatRunner(
            session_manager=session_manager,
            work_delivery=work_delivery,
        )

        with (
            patch(
                "system_messages.heartbeat.get_heartbeat_config",
                return_value={"enabled": True, "prompt": "check status"},
            ),
            patch(
                "system_messages.heartbeat.resolve_agent_workspace",
                return_value=Path("/missing-workspace"),
            ),
            patch(
                "system_messages.heartbeat.is_within_active_hours",
                return_value=True,
            ),
            patch(
                "config.resolve_agent_config",
                return_value={"user_timezone": "UTC"},
            ),
        ):
            await runner._run_heartbeat("main")

        session_manager.resolve_main_session_id.assert_called_once_with("main")
        work_delivery.deliver.assert_called_once()
        self.assertEqual(work_delivery.deliver.call_args.kwargs["kind"], "heartbeat")
        self.assertEqual(
            work_delivery.deliver.call_args.kwargs["session_id"],
            "main-main",
        )

    async def test_execution_failure_is_recorded_as_failed(self) -> None:
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"
        work_delivery = Mock()
        runner = HeartbeatRunner(
            session_manager=session_manager,
            work_delivery=work_delivery,
        )

        with (
            patch(
                "system_messages.heartbeat.get_heartbeat_config",
                return_value={"enabled": True, "prompt": "check status"},
            ),
            patch(
                "system_messages.heartbeat.resolve_agent_workspace",
                return_value=Path("/missing-workspace"),
            ),
            patch(
                "system_messages.heartbeat.is_within_active_hours",
                return_value=True,
            ),
            patch(
                "config.resolve_agent_config",
                return_value={"user_timezone": "UTC"},
            ),
            patch("system_messages.heartbeat.emit_heartbeat_event") as emit,
            patch("system_messages.heartbeat.audit_logger.log") as audit,
        ):
            await runner._run_heartbeat("main")
            await work_delivery.deliver.call_args.kwargs["on_failure_async"](
                RuntimeError("model failed")
            )

        event = emit.call_args.args[1]
        self.assertEqual(event.status, "failed")
        self.assertEqual(event.reason, "model failed")
        audit.assert_called_with("main", "heartbeat_failed", {"error": "model failed"})


if __name__ == "__main__":
    unittest.main()
