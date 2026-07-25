from __future__ import annotations

import asyncio
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from system_messages.heartbeat import HeartbeatRunner
from system_messages.heartbeat_run_lifecycle import HeartbeatRunLifecycle


class HeartbeatRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_handles_ack_and_timeout_outcomes(self) -> None:
        rollback = Mock()
        events = []
        audit = Mock()
        webchat = Mock()
        lifecycle = HeartbeatRunLifecycle(
            rollback_last_turn=rollback,
            event_sink=lambda agent_id, event: events.append((agent_id, event)),
            audit_event=audit,
            emit_webchat_message=webchat,
            now=lambda: 2.0,
        )
        callbacks = lifecycle.callbacks(
            agent_id="main",
            session_id="main-main",
            config={"ackMaxChars": 300},
            started_at=1.0,
        )

        await callbacks["result_handler"]("HEARTBEAT_OK")
        await callbacks["on_failure_async"](asyncio.TimeoutError())

        rollback.assert_called_once_with("main-main", "main")
        self.assertEqual(
            [event.status for _, event in events],
            ["ok-token", "skipped"],
        )
        self.assertEqual(events[1][1].reason, "session-busy")
        webchat.assert_not_called()
        audit.assert_any_call("main", "heartbeat_ok", {})
        audit.assert_any_call(
            "main",
            "heartbeat_skipped",
            {"reason": "session-busy"},
        )

    async def test_lifecycle_emits_webchat_response(self) -> None:
        events = []
        audit = Mock()
        webchat = Mock()
        lifecycle = HeartbeatRunLifecycle(
            rollback_last_turn=Mock(),
            event_sink=lambda agent_id, event: events.append((agent_id, event)),
            audit_event=audit,
            emit_webchat_message=webchat,
            now=lambda: 2.0,
        )
        callbacks = lifecycle.callbacks(
            agent_id="main",
            session_id="main-main",
            config={"target": "webchat"},
            started_at=1.0,
        )

        await callbacks["result_handler"]("needs attention")

        self.assertEqual(events[0][1].status, "sent")
        self.assertEqual(events[0][1].preview, "needs attention")
        webchat.assert_called_once_with("main", "main-main")
        audit.assert_called_once_with(
            "main",
            "heartbeat_response",
            {"response": "needs attention"},
        )

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

    async def test_run_heartbeat_uses_injected_lifecycle_callbacks(self) -> None:
        self.assertIn(
            "run_lifecycle",
            inspect.signature(HeartbeatRunner).parameters,
        )
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"
        work_delivery = Mock()
        result_handler = AsyncMock()
        failure_handler = AsyncMock()
        lifecycle = Mock()
        lifecycle.callbacks.return_value = {
            "result_handler": result_handler,
            "on_failure_async": failure_handler,
        }
        runner = HeartbeatRunner(
            session_manager=session_manager,
            work_delivery=work_delivery,
            run_lifecycle=lifecycle,
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

        lifecycle.callbacks.assert_called_once()
        delivery_options = work_delivery.deliver.call_args.kwargs
        self.assertIs(delivery_options["result_handler"], result_handler)
        self.assertIs(delivery_options["on_failure_async"], failure_handler)

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

    async def test_run_heartbeat_uses_injected_event_sink(self) -> None:
        session_manager = Mock()
        session_manager.resolve_main_session_id.return_value = "main-main"
        work_delivery = Mock()
        events = []
        runner = HeartbeatRunner(
            session_manager=session_manager,
            work_delivery=work_delivery,
            event_sink=lambda agent_id, event: events.append((agent_id, event)),
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
            await work_delivery.deliver.call_args.kwargs["on_failure_async"](
                RuntimeError("injected failure")
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "main")
        self.assertEqual(events[0][1].status, "failed")


if __name__ == "__main__":
    unittest.main()
