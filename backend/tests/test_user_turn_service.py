from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from turns.coordinator import user_turn_coordinator
from turns.events import TurnEvent
from turns.service import UserTurnService, user_turn_service
from sessions.session_work_runtime import session_work_runtime


class _FakeLock:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()


class _FakeDispatcher:
    def __init__(self, position: int = 1) -> None:
        self.position = position
        self.submitted = []

    def submit(self, task) -> int:
        self.submitted.append(task)
        return len(self.submitted)

    def turn_queue_position(self, turn_id: str) -> int | None:
        return self.position


class UserTurnServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        user_turn_coordinator._runtimes.clear()
        user_turn_coordinator._session_to_turn.clear()
        user_turn_coordinator._terminal_turns.clear()

    async def test_submit_uses_injected_turn_dependencies(self) -> None:
        runtime = SimpleNamespace(
            turn_id="turn-injected",
            stream_queue=asyncio.Queue(),
        )
        coordinator = Mock()
        coordinator.has_active_user_turn.return_value = False
        coordinator.create_queued.return_value = runtime
        lock = _FakeLock()
        lock_manager = Mock()
        lock_manager.get_lock.return_value = lock
        dispatcher = _FakeDispatcher(position=2)
        dispatcher_manager = Mock()
        dispatcher_manager.get.return_value = dispatcher
        service = UserTurnService(
            coordinator=coordinator,
            lock_manager=lock_manager,
            dispatcher_manager=dispatcher_manager,
        )

        result = await service.submit(" hello ", "main", "main-main")

        self.assertEqual(result["turn_id"], "turn-injected")
        coordinator.create_queued.assert_called_once_with("main", "main-main")
        lock_manager.get_lock.assert_called_once_with("main", "main-main")
        dispatcher_manager.get.assert_called_once_with(
            "main",
            "main-main",
            lock.lock,
        )

    async def test_submit_uses_injected_session_work_runtime(self) -> None:
        turn_runtime = SimpleNamespace(
            turn_id="turn-runtime",
            stream_queue=asyncio.Queue(),
        )
        coordinator = Mock()
        coordinator.has_active_user_turn.return_value = False
        coordinator.create_queued.return_value = turn_runtime
        lock = _FakeLock()
        lock_manager = Mock()
        lock_manager.get_lock.return_value = lock
        dispatcher = _FakeDispatcher(position=2)
        dispatcher_manager = Mock()
        dispatcher_manager.get.return_value = dispatcher
        work_runtime = SimpleNamespace(
            lock_manager=lock_manager,
            dispatcher_manager=dispatcher_manager,
        )
        service = UserTurnService(
            coordinator=coordinator,
            runtime=work_runtime,
        )

        result = await service.submit("hello", "main", "main-main")

        self.assertEqual(result["turn_id"], "turn-runtime")
        self.assertIs(service.runtime, work_runtime)
        lock_manager.get_lock.assert_called_once_with("main", "main-main")
        dispatcher_manager.get.assert_called_once_with(
            "main",
            "main-main",
            lock.lock,
        )

    async def test_rejects_runtime_mixed_with_legacy_dependencies(self) -> None:
        work_runtime = SimpleNamespace(
            lock_manager=Mock(),
            dispatcher_manager=Mock(),
        )

        with self.assertRaisesRegex(ValueError, "runtime cannot be combined"):
            UserTurnService(
                runtime=work_runtime,
                lock_manager=Mock(),
            )

    async def test_default_service_uses_shared_session_work_runtime(self) -> None:
        self.assertIs(user_turn_service.runtime, session_work_runtime)

    async def test_production_service_explicitly_binds_shared_runtime(self) -> None:
        self.assertIs(user_turn_service._runtime, session_work_runtime)

    async def test_submit_inherits_lock_from_dispatcher_manager(self) -> None:
        runtime = SimpleNamespace(
            turn_id="turn-inherited-lock",
            stream_queue=asyncio.Queue(),
        )
        coordinator = Mock()
        coordinator.has_active_user_turn.return_value = False
        coordinator.create_queued.return_value = runtime
        lock = _FakeLock()
        lock_manager = Mock()
        lock_manager.get_lock.return_value = lock
        dispatcher = _FakeDispatcher(position=1)
        dispatcher_manager = SimpleNamespace(
            lock_manager=lock_manager,
            get=Mock(return_value=dispatcher),
        )
        service = UserTurnService(
            coordinator=coordinator,
            dispatcher_manager=dispatcher_manager,
        )

        await service.submit("hello", "main", "main-main")

        lock_manager.get_lock.assert_called_once_with("main", "main-main")
        dispatcher_manager.get.assert_called_once_with(
            "main",
            "main-main",
            lock.lock,
        )

    async def test_submit_creates_turn_and_enqueues_user_work_item(self) -> None:
        dispatcher = _FakeDispatcher(position=1)
        with (
            patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
            patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
        ):
            result = await user_turn_service.submit(" hello ", "main", "main-main")

        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["position"], 1)
        self.assertEqual(len(dispatcher.submitted), 1)
        work_item = dispatcher.submitted[0]
        self.assertEqual(work_item.kind, "user")
        self.assertEqual(work_item.content, "hello")
        self.assertEqual(work_item.turn_id, result["turn_id"])
        runtime = user_turn_coordinator.get(result["turn_id"])
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.status, "queued")

    async def test_submit_rejects_second_active_turn_in_same_session(self) -> None:
        dispatcher = _FakeDispatcher(position=1)
        with (
            patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
            patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
        ):
            await user_turn_service.submit("first", "main", "main-main")
            with self.assertRaises(HTTPException) as ctx:
                await user_turn_service.submit("second", "main", "main-main")

        self.assertEqual(ctx.exception.status_code, 409)

    async def test_status_returns_queued_position(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        dispatcher = _FakeDispatcher(position=3)
        with (
            patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
            patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
        ):
            result = await user_turn_service.status(runtime.turn_id)

        self.assertEqual(result["turn_id"], runtime.turn_id)
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["position"], 3)

    async def test_status_returns_retained_terminal_error(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        user_turn_coordinator.set_error(runtime.turn_id, "model failed")

        result = await user_turn_service.status(runtime.turn_id)

        self.assertEqual(result["turn_id"], runtime.turn_id)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["position"], 0)
        self.assertEqual(result["error"], "model failed")

    async def test_pending_returns_active_turn_for_session(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        dispatcher = _FakeDispatcher(position=2)
        with (
            patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
            patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
        ):
            result = await user_turn_service.pending("main", "main-main")

        self.assertEqual(result["turn_id"], runtime.turn_id)
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["position"], 2)

    async def test_stream_yields_runtime_queue_items_for_running_turn(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        user_turn_coordinator.set_running(runtime.turn_id)
        event = TurnEvent.from_payload({"type": "token", "content": "hello"})
        await runtime.stream_queue.put(event)
        await runtime.stream_queue.put(None)

        items = []
        async for item in user_turn_service.stream(runtime.turn_id):
            items.append(item)

        self.assertEqual(items, [event])

    async def test_stream_allows_queued_turn_and_waits_for_events(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")

        async def _produce() -> None:
            await asyncio.sleep(0)
            await runtime.stream_queue.put(
                TurnEvent.from_payload({"type": "token", "content": "hello"})
            )
            await runtime.stream_queue.put(None)

        producer = asyncio.create_task(_produce())
        items = []
        async for item in user_turn_service.stream(runtime.turn_id):
            items.append(item)
        await producer

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TurnEvent)
        self.assertEqual(items[0].payload["content"], "hello")

    async def test_stream_ends_cleanly_if_turn_finished_before_subscription(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        user_turn_coordinator.set_done(runtime.turn_id)

        items = []
        async for item in user_turn_service.stream(runtime.turn_id):
            items.append(item)

        self.assertEqual(items, [])

    async def test_stream_replays_terminal_error_if_turn_failed_before_subscription(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        user_turn_coordinator.set_error(runtime.turn_id, "model failed")

        items = []
        async for item in user_turn_service.stream(runtime.turn_id):
            items.append(item)

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], TurnEvent)
        self.assertEqual(items[0].type, "error")
        self.assertEqual(items[0].error_message, "model failed")

    async def test_abort_cancels_bound_execution_task(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        user_turn_coordinator.set_running(runtime.turn_id)

        async def _never_finishes() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_never_finishes())
        user_turn_coordinator.bind_execution_task(runtime.turn_id, task)

        result = await user_turn_service.abort("main", "main-main", turn_id=runtime.turn_id)

        self.assertTrue(result["aborted"])
        self.assertTrue(task.cancelled() or task.cancelling() > 0)
        self.assertEqual(
            user_turn_coordinator.get_cancel_reason(runtime.turn_id),
            "stopped_by_user",
        )

        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    unittest.main()
