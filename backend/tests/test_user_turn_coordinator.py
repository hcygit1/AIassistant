from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from turns.coordinator import user_turn_coordinator


class UserTurnCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        user_turn_coordinator._runtimes.clear()
        user_turn_coordinator._session_to_turn.clear()

    async def test_create_queued_registers_runtime_and_session_mapping(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")

        self.assertEqual(runtime.status, "queued")
        self.assertTrue(user_turn_coordinator.has_active_user_turn("main", "main-main"))
        self.assertEqual(
            user_turn_coordinator.current_turn_id_for_session("main", "main-main"),
            runtime.turn_id,
        )

    async def test_create_queued_rejects_second_active_turn(self) -> None:
        user_turn_coordinator.create_queued("main", "main-main")

        with self.assertRaises(RuntimeError):
            user_turn_coordinator.create_queued("main", "main-main")

    async def test_set_running_and_get_pending_for_session(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")

        user_turn_coordinator.set_running(runtime.turn_id)
        pending = user_turn_coordinator.get_pending_for_session("main", "main-main")

        self.assertIsNotNone(pending)
        self.assertEqual(pending.status, "running")

    async def test_abort_turn_cancels_bound_task_and_records_reason(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        user_turn_coordinator.set_running(runtime.turn_id)

        async def _never_finishes() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_never_finishes())
        user_turn_coordinator.bind_execution_task(runtime.turn_id, task)

        aborted = user_turn_coordinator.abort_turn("main", "main-main", turn_id=runtime.turn_id)

        self.assertTrue(aborted)
        self.assertEqual(
            user_turn_coordinator.get_cancel_reason(runtime.turn_id),
            "stopped_by_user",
        )
        self.assertTrue(task.cancelled() or task.cancelling() > 0)

        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_abort_turn_uses_client_disconnected_reason(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        user_turn_coordinator.set_running(runtime.turn_id)

        async def _never_finishes() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_never_finishes())
        user_turn_coordinator.bind_execution_task(runtime.turn_id, task)

        aborted = user_turn_coordinator.abort_turn(
            "main",
            "main-main",
            turn_id=runtime.turn_id,
            user_initiated=False,
        )

        self.assertTrue(aborted)
        self.assertEqual(
            user_turn_coordinator.get_cancel_reason(runtime.turn_id),
            "client_disconnected",
        )

        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_terminal_transition_purges_turn_from_coordinator(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")

        user_turn_coordinator.set_done(runtime.turn_id)

        self.assertIsNone(user_turn_coordinator.get(runtime.turn_id))
        self.assertFalse(user_turn_coordinator.has_active_user_turn("main", "main-main"))

    async def test_clear_session_removes_runtime(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")

        user_turn_coordinator.clear_session("main", "main-main")

        self.assertIsNone(user_turn_coordinator.get(runtime.turn_id))
        self.assertIsNone(
            user_turn_coordinator.current_turn_id_for_session("main", "main-main")
        )


if __name__ == "__main__":
    unittest.main()
