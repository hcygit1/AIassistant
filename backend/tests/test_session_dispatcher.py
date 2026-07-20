from __future__ import annotations

import asyncio
import gc
import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_dispatcher import (
    DispatcherManager,
    PRIORITY_ANNOUNCE,
    PRIORITY_CRON,
    PRIORITY_HEARTBEAT,
    PRIORITY_USER,
    SessionDispatcher,
    SessionWorkItem,
)
from turns.events import TurnEvent
from turns.coordinator import user_turn_coordinator


class _RecordingDispatcher(SessionDispatcher):
    def __init__(self, expected_count: int):
        super().__init__(lock=asyncio.Lock())
        self.expected_count = expected_count
        self.done_event = asyncio.Event()
        self.started: list[str] = []
        self.finished: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def _execute(self, task: SessionWorkItem) -> None:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.started.append(task.kind)
        await asyncio.sleep(0.01)
        self.finished.append(task.kind)
        self.concurrent -= 1
        if len(self.finished) >= self.expected_count:
            self.done_event.set()


class SessionDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_work_removes_queued_item_and_calls_hook(self) -> None:
        cancelled: list[str] = []
        dispatcher = SessionDispatcher(lock=asyncio.Lock())
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="cron",
                agent_id="main",
                session_id="main-main",
                work_id="work-1",
                on_cancel=lambda: cancelled.append("work-1"),
            )
        )

        removed = dispatcher.cancel_work("work-1")

        self.assertTrue(removed)
        self.assertEqual(dispatcher.pending_count, 0)
        self.assertEqual(cancelled, ["work-1"])

    async def test_user_work_item_has_highest_priority(self) -> None:
        dispatcher = _RecordingDispatcher(expected_count=3)
        dispatcher.start()

        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="cron",
                agent_id="main",
                session_id="main-main",
            )
        )
        dispatcher.submit(
            SessionWorkItem(
                kind="announce",
                priority=PRIORITY_ANNOUNCE,
                content="announce",
                agent_id="main",
                session_id="main-main",
            )
        )
        dispatcher.submit(
            SessionWorkItem(
                kind="user",
                priority=PRIORITY_USER,
                content="user",
                agent_id="main",
                session_id="main-main",
            )
        )

        await asyncio.wait_for(dispatcher.done_event.wait(), timeout=1)
        dispatcher.stop()

        self.assertEqual(dispatcher.started, ["user", "announce", "cron"])

    async def test_dispatcher_executes_work_items_serially(self) -> None:
        dispatcher = _RecordingDispatcher(expected_count=3)
        dispatcher.start()

        dispatcher.submit(
            SessionWorkItem(
                kind="announce",
                priority=PRIORITY_ANNOUNCE,
                content="a1",
                agent_id="main",
                session_id="main-main",
            )
        )
        dispatcher.submit(
            SessionWorkItem(
                kind="heartbeat",
                priority=PRIORITY_HEARTBEAT,
                content="h1",
                agent_id="main",
                session_id="main-main",
            )
        )
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="c1",
                agent_id="main",
                session_id="main-main",
            )
        )

        await asyncio.wait_for(dispatcher.done_event.wait(), timeout=1)
        dispatcher.stop()

        self.assertEqual(dispatcher.max_concurrent, 1)
        self.assertEqual(dispatcher.started, ["announce", "cron", "heartbeat"])
        self.assertEqual(dispatcher.finished, ["announce", "cron", "heartbeat"])

    async def test_idle_dispatcher_releases_last_work_item_and_stream_queue(self) -> None:
        dispatcher = _RecordingDispatcher(expected_count=1)
        dispatcher.start()
        stream_queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        work_item = SessionWorkItem(
            kind="user",
            priority=PRIORITY_USER,
            content="user",
            agent_id="main",
            session_id="main-main",
            turn_id="turn-1",
            stream_queue=stream_queue,
        )
        work_item_ref = weakref.ref(work_item)
        stream_queue_ref = weakref.ref(stream_queue)

        try:
            dispatcher.submit(work_item)
            del work_item
            del stream_queue
            await asyncio.wait_for(dispatcher.done_event.wait(), timeout=1)
            await asyncio.sleep(0)
            gc.collect()

            self.assertIsNone(work_item_ref())
            self.assertIsNone(stream_queue_ref())
        finally:
            dispatcher.stop()

    async def test_turn_queue_position_reflects_sorted_order(self) -> None:
        dispatcher = SessionDispatcher(lock=asyncio.Lock())

        dispatcher.submit(
            SessionWorkItem(
                kind="heartbeat",
                priority=PRIORITY_HEARTBEAT,
                content="heartbeat",
                agent_id="main",
                session_id="main-main",
                turn_id="heartbeat-turn",
            )
        )
        dispatcher.submit(
            SessionWorkItem(
                kind="user",
                priority=PRIORITY_USER,
                content="user",
                agent_id="main",
                session_id="main-main",
                turn_id="user-turn",
            )
        )
        dispatcher.submit(
            SessionWorkItem(
                kind="announce",
                priority=PRIORITY_ANNOUNCE,
                content="announce",
                agent_id="main",
                session_id="main-main",
                turn_id="announce-turn",
            )
        )

        self.assertEqual(dispatcher.turn_queue_position("user-turn"), 1)
        self.assertEqual(dispatcher.turn_queue_position("announce-turn"), 2)
        self.assertEqual(dispatcher.turn_queue_position("heartbeat-turn"), 3)


class UserTurnDispatcherLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        user_turn_coordinator._runtimes.clear()
        user_turn_coordinator._session_to_turn.clear()
        user_turn_coordinator._terminal_turns.clear()

    async def test_error_event_retains_error_terminal_status(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        dispatcher = SessionDispatcher(lock=asyncio.Lock())
        task = SessionWorkItem(
            kind="user",
            priority=PRIORITY_USER,
            content="hello",
            agent_id="main",
            session_id="main-main",
            turn_id=runtime.turn_id,
            stream_queue=runtime.stream_queue,
        )

        async def fake_stream(*args, **kwargs):
            yield TurnEvent.error("model failed")

        with patch(
            "runtime.user_turn_stream.iter_user_turn_events",
            fake_stream,
        ):
            await dispatcher._execute_user(task)

        streamed = await task.stream_queue.get()
        self.assertIsInstance(streamed, TurnEvent)
        self.assertEqual(streamed.type, "error")
        retained = user_turn_coordinator.get(runtime.turn_id)
        self.assertIsNotNone(retained)
        self.assertEqual(retained.status, "error")
        self.assertEqual(retained.error, "model failed")

    async def test_aborted_event_retains_cancelled_terminal_status(self) -> None:
        runtime = user_turn_coordinator.create_queued("main", "main-main")
        dispatcher = SessionDispatcher(lock=asyncio.Lock())
        task = SessionWorkItem(
            kind="user",
            priority=PRIORITY_USER,
            content="hello",
            agent_id="main",
            session_id="main-main",
            turn_id=runtime.turn_id,
            stream_queue=runtime.stream_queue,
        )

        async def fake_stream(*args, **kwargs):
            yield TurnEvent.from_payload(
                {
                    "type": "aborted",
                    "reason": "stopped_by_user",
                }
            )

        with patch(
            "runtime.user_turn_stream.iter_user_turn_events",
            fake_stream,
        ):
            await dispatcher._execute_user(task)

        retained = user_turn_coordinator.get(runtime.turn_id)
        self.assertIsNotNone(retained)
        self.assertEqual(retained.status, "cancelled")


class SystemWorkDispatcherLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatcher_manager_passes_store_to_session_dispatcher(self) -> None:
        work_store = _RecordingWorkStore()
        manager = DispatcherManager(work_store=work_store)

        dispatcher = manager.get("main", "main-main", asyncio.Lock())

        self.assertIs(dispatcher._work_store, work_store)
        manager.cleanup("main", "main-main")

    async def test_error_event_marks_work_failed_without_success_callback(self) -> None:
        work_store = _RecordingWorkStore()
        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            work_store=work_store,
        )
        succeeded: list[str] = []
        failed: list[str] = []
        task = SessionWorkItem(
            kind="cron",
            priority=PRIORITY_CRON,
            content="run scheduled task",
            agent_id="main",
            session_id="main-main",
            work_id="work-1",
            on_success=lambda: succeeded.append("done"),
            on_failure_async=lambda error: _record_failure(failed, error),
        )

        async def fake_stream(*args, **kwargs):
            yield {"type": "error", "error": "model failed"}

        with patch("runtime.agent.agent_manager.astream", fake_stream):
            await dispatcher._execute_system(task)

        self.assertEqual(succeeded, [])
        self.assertEqual(failed, ["model failed"])
        self.assertEqual(work_store.running, ["work-1"])
        self.assertEqual(work_store.done, [])
        self.assertEqual(work_store.failed, [("work-1", "model failed")])


class _RecordingWorkStore:
    def __init__(self) -> None:
        self.running: list[str] = []
        self.done: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def mark_running(self, work_id: str) -> bool:
        self.running.append(work_id)
        return True

    def mark_done(self, work_id: str) -> None:
        self.done.append(work_id)

    def mark_failed(self, work_id: str, error: str) -> None:
        self.failed.append((work_id, error))


async def _record_failure(target: list[str], error: Exception) -> None:
    target.append(str(error))


if __name__ == "__main__":
    unittest.main()
