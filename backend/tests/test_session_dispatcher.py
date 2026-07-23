from __future__ import annotations

import asyncio
import gc
import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

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
    async def test_stop_cancels_queued_user_and_rejects_new_work(self) -> None:
        coordinator = Mock()
        stream_queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            turn_coordinator=coordinator,
        )
        queued_user = SessionWorkItem(
            kind="user",
            priority=PRIORITY_USER,
            content="queued user",
            agent_id="main",
            session_id="main-main",
            turn_id="turn-queued",
            stream_queue=stream_queue,
        )
        dispatcher.submit(queued_user)

        await dispatcher.aclose()

        coordinator.set_cancelled.assert_called_once_with("turn-queued")
        self.assertIsNone(await stream_queue.get())
        with self.assertRaisesRegex(RuntimeError, "dispatcher is closing"):
            dispatcher.submit(queued_user)

    async def test_aclose_propagates_caller_cancellation(self) -> None:
        cancellation_started = asyncio.Event()
        release_cancellation = asyncio.Event()

        async def slow_consumer() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_started.set()
                await release_cancellation.wait()
                raise

        dispatcher = SessionDispatcher(lock=asyncio.Lock())
        dispatcher._task = asyncio.create_task(slow_consumer())
        close_task = asyncio.create_task(dispatcher.aclose())
        await asyncio.wait_for(cancellation_started.wait(), timeout=1)

        close_task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await close_task
        release_cancellation.set()

    async def test_aclose_propagates_consumer_failure(self) -> None:
        started = asyncio.Event()
        cancellation_started = asyncio.Event()

        async def failing_consumer() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_started.set()
                raise RuntimeError("consumer close failed")

        dispatcher = SessionDispatcher(lock=asyncio.Lock())
        dispatcher._task = asyncio.create_task(failing_consumer())
        await asyncio.wait_for(started.wait(), timeout=1)

        with self.assertRaisesRegex(RuntimeError, "consumer close failed"):
            await dispatcher.aclose()
        self.assertTrue(cancellation_started.is_set())

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

    async def test_dispatcher_uses_injected_user_stream_and_coordinator(self) -> None:
        observed: list[tuple[str, str, str, str]] = []
        coordinator = Mock()
        stream_queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()

        async def injected_stream(message, session_id, agent_id, turn_id):
            observed.append((message, session_id, agent_id, turn_id))
            yield TurnEvent.from_payload({"type": "token", "content": "ok"})

        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            user_stream=injected_stream,
            turn_coordinator=coordinator,
        )
        task = SessionWorkItem(
            kind="user",
            priority=PRIORITY_USER,
            content="injected user turn",
            agent_id="main",
            session_id="main-main",
            turn_id="turn-injected",
            stream_queue=stream_queue,
        )

        await dispatcher._execute_user(task)

        self.assertEqual(
            observed,
            [("injected user turn", "main-main", "main", "turn-injected")],
        )
        self.assertTrue(coordinator.set_running.called)
        coordinator.set_done.assert_called_once_with("turn-injected")
        self.assertEqual((await stream_queue.get()).type, "token")
        self.assertIsNone(await stream_queue.get())

    async def test_dispatcher_delegates_to_injected_user_executor(self) -> None:
        executor = Mock()
        executor.execute = AsyncMock()
        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            user_executor=executor,
        )
        task = SessionWorkItem(
            kind="user",
            priority=PRIORITY_USER,
            content="run delegated user turn",
            agent_id="main",
            session_id="main-main",
            turn_id="turn-delegated",
            stream_queue=asyncio.Queue(),
        )

        await dispatcher._execute_user(task)

        executor.execute.assert_awaited_once_with(task)

    async def test_aclose_cancels_user_turn_waiting_for_session_lock(self) -> None:
        coordinator = Mock()
        stream_queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        lock = asyncio.Lock()
        await lock.acquire()
        dispatcher = SessionDispatcher(
            lock=lock,
            turn_coordinator=coordinator,
        )
        dispatcher.start()
        dispatcher.submit(
            SessionWorkItem(
                kind="user",
                priority=PRIORITY_USER,
                content="waiting user",
                agent_id="main",
                session_id="main-main",
                turn_id="turn-waiting",
                stream_queue=stream_queue,
            )
        )
        await asyncio.sleep(0)

        await dispatcher.aclose()

        coordinator.set_cancelled.assert_called_once_with("turn-waiting")
        coordinator.set_error.assert_not_called()
        self.assertIsNone(await stream_queue.get())
        self.assertTrue(lock.locked())
        lock.release()


class SystemWorkDispatcherLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_aclose_settles_current_and_queued_system_work(self) -> None:
        started = asyncio.Event()
        hold = asyncio.Event()
        cancelled: list[str] = []
        work_store = Mock()
        work_store.mark_running.return_value = True
        work_store.cancel_queued.return_value = True

        async def blocking_stream(**_kwargs):
            started.set()
            await hold.wait()
            yield {"type": "done", "content": "unexpected"}

        lock = asyncio.Lock()
        dispatcher = SessionDispatcher(
            lock=lock,
            work_store=work_store,
            system_stream=blocking_stream,
        )
        dispatcher.start()
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="running",
                agent_id="main",
                session_id="main-main",
                work_id="work-running",
                on_cancel=lambda: cancelled.append("work-running"),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        dispatcher.submit(
            SessionWorkItem(
                kind="heartbeat",
                priority=PRIORITY_HEARTBEAT,
                content="queued",
                agent_id="main",
                session_id="main-main",
                work_id="work-queued",
                on_cancel=lambda: cancelled.append("work-queued"),
            )
        )

        await dispatcher.aclose()

        self.assertFalse(lock.locked())
        self.assertEqual(dispatcher.pending_count, 0)
        work_store.mark_cancelled.assert_called_once_with("work-running")
        work_store.cancel_queued.assert_called_once_with("work-queued")
        self.assertCountEqual(
            cancelled,
            ["work-running", "work-queued"],
        )

    async def test_aclose_keeps_cancelled_status_when_stream_swallows_cancel(self) -> None:
        started = asyncio.Event()
        cancelled: list[str] = []
        succeeded: list[str] = []
        work_store = Mock()
        work_store.mark_running.return_value = True

        async def cancellation_suppressing_stream(**_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            yield {"type": "done", "content": "unexpected"}

        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            work_store=work_store,
            system_stream=cancellation_suppressing_stream,
        )
        dispatcher.start()
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="running",
                agent_id="main",
                session_id="main-main",
                work_id="work-running",
                on_success=lambda: succeeded.append("work-running"),
                on_cancel=lambda: cancelled.append("work-running"),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        await dispatcher.aclose()

        work_store.mark_cancelled.assert_called_once_with("work-running")
        work_store.mark_done.assert_not_called()
        self.assertEqual(cancelled, ["work-running"])
        self.assertEqual(succeeded, [])

    async def test_aclose_stops_stream_that_keeps_yielding_after_cancel(self) -> None:
        started = asyncio.Event()
        yielded_after_cancel = asyncio.Event()
        stream_closed = asyncio.Event()
        work_store = Mock()
        work_store.mark_running.return_value = True

        async def cancellation_suppressing_stream(**_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            try:
                while True:
                    yielded_after_cancel.set()
                    yield {"type": "token", "content": "unexpected"}
                    await asyncio.sleep(0)
            finally:
                stream_closed.set()

        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            work_store=work_store,
            system_stream=cancellation_suppressing_stream,
        )
        dispatcher.start()
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="running",
                agent_id="main",
                session_id="main-main",
                work_id="work-running",
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        close_task = asyncio.create_task(dispatcher.aclose())
        await asyncio.wait_for(yielded_after_cancel.wait(), timeout=1)
        await asyncio.sleep(0)

        try:
            self.assertTrue(close_task.done())
        finally:
            if not close_task.done():
                close_task.cancel()
            try:
                await close_task
            except asyncio.CancelledError:
                pass

        work_store.mark_cancelled.assert_called_once_with("work-running")
        self.assertTrue(stream_closed.is_set())

    async def test_aclose_cancels_stream_error_after_swallowed_cancel(self) -> None:
        started = asyncio.Event()
        cancelled: list[str] = []
        failed: list[str] = []
        work_store = Mock()
        work_store.mark_running.return_value = True

        async def failing_after_cancel_stream(**_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            raise RuntimeError("shutdown cleanup failed")
            yield {"type": "done", "content": "unexpected"}

        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            work_store=work_store,
            system_stream=failing_after_cancel_stream,
        )
        dispatcher.start()
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="running",
                agent_id="main",
                session_id="main-main",
                work_id="work-running",
                on_cancel=lambda: cancelled.append("work-running"),
                on_failure=lambda: failed.append("work-running"),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        await dispatcher.aclose()

        work_store.mark_cancelled.assert_called_once_with("work-running")
        work_store.mark_failed.assert_not_called()
        self.assertEqual(cancelled, ["work-running"])
        self.assertEqual(failed, [])

    async def test_dispatcher_resolves_default_work_store_at_construction(self) -> None:
        dispatcher = SessionDispatcher(lock=asyncio.Lock())

        self.assertIsNotNone(dispatcher._work_store)

    async def test_dispatcher_uses_injected_system_stream(self) -> None:
        work_store = _RecordingWorkStore()
        observed: list[tuple[str, str, str]] = []
        succeeded: list[str] = []

        async def injected_stream(**kwargs):
            observed.append(
                (
                    kwargs["message"],
                    kwargs["session_id"],
                    kwargs["agent_id"],
                )
            )
            yield {"type": "done", "content": "ok"}

        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            work_store=work_store,
            system_stream=injected_stream,
        )
        task = SessionWorkItem(
            kind="cron",
            priority=PRIORITY_CRON,
            content="run injected stream",
            agent_id="main",
            session_id="main-main",
            work_id="work-injected",
            on_success=lambda: (
                succeeded.append("done"),
                work_store.events.append(("callback", "success")),
            ),
        )

        await dispatcher._execute_system(task)

        self.assertEqual(observed, [("run injected stream", "main-main", "main")])
        self.assertEqual(succeeded, ["done"])
        self.assertEqual(work_store.done, ["work-injected"])
        self.assertEqual(
            work_store.events,
            [
                ("running", "work-injected"),
                ("done", "work-injected"),
                ("callback", "success"),
            ],
        )

    async def test_dispatcher_delegates_to_injected_system_executor(self) -> None:
        executor = Mock()
        executor.execute = AsyncMock()
        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            system_executor=executor,
        )
        task = SessionWorkItem(
            kind="cron",
            priority=PRIORITY_CRON,
            content="run delegated work",
            agent_id="main",
            session_id="main-main",
            work_id="work-delegated",
        )

        await dispatcher._execute_system(task)
        dispatcher._cancel_running_system_item(task)

        executor.execute.assert_awaited_once_with(task)
        executor.cancel_running.assert_called_once_with(task)

    async def test_dispatcher_manager_passes_store_to_session_dispatcher(self) -> None:
        work_store = _RecordingWorkStore()
        manager = DispatcherManager(work_store=work_store)

        dispatcher = manager.get("main", "main-main", asyncio.Lock())

        self.assertIs(dispatcher._work_store, work_store)
        manager.cleanup("main", "main-main")

    async def test_dispatcher_manager_exposes_injected_lock_manager(self) -> None:
        lock = asyncio.Lock()
        lock_manager = Mock()
        lock_manager.get_lock.return_value = Mock(lock=lock)

        manager = DispatcherManager(lock_manager=lock_manager)
        dispatcher = manager.get("main", "main-main")

        self.assertIs(manager.lock_manager, lock_manager)
        self.assertIs(dispatcher._lock, lock)
        lock_manager.get_lock.assert_called_once_with("main", "main-main")
        manager.cleanup("main", "main-main")

    async def test_manager_keeps_closing_dispatcher_until_consumer_exits(self) -> None:
        started = asyncio.Event()
        cancellation_started = asyncio.Event()
        release_cancellation = asyncio.Event()
        closed: list[str] = []
        work_store = Mock()
        work_store.mark_running.return_value = True

        async def blocking_stream(**_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_started.set()
                await release_cancellation.wait()
                raise
            yield {"type": "done", "content": "unexpected"}

        manager = DispatcherManager(
            work_store=work_store,
            system_stream=blocking_stream,
        )
        dispatcher = manager.get("main", "main-main", asyncio.Lock())
        original_stop = dispatcher.stop
        dispatcher.stop = Mock(wraps=original_stop)
        dispatcher.submit(
            SessionWorkItem(
                kind="cron",
                priority=PRIORITY_CRON,
                content="running",
                agent_id="main",
                session_id="main-main",
                work_id="work-running",
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        manager.cleanup_when_closed(
            "main",
            "main-main",
            on_closed=lambda: closed.append("first"),
        )
        await asyncio.wait_for(cancellation_started.wait(), timeout=1)
        manager.cleanup_when_closed(
            "main",
            "main-main",
            on_closed=lambda: closed.append("second"),
        )
        await asyncio.sleep(0)

        self.assertIs(manager.get("main", "main-main"), dispatcher)
        self.assertEqual(closed, [])
        dispatcher.stop.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "dispatcher is closing"):
            dispatcher.submit(
                SessionWorkItem(
                    kind="cron",
                    priority=PRIORITY_CRON,
                    content="rejected",
                    agent_id="main",
                    session_id="main-main",
                )
            )

        release_cancellation.set()
        await manager.aclose_session("main", "main-main")
        self.assertCountEqual(closed, ["first", "second"])

        replacement = manager.get("main", "main-main", asyncio.Lock())
        self.assertIsNot(replacement, dispatcher)
        await manager.aclose_session("main", "main-main")

    async def test_error_event_marks_work_failed_without_success_callback(self) -> None:
        work_store = _RecordingWorkStore()
        dispatcher = SessionDispatcher(
            lock=asyncio.Lock(),
            work_store=work_store,
        )
        succeeded: list[str] = []
        failed: list[str] = []
        async def record_failure(error: Exception) -> None:
            failed.append(str(error))
            work_store.events.append(("callback", "failure"))

        task = SessionWorkItem(
            kind="cron",
            priority=PRIORITY_CRON,
            content="run scheduled task",
            agent_id="main",
            session_id="main-main",
            work_id="work-1",
            on_success=lambda: succeeded.append("done"),
            on_failure_async=record_failure,
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
        self.assertEqual(
            work_store.events,
            [
                ("running", "work-1"),
                ("failed", "work-1"),
                ("callback", "failure"),
            ],
        )

    async def test_lock_timeout_marks_work_failed_before_failure_callback(self) -> None:
        lock = asyncio.Lock()
        await lock.acquire()
        work_store = _RecordingWorkStore()

        async def record_failure(_error: Exception) -> None:
            work_store.events.append(("callback", "failure"))

        dispatcher = SessionDispatcher(
            lock=lock,
            work_store=work_store,
        )
        task = SessionWorkItem(
            kind="heartbeat",
            priority=PRIORITY_HEARTBEAT,
            content="heartbeat",
            agent_id="main",
            session_id="main-main",
            work_id="work-timeout",
            on_failure_async=record_failure,
        )

        try:
            with patch("sessions.session_dispatcher.SYSTEM_TIMEOUT_SEC", 0.001):
                await dispatcher._execute_system(task)
        finally:
            lock.release()

        self.assertEqual(
            work_store.events,
            [
                ("running", "work-timeout"),
                ("failed", "work-timeout"),
                ("callback", "failure"),
            ],
        )


class _RecordingWorkStore:
    def __init__(self) -> None:
        self.running: list[str] = []
        self.done: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.events: list[tuple[str, str]] = []

    def mark_running(self, work_id: str) -> bool:
        self.running.append(work_id)
        self.events.append(("running", work_id))
        return True

    def mark_done(self, work_id: str) -> None:
        self.done.append(work_id)
        self.events.append(("done", work_id))

    def mark_failed(self, work_id: str, error: str) -> None:
        self.failed.append((work_id, error))
        self.events.append(("failed", work_id))
if __name__ == "__main__":
    unittest.main()
