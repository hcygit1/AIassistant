from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_work_delivery import SessionWorkDelivery
from sessions.session_work_store import SessionWorkStore


class _FakeLock:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()


class _FakeDispatcher:
    def __init__(self) -> None:
        self.submitted = []

    def submit(self, work_item) -> int:
        self.submitted.append(work_item)
        return len(self.submitted)


class _FailingDispatcher(_FakeDispatcher):
    def submit(self, work_item) -> int:
        raise RuntimeError("dispatcher unavailable")


class _FakeDispatcherManager:
    def __init__(self, dispatcher: _FakeDispatcher) -> None:
        self.dispatcher = dispatcher
        self.calls = []

    def get(self, agent_id: str, session_id: str, lock: asyncio.Lock):
        self.calls.append((agent_id, session_id, lock))
        return self.dispatcher


class _FakeDispatcherManagerWithStore(_FakeDispatcherManager):
    def __init__(self, dispatcher, work_store) -> None:
        super().__init__(dispatcher)
        self.work_store = work_store


class _FakeDispatcherManagerWithRuntime(_FakeDispatcherManagerWithStore):
    def __init__(self, dispatcher, work_store, lock_manager) -> None:
        super().__init__(dispatcher, work_store)
        self.lock_manager = lock_manager


class _FakeLockManager:
    def __init__(self) -> None:
        self.session_lock = _FakeLock()
        self.calls = []

    def get_lock(self, agent_id: str, session_id: str) -> _FakeLock:
        self.calls.append((agent_id, session_id))
        return self.session_lock


class SessionWorkDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.delivery = SessionWorkDelivery()

    def test_deliver_persists_record_and_submits_work_item(self) -> None:
        dispatcher = _FakeDispatcher()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            dispatcher_manager = _FakeDispatcherManager(dispatcher)
            lock_manager = _FakeLockManager()
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=dispatcher_manager,
                lock_manager=lock_manager,
            )
            pos = delivery.deliver(
                kind="cron",
                priority=2,
                content="hello",
                agent_id="main",
                session_id="main-main",
                run_id="cron-1",
                recover_on_restart=True,
            )

            records = store.get_recoverable_pending()

        self.assertEqual(pos, 1)
        self.assertEqual(len(dispatcher.submitted), 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].kind, "cron")
        self.assertEqual(dispatcher.submitted[0].work_id, records[0].id)
        self.assertEqual(lock_manager.calls, [("main", "main-main")])
        self.assertEqual(
            dispatcher_manager.calls,
            [("main", "main-main", lock_manager.session_lock.lock)],
        )

    def test_store_marks_queued_and_running_work_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            queued = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="queued",
                priority=2,
            )
            running = store.create_record(
                kind="heartbeat",
                agent_id="main",
                session_id="main-main",
                content="running",
                priority=3,
            )
            running.status = "running"
            store.insert(queued)
            store.insert(running)

            store.mark_cancelled(queued.id)
            store.mark_cancelled(running.id)

            self.assertEqual(store.get(queued.id).status, "cancelled")
            self.assertEqual(store.get(running.id).status, "cancelled")

    def test_deliver_marks_record_failed_when_dispatcher_submission_fails(self) -> None:
        dispatcher = _FailingDispatcher()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=_FakeDispatcherManager(dispatcher),
                lock_manager=_FakeLockManager(),
            )

            with self.assertRaisesRegex(RuntimeError, "dispatcher unavailable"):
                delivery.deliver(
                    kind="cron",
                    priority=2,
                    content="hello",
                    agent_id="main",
                    session_id="main-main",
                    run_id="cron-submit-failed",
                    recover_on_restart=True,
                )
            records = store.query(run_id="cron-submit-failed")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "failed")
        self.assertEqual(records[0].last_error, "dispatcher unavailable")

    def test_deliver_marks_record_failed_when_created_hook_fails(self) -> None:
        dispatcher = _FakeDispatcher()

        def fail_binding(_record) -> None:
            raise RuntimeError("binding failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=_FakeDispatcherManager(dispatcher),
                lock_manager=_FakeLockManager(),
            )

            with self.assertRaisesRegex(RuntimeError, "binding failed"):
                delivery.deliver(
                    kind="announce",
                    priority=0,
                    content="hello",
                    agent_id="main",
                    session_id="main-main",
                    run_id="announce-bind-failed",
                    on_record_created=fail_binding,
                )
            records = store.query(run_id="announce-bind-failed")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "failed")
        self.assertEqual(records[0].last_error, "binding failed")
        self.assertEqual(dispatcher.submitted, [])

    def test_recover_pending_work_resubmits_recoverable_records(self) -> None:
        dispatcher = _FakeDispatcher()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            record = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="recover me",
                priority=2,
                run_id="cron-2",
                recover_on_restart=True,
            )
            store.insert(record)
            dispatcher_manager = _FakeDispatcherManager(dispatcher)
            lock_manager = _FakeLockManager()
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=dispatcher_manager,
                lock_manager=lock_manager,
            )

            recovered = delivery.recover_pending_work()

        self.assertEqual(recovered, 1)
        self.assertEqual(len(dispatcher.submitted), 1)
        self.assertEqual(dispatcher.submitted[0].work_id, record.id)

    def test_recovery_restores_callbacks_for_recoverable_work(self) -> None:
        dispatcher = _FakeDispatcher()
        on_success = Mock()
        resolver = Mock(return_value={"on_success": on_success})
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            record = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="recover callbacks",
                priority=2,
                run_id="cron-recover",
                recover_on_restart=True,
            )
            store.insert(record)
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=_FakeDispatcherManager(dispatcher),
                lock_manager=_FakeLockManager(),
                recovery_callback_resolver=resolver,
            )

            recovered = delivery.recover_pending_work()

        self.assertEqual(recovered, 1)
        resolver.assert_called_once()
        self.assertEqual(resolver.call_args.args[0].id, record.id)
        self.assertIs(dispatcher.submitted[0].on_success, on_success)

    def test_recovery_drops_work_rejected_by_callback_resolver(self) -> None:
        dispatcher = _FakeDispatcher()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            record = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="stale work",
                priority=2,
                run_id="cron-stale",
                recover_on_restart=True,
            )
            store.insert(record)
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=_FakeDispatcherManager(dispatcher),
                lock_manager=_FakeLockManager(),
                recovery_callback_resolver=lambda _record: None,
            )

            recovered = delivery.recover_pending_work()
            current = store.get(record.id)

        self.assertEqual(recovered, 0)
        self.assertEqual(dispatcher.submitted, [])
        self.assertEqual(current.status, "failed")
        self.assertEqual(current.last_error, "stale recoverable work claim")

    def test_recovery_requeues_interrupted_running_record(self) -> None:
        dispatcher = _FakeDispatcher()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            record = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="recover running",
                priority=2,
                recover_on_restart=True,
            )
            record.status = "running"
            record.started_at_ms = 123
            store.insert(record)
            dispatcher_manager = _FakeDispatcherManager(dispatcher)
            lock_manager = _FakeLockManager()
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=dispatcher_manager,
                lock_manager=lock_manager,
            )

            recovered = delivery.recover_pending_work()
            current = store.get(record.id)

        self.assertEqual(recovered, 1)
        self.assertEqual(current.status, "queued")
        self.assertIsNone(current.started_at_ms)

    def test_constructor_resolves_defaults_once(self) -> None:
        dispatcher = _FakeDispatcher()
        dispatcher_manager = _FakeDispatcherManager(dispatcher)
        lock_manager = _FakeLockManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            delivery = SessionWorkDelivery(
                work_store=store,
                dispatcher_manager=dispatcher_manager,
                lock_manager=lock_manager,
            )

            self.assertIs(delivery.work_store, store)
            self.assertIs(delivery.dispatcher_manager, dispatcher_manager)
            self.assertIs(delivery.lock_manager, lock_manager)

    def test_delivery_uses_dispatcher_store_when_store_is_omitted(self) -> None:
        dispatcher = _FakeDispatcher()
        lock_manager = _FakeLockManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            dispatcher_manager = _FakeDispatcherManagerWithStore(
                dispatcher,
                store,
            )
            delivery = SessionWorkDelivery(
                dispatcher_manager=dispatcher_manager,
                lock_manager=lock_manager,
            )

            self.assertIs(delivery.work_store, store)

    def test_delivery_uses_dispatcher_lock_when_lock_is_omitted(self) -> None:
        dispatcher = _FakeDispatcher()
        lock_manager = _FakeLockManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            dispatcher_manager = _FakeDispatcherManagerWithRuntime(
                dispatcher,
                store,
                lock_manager,
            )

            delivery = SessionWorkDelivery(
                dispatcher_manager=dispatcher_manager,
            )

            self.assertIs(delivery.lock_manager, lock_manager)

    def test_new_dispatcher_uses_explicit_delivery_lock_manager(self) -> None:
        lock_manager = _FakeLockManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")

            delivery = SessionWorkDelivery(
                work_store=store,
                lock_manager=lock_manager,
            )

            self.assertIs(
                delivery.dispatcher_manager.lock_manager,
                lock_manager,
            )

    def test_store_fails_only_unrecoverable_pending_work_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            queued = store.create_record(
                kind="announce",
                agent_id="main",
                session_id="main-main",
                content="queued",
                priority=0,
                recover_on_restart=False,
            )
            running = store.create_record(
                kind="heartbeat",
                agent_id="main",
                session_id="main-main",
                content="running",
                priority=3,
                recover_on_restart=False,
            )
            running.status = "running"
            recoverable = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="recoverable",
                priority=2,
                recover_on_restart=True,
            )
            store.insert(queued)
            store.insert(running)
            store.insert(recoverable)

            failed = store.fail_unrecoverable_pending(
                "interrupted by process restart"
            )

            queued_current = store.get(queued.id)
            running_current = store.get(running.id)
            recoverable_current = store.get(recoverable.id)

        self.assertEqual(failed, 2)
        self.assertEqual(queued_current.status, "failed")
        self.assertEqual(running_current.status, "failed")
        self.assertEqual(
            queued_current.last_error,
            "interrupted by process restart",
        )
        self.assertEqual(recoverable_current.status, "queued")

    def test_store_query_can_find_failed_announce_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            record = store.create_record(
                kind="announce",
                agent_id="main",
                session_id="main-main",
                content="announce body",
                priority=0,
                run_id="run-announce-1",
                recover_on_restart=False,
            )
            store.insert(record)
            store.mark_failed(record.id, "dispatcher timeout")

            items = store.query(kind="announce", status="failed", run_id="run-announce-1")
            total = store.count(kind="announce", status="failed", run_id="run-announce-1")

        self.assertEqual(total, 1)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].run_id, "run-announce-1")
        self.assertEqual(items[0].last_error, "dispatcher timeout")

    def test_cancelled_record_cannot_be_marked_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            record = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="cancel me",
                priority=2,
            )
            store.insert(record)

            cancelled = store.cancel_queued(record.id)
            claimed = store.mark_running(record.id)

            current = store.get(record.id)
        self.assertTrue(cancelled)
        self.assertFalse(claimed)
        self.assertEqual(current.status, "cancelled")

    def test_store_prune_finished_older_than_deletes_only_old_terminal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            old_done = store.create_record(
                kind="announce",
                agent_id="main",
                session_id="main-main",
                content="old done",
                priority=0,
            )
            old_done.finished_at_ms = 1_000
            old_done.status = "done"
            store.insert(old_done)

            old_failed = store.create_record(
                kind="heartbeat",
                agent_id="main",
                session_id="main-main",
                content="old failed",
                priority=3,
            )
            old_failed.finished_at_ms = 2_000
            old_failed.status = "failed"
            old_failed.last_error = "boom"
            store.insert(old_failed)

            recent_done = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="recent done",
                priority=2,
            )
            recent_done.finished_at_ms = 9_000
            recent_done.status = "done"
            store.insert(recent_done)

            queued = store.create_record(
                kind="cron",
                agent_id="main",
                session_id="main-main",
                content="queued",
                priority=2,
            )
            store.insert(queued)

            deleted = store.prune_finished_older_than(older_than_ms=5_000)
            remaining = store.query(limit=10)

        self.assertEqual(deleted, 2)
        remaining_ids = {item.id for item in remaining}
        self.assertNotIn(old_done.id, remaining_ids)
        self.assertNotIn(old_failed.id, remaining_ids)
        self.assertIn(recent_done.id, remaining_ids)
        self.assertIn(queued.id, remaining_ids)

    def test_store_prune_finished_older_than_can_filter_by_kind_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            announce_a = store.create_record(
                kind="announce",
                agent_id="main",
                session_id="main-main",
                content="announce a",
                priority=0,
            )
            announce_a.finished_at_ms = 1_000
            announce_a.status = "done"
            store.insert(announce_a)

            announce_b = store.create_record(
                kind="announce",
                agent_id="main",
                session_id="main-main",
                content="announce b",
                priority=0,
            )
            announce_b.finished_at_ms = 2_000
            announce_b.status = "done"
            store.insert(announce_b)

            heartbeat = store.create_record(
                kind="heartbeat",
                agent_id="main",
                session_id="main-main",
                content="heartbeat",
                priority=3,
            )
            heartbeat.finished_at_ms = 1_500
            heartbeat.status = "done"
            store.insert(heartbeat)

            deleted = store.prune_finished_older_than(
                older_than_ms=5_000,
                kinds=["announce"],
                limit=1,
            )
            announce_left = store.count(kind="announce")
            heartbeat_left = store.count(kind="heartbeat")

        self.assertEqual(deleted, 1)
        self.assertEqual(announce_left, 1)
        self.assertEqual(heartbeat_left, 1)


if __name__ == "__main__":
    unittest.main()
