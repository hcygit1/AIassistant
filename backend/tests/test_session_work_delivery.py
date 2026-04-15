from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class SessionWorkDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.delivery = SessionWorkDelivery()

    def test_deliver_persists_record_and_submits_work_item(self) -> None:
        dispatcher = _FakeDispatcher()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionWorkStore(Path(tmpdir) / "session_work.db")
            with (
                patch("sessions.session_work_delivery.session_work_store", store),
                patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
                patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
            ):
                pos = self.delivery.deliver(
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

            with (
                patch("sessions.session_work_delivery.session_work_store", store),
                patch("sessions.session_lock_manager.session_lock_manager.get_lock", return_value=_FakeLock()),
                patch("sessions.session_dispatcher.dispatcher_manager.get", return_value=dispatcher),
            ):
                recovered = self.delivery.recover_pending_work()

        self.assertEqual(recovered, 1)
        self.assertEqual(len(dispatcher.submitted), 1)
        self.assertEqual(dispatcher.submitted[0].work_id, record.id)

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
