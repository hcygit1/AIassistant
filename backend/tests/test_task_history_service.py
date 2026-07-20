from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.task_history_service import (
    TaskHistoryError,
    TaskHistoryService,
)
from scheduler.task_store import (
    TaskKind,
    TaskRecord,
    TaskStatus,
    TaskStore,
)
from sessions.session_dispatcher import DispatcherManager
from sessions.session_dispatcher import dispatcher_manager
from sessions.session_work_store import SessionWorkStore
from sessions.session_work_store import session_work_store
from sessions.session_work_runtime import SessionWorkRuntime


class _DispatcherManager:
    def __init__(self) -> None:
        self.cancelled: list[tuple[str, str, str]] = []

    def cancel_work(
        self,
        agent_id: str,
        session_id: str,
        work_id: str,
    ) -> bool:
        self.cancelled.append((agent_id, session_id, work_id))
        return True


class TaskHistoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.task_store = TaskStore(self.root / "tasks.db")
        self.work_store = SessionWorkStore(self.root / "work.db")
        self.dispatchers = _DispatcherManager()
        self.service = TaskHistoryService(
            task_store=self.task_store,
            work_store=self.work_store,
            dispatcher_manager=self.dispatchers,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_query_merges_canonical_sources_without_heartbeat_duplicates(self) -> None:
        self.task_store.insert(
            TaskRecord(
                id="heartbeat-result",
                kind=TaskKind.HEARTBEAT,
                agent_id="main",
                name="heartbeat:sent",
                status=TaskStatus.SUCCESS,
                created_at_ms=1_000,
            )
        )
        self._insert_work("heartbeat-work", "heartbeat", "done", 1_100)
        self._insert_work("cron-work", "cron", "queued", 1_200, "cron-1")
        self._insert_work(
            "reminder-work",
            "cron",
            "done",
            1_300,
            "reminder-1",
        )
        self._insert_work("announce-work", "announce", "failed", 1_400)

        page = self.service.query(agent_id="main", limit=10)

        self.assertEqual(page.total, 4)
        self.assertEqual(
            [item["id"] for item in page.items],
            [
                "announce-work",
                "reminder-work",
                "cron-work",
                "heartbeat-result",
            ],
        )
        self.assertEqual(
            [item["kind"] for item in page.items],
            ["system", "reminder", "cron", "heartbeat"],
        )
        self.assertEqual(
            [item["status"] for item in page.items],
            ["failed", "success", "pending", "success"],
        )

    def test_query_applies_filters_before_counting_and_pagination(self) -> None:
        self._insert_work("cron-old", "cron", "queued", 1_000, "cron-1")
        self._insert_work("cron-new", "cron", "queued", 2_000, "cron-2")
        self._insert_work(
            "reminder",
            "cron",
            "queued",
            3_000,
            "reminder-1",
        )

        page = self.service.query(
            kind="cron",
            status="pending",
            limit=1,
            offset=1,
        )

        self.assertEqual(page.total, 2)
        self.assertEqual([item["id"] for item in page.items], ["cron-old"])

    def test_query_uses_stable_id_order_for_equal_timestamps(self) -> None:
        for task_id in ("heartbeat-a", "heartbeat-z"):
            self.task_store.insert(
                TaskRecord(
                    id=task_id,
                    kind=TaskKind.HEARTBEAT,
                    agent_id="main",
                    status=TaskStatus.SUCCESS,
                    created_at_ms=1_000,
                )
            )

        page = self.service.query(
            kind="heartbeat",
            limit=1,
            offset=0,
        )

        self.assertEqual(page.total, 2)
        self.assertEqual([item["id"] for item in page.items], ["heartbeat-z"])

    def test_cancel_removes_queued_work_and_rejects_running_work(self) -> None:
        queued = self._insert_work("queued", "cron", "queued", 1_000)
        running = self._insert_work("running", "cron", "running", 2_000)

        result = self.service.cancel(queued.id)

        self.assertTrue(result.ok)
        self.assertEqual(self.work_store.get(queued.id).status, "cancelled")
        self.assertEqual(
            self.dispatchers.cancelled,
            [("main", "main-main", queued.id)],
        )
        with self.assertRaises(TaskHistoryError) as raised:
            self.service.cancel(running.id)
        self.assertEqual(raised.exception.code, "running")

    def test_inherits_work_store_from_injected_dispatcher_manager(self) -> None:
        inherited_store = SessionWorkStore(self.root / "inherited-work.db")
        dispatcher_manager = DispatcherManager(work_store=inherited_store)
        service = TaskHistoryService(
            task_store=self.task_store,
            dispatcher_manager=dispatcher_manager,
        )
        record = inherited_store.create_record(
            kind="cron",
            agent_id="main",
            session_id="main-main",
            content="inherited",
            priority=2,
        )
        record.id = "inherited-work"
        inherited_store.insert(record)

        page = service.query(kind="cron")

        self.assertEqual([item["id"] for item in page.items], [record.id])

    def test_explicit_work_store_overrides_dispatcher_store(self) -> None:
        dispatcher_store = SessionWorkStore(self.root / "dispatcher-work.db")
        service = TaskHistoryService(
            task_store=self.task_store,
            work_store=self.work_store,
            dispatcher_manager=DispatcherManager(work_store=dispatcher_store),
        )

        self.assertIs(service._work_store, self.work_store)

    def test_default_runtime_keeps_global_store_and_dispatcher(self) -> None:
        service = TaskHistoryService(task_store=self.task_store)

        self.assertIs(service._work_store, session_work_store)
        self.assertIs(service._dispatcher_manager, dispatcher_manager)

    def test_uses_explicit_session_work_runtime(self) -> None:
        dispatcher_manager = DispatcherManager(work_store=self.work_store)
        runtime = SessionWorkRuntime(
            work_store=self.work_store,
            lock_manager=dispatcher_manager.lock_manager,
            dispatcher_manager=dispatcher_manager,
        )

        service = TaskHistoryService(
            task_store=self.task_store,
            runtime=runtime,
        )

        self.assertIs(service._work_store, runtime.work_store)
        self.assertIs(
            service._dispatcher_manager,
            runtime.dispatcher_manager,
        )

    def test_rejects_runtime_mixed_with_legacy_dependencies(self) -> None:
        dispatcher_manager = DispatcherManager(work_store=self.work_store)
        runtime = SessionWorkRuntime(
            work_store=self.work_store,
            lock_manager=dispatcher_manager.lock_manager,
            dispatcher_manager=dispatcher_manager,
        )

        with self.assertRaisesRegex(ValueError, "runtime cannot be combined"):
            TaskHistoryService(
                task_store=self.task_store,
                runtime=runtime,
                work_store=self.work_store,
            )

    def _insert_work(
        self,
        work_id: str,
        kind: str,
        status: str,
        created_at_ms: int,
        run_id: str | None = None,
    ):
        record = self.work_store.create_record(
            kind=kind,
            agent_id="main",
            session_id="main-main",
            content=f"content for {work_id}",
            priority=2,
            run_id=run_id,
        )
        record.id = work_id
        record.status = status
        record.created_at_ms = created_at_ms
        self.work_store.insert(record)
        return record


if __name__ == "__main__":
    unittest.main()
