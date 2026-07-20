from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.cron_api import (
    CronJobCreate,
    cancel_task,
    create_cron_job,
    get_system_work_history,
    get_task_history,
)
from scheduler.cron_service import ProcessDueResult
from scheduler.task_history_service import TaskHistoryError
from scheduler.cron_types import CronJob, CronPayload, CronSchedule
from scheduler.cron_scheduler import CronScheduler
from tools.cron_tools import get_cron_tools


class CronAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_create_delegates_to_service(self) -> None:
        service = Mock()
        service.create_job.return_value = self._job()
        body = CronJobCreate(
            name="daily report",
            description="report",
            agent_id="main",
            enabled=True,
            deleteAfterRun=False,
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "report"},
        )

        with patch(
            "scheduler.cron_service.cron_service",
            service,
        ):
            result = await create_cron_job(body)

        service.create_job.assert_called_once_with(
            name="daily report",
            description="report",
            agent_id="main",
            enabled=True,
            delete_after_run=False,
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "report"},
        )
        self.assertEqual(result["id"], "cron-id0000000001")

    async def test_tool_add_and_list_delegate_with_agent_scope(self) -> None:
        service = Mock()
        service.is_enabled.return_value = True
        service.create_job.return_value = self._job()
        service.list_jobs.return_value = [self._job()]
        tool = get_cron_tools(
            "main",
            cron_service=service,
        )[0]

        created = tool._run(
            action="add",
            name="daily report",
            description="report",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"text": "report"},
        )
        listed = tool._run(action="list")

        service.create_job.assert_called_once_with(
            name="daily report",
            description="report",
            agent_id="main",
            enabled=True,
            delete_after_run=False,
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "report"},
        )
        service.list_jobs.assert_called_once_with(agent_id="main")
        self.assertIn("已创建任务 cron-id0000000001", created)
        self.assertIn("cron-id0000000001", listed)

    async def test_tool_list_remains_available_when_cron_is_disabled(self) -> None:
        service = Mock()
        service.is_enabled.return_value = False
        service.list_jobs.return_value = [self._job()]
        tool = get_cron_tools(
            "main",
            cron_service=service,
        )[0]

        listed = tool._run(action="list")

        service.list_jobs.assert_called_once_with(agent_id="main")
        self.assertIn("cron-id0000000001", listed)

    async def test_scheduler_tick_uses_service_and_next_wake(self) -> None:
        service = Mock()
        service.process_due_jobs.return_value = ProcessDueResult(
            fired=1,
            failed=0,
            next_wake_at_ms=1_005_000,
        )
        scheduler = CronScheduler(
            service=service,
            now_ms=lambda: 1_000_000,
        )

        sleep_seconds = scheduler.tick()

        service.process_due_jobs.assert_called_once_with(
            now_ms=1_000_000
        )
        self.assertEqual(sleep_seconds, 5)

    async def test_scheduler_reconciles_terminal_work_before_loop(self) -> None:
        service = Mock()
        get_work = Mock()
        scheduler = CronScheduler(
            service=service,
            get_work=get_work,
        )

        await scheduler.start()
        await scheduler.stop()

        service.reconcile_active_work.assert_called_once_with(get_work)

    async def test_scheduler_uses_injected_session_work_runtime(self) -> None:
        service = Mock()
        work_store = Mock()
        runtime = SimpleNamespace(work_store=work_store)
        scheduler = CronScheduler(
            service=service,
            runtime=runtime,
        )

        await scheduler.start()
        await scheduler.stop()

        service.reconcile_active_work.assert_called_once_with(work_store.get)

    async def test_task_history_api_delegates_to_unified_service(self) -> None:
        service = Mock()
        service.query.return_value = SimpleNamespace(
            items=[{"id": "work-1"}],
            total=1,
            limit=10,
            offset=0,
        )

        with patch(
            "scheduler.task_history_service.task_history_service",
            service,
        ):
            result = await get_task_history(
                agent_id="main",
                kind="cron",
                status="pending",
                limit=10,
                offset=0,
            )

        service.query.assert_called_once_with(
            agent_id="main",
            kind="cron",
            status="pending",
            limit=10,
            offset=0,
        )
        self.assertEqual(result["total"], 1)

    async def test_system_work_history_delegates_to_history_service(self) -> None:
        record = SimpleNamespace(
            id="work-1",
            kind="heartbeat",
            agent_id="main",
            session_id="main-main",
            run_id=None,
            status="done",
            recover_on_restart=False,
            created_at_ms=1,
            started_at_ms=2,
            finished_at_ms=3,
            last_error=None,
            content="heartbeat prompt",
        )
        service = Mock()
        service.query.return_value = SimpleNamespace(
            items=[{
                "id": record.id,
                "kind": record.kind,
                "agent_id": record.agent_id,
                "session_id": record.session_id,
                "run_id": record.run_id,
                "status": record.status,
                "recover_on_restart": record.recover_on_restart,
                "created_at_ms": record.created_at_ms,
                "started_at_ms": record.started_at_ms,
                "finished_at_ms": record.finished_at_ms,
                "last_error": record.last_error,
                "content_preview": record.content[:200],
            }],
            total=1,
            limit=10,
            offset=0,
        )

        with patch(
            "api.cron_api._system_work_history_service",
            return_value=service,
        ):
            result = await get_system_work_history(
                kind="heartbeat",
                status="done",
                agent_id="main",
                session_id="main-main",
                run_id=None,
                limit=10,
                offset=0,
            )

        expected_filters = {
            "kind": "heartbeat",
            "status": "done",
            "agent_id": "main",
            "session_id": "main-main",
            "run_id": None,
        }
        service.query.assert_called_once_with(
            **expected_filters,
            limit=10,
            offset=0,
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["id"], "work-1")

    async def test_cancel_api_rejects_running_work(self) -> None:
        service = Mock()
        service.cancel.side_effect = TaskHistoryError(
            "running",
            "Running session work cannot be cancelled",
        )

        with (
            patch(
                "scheduler.task_history_service.task_history_service",
                service,
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await cancel_task("work-1")

        self.assertEqual(raised.exception.status_code, 409)

    @staticmethod
    def _job() -> CronJob:
        return CronJob(
            id="cron-id0000000001",
            name="daily report",
            description="report",
            agent_id="main",
            schedule=CronSchedule(
                kind="every",
                every_ms=60_000,
            ),
            payload=CronPayload(
                kind="systemEvent",
                text="report",
            ),
            created_at_ms=1_000_000,
            updated_at_ms=1_000_000,
            next_run_at_ms=1_060_000,
        )


if __name__ == "__main__":
    unittest.main()
