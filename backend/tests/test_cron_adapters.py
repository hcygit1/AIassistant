from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.cron_api import CronJobCreate, create_cron_job
from scheduler.cron_service import ProcessDueResult
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
