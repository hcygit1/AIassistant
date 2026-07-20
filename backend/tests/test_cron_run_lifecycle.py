from __future__ import annotations

import copy
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.cron_run_lifecycle import CronRunLifecycle
from scheduler.cron_types import CronJob, CronPayload, CronSchedule, CronStore


class CronRunLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("/tmp/cron-run-lifecycle.json")
        self.job = CronJob(
            id="cron-1",
            name="one time",
            agent_id="main",
            enabled=True,
            delete_after_run=True,
            schedule=CronSchedule(
                kind="at",
                at="1970-01-01T00:16:40+00:00",
            ),
            payload=CronPayload(kind="systemEvent", text="remind me"),
            active_run_token="token-1",
            active_run_due_at_ms=1_000_000,
            active_run_schedule_revision=0,
            schedule_revision=0,
            last_run_at_ms=1_000_000,
            last_run_status="running",
        )
        self.store = CronStore(jobs=[self.job])
        self.save_store = Mock()
        self.deliver = Mock(return_value=2)
        self.lifecycle = CronRunLifecycle(
            transaction=self._transaction,
            save_store=self.save_store,
            list_jobs=lambda: copy.deepcopy(self.store.jobs),
            deliver=self.deliver,
            retry_delay_ms=60_000,
        )

    @contextmanager
    def _transaction(self):
        yield self.store, self.path

    def test_delivery_binds_work_and_success_finalizes_matching_claim(self) -> None:
        position = self.lifecycle.deliver_job(
            copy.deepcopy(self.job),
            "token-1",
            attempted_at_ms=1_000_000,
        )
        delivery = self.deliver.call_args.kwargs

        self.assertEqual(position, 2)
        self.assertTrue(
            delivery["on_record_created"](
                SimpleNamespace(id="work-1")
            )
        )
        self.assertEqual(self.job.active_run_work_id, "work-1")
        self.assertIsNone(
            self.lifecycle.recovery_callbacks("cron-1", "stale-work")
        )

        delivery["on_success"]()

        self.assertEqual(self.store.jobs, [])

    def test_reconcile_maps_terminal_work_status_to_claim_result(self) -> None:
        self.job.active_run_work_id = "work-1"
        get_work = Mock(
            return_value=SimpleNamespace(id="work-1", status="failed")
        )

        reconciled = self.lifecycle.reconcile_active_work(get_work)

        self.assertEqual(reconciled, 1)
        self.assertEqual(self.job.last_run_status, "error")
        self.assertEqual(self.job.next_run_at_ms, 1_060_000)
        self.assertIsNone(self.job.active_run_token)
        get_work.assert_called_once_with("work-1")


if __name__ == "__main__":
    unittest.main()
