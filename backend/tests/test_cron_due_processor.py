from __future__ import annotations

import copy
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.cron_due_processor import (
    CronDueProcessor,
    ProcessDueResult,
)
from scheduler.cron_types import CronJob, CronPayload, CronSchedule, CronStore


class CronDueProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("/tmp/cron-due-processor.json")
        self.job = CronJob(
            id="cron-1",
            name="interval",
            created_at_ms=1_000_000,
            next_run_at_ms=1_060_000,
            schedule=CronSchedule(kind="every", every_ms=60_000),
            payload=CronPayload(kind="systemEvent", text="wake"),
        )
        self.store = CronStore(jobs=[self.job])
        self.save_store = Mock()
        self.deliver_job = Mock()
        self.finalize_claim = Mock(return_value=True)
        self.processor = CronDueProcessor(
            transaction=self._transaction,
            save_store=self.save_store,
            deliver_job=self.deliver_job,
            finalize_claim=self.finalize_claim,
            token_factory=lambda: "token-1",
        )

    @contextmanager
    def _transaction(self):
        yield self.store, self.path

    def test_claims_due_job_and_schedules_next_wake(self) -> None:
        result = self.processor.process(1_060_000)

        self.assertEqual(result, ProcessDueResult(1, 0, 1_120_000))
        self.assertEqual(self.job.active_run_token, "token-1")
        self.assertEqual(self.job.active_run_due_at_ms, 1_060_000)
        self.assertEqual(self.job.last_run_at_ms, 1_060_000)
        self.assertEqual(self.job.last_run_status, "running")
        self.assertEqual(self.job.next_run_at_ms, 1_120_000)
        delivered_job, delivered_token = self.deliver_job.call_args.args[:2]
        self.assertIsNot(delivered_job, self.job)
        self.assertEqual(delivered_job, copy.deepcopy(self.job))
        self.assertEqual(delivered_token, "token-1")
        self.assertEqual(
            self.deliver_job.call_args.kwargs["attempted_at_ms"],
            1_060_000,
        )

    def test_stale_claim_is_reclaimed_and_delivery_failure_is_finalized(self) -> None:
        self.job.active_run_token = "stale-token"
        self.job.active_run_work_id = "work-old"
        self.job.active_run_due_at_ms = 1_000_000
        self.job.active_run_schedule_revision = 0
        self.job.last_run_at_ms = 700_000
        self.deliver_job.side_effect = RuntimeError("queue unavailable")

        result = self.processor.process(1_000_001)

        self.assertEqual(result.failed, 1)
        self.assertEqual(result.fired, 0)
        self.assertEqual(self.job.active_run_token, "token-1")
        self.assertIsNone(self.job.active_run_work_id)
        self.finalize_claim.assert_called_once_with(
            "cron-1",
            "token-1",
            status="error",
            attempted_at_ms=1_000_001,
        )

    def test_cron_service_delegates_to_injected_due_processor(self) -> None:
        from scheduler.cron_service import CronService

        expected = ProcessDueResult(2, 1, 1_100_000)
        due_processor = Mock()
        due_processor.process.return_value = expected
        service = CronService(
            load_store=lambda _path: CronStore(),
            save_store=lambda _store, _path: None,
            resolve_store_path=lambda: self.path,
            is_enabled=lambda: True,
            deliver=lambda **_kwargs: 1,
            due_processor=due_processor,
        )

        result = service.process_due_jobs(now_ms=1_000_000)

        self.assertIs(result, expected)
        due_processor.process.assert_called_once_with(1_000_000)

    def test_default_processor_keeps_dynamic_compute_entrypoint(self) -> None:
        from scheduler.cron_service import CronService

        self.job.next_run_at_ms = None
        service = CronService(
            load_store=lambda _path: self.store,
            save_store=lambda _store, _path: None,
            resolve_store_path=lambda: self.path,
            is_enabled=lambda: True,
            deliver=lambda **_kwargs: 1,
        )

        with patch(
            "scheduler.cron_service.compute_next_run",
            return_value=1_100_000,
        ) as next_run:
            service.process_due_jobs(now_ms=1_000_000)

        next_run.assert_called_once_with(self.job, 1_000_000, None)


if __name__ == "__main__":
    unittest.main()
