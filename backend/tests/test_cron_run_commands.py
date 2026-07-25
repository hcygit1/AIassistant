from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.cron_errors import CronServiceError
from scheduler.cron_run_commands import CronRunCommands, RunReceipt
from scheduler.cron_types import (
    CronJob,
    CronPayload,
    CronSchedule,
    CronStore,
)


class CronRunCommandsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("/tmp/cron-run-commands.json")
        self.job = CronJob(
            id="cron-1",
            name="report",
            agent_id="main",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            payload=CronPayload(kind="systemEvent", text="report"),
            schedule_revision=3,
        )
        self.store = CronStore(jobs=[self.job])
        self.transaction_depth = 0
        self.save_store = Mock(side_effect=self._assert_in_transaction)
        self.ensure_enabled = Mock()
        self.lifecycle = Mock()
        self.lifecycle.deliver_job.return_value = 4
        self.deliver = Mock(return_value=2)
        self.commands = CronRunCommands(
            transaction=self._transaction,
            save_store=self.save_store,
            ensure_enabled=self.ensure_enabled,
            now_ms=lambda: 1_000_000,
            token_factory=lambda: "token-1",
            run_lifecycle=self.lifecycle,
            deliver=self.deliver,
        )

    @contextmanager
    def _transaction(self):
        self.transaction_depth += 1
        try:
            yield self.store, self.path
        finally:
            self.transaction_depth -= 1

    def _assert_in_transaction(self, *_args) -> None:
        self.assertGreater(self.transaction_depth, 0)

    def test_trigger_claims_and_delivers_detached_job(self) -> None:
        receipt = self.commands.trigger_job("cron-1", agent_id="main")

        self.assertEqual(receipt, RunReceipt("cron-1", 4))
        self.assertEqual(self.job.active_run_token, "token-1")
        self.assertIsNone(self.job.active_run_work_id)
        self.assertIsNone(self.job.active_run_due_at_ms)
        self.assertEqual(self.job.active_run_schedule_revision, 3)
        self.assertEqual(self.job.last_run_at_ms, 1_000_000)
        self.assertEqual(self.job.last_run_status, "running")
        delivered_job = self.lifecycle.deliver_job.call_args.args[0]
        self.assertIsNot(delivered_job, self.job)
        self.assertEqual(delivered_job, self.job)
        self.lifecycle.deliver_job.assert_called_once_with(
            delivered_job,
            "token-1",
            attempted_at_ms=1_000_000,
        )
        self.save_store.assert_called_once_with(self.store, self.path)

    def test_trigger_rejects_busy_or_out_of_scope_job(self) -> None:
        self.job.active_run_token = "active-token"

        with self.assertRaises(CronServiceError) as busy:
            self.commands.trigger_job("cron-1", agent_id="main")
        with self.assertRaises(CronServiceError) as missing:
            self.commands.trigger_job("cron-1", agent_id="worker")

        self.assertEqual(busy.exception.code, "busy")
        self.assertEqual(missing.exception.code, "not_found")
        self.lifecycle.deliver_job.assert_not_called()

    def test_delivery_failure_finalizes_claim_and_preserves_cause(
        self,
    ) -> None:
        error = RuntimeError("queue unavailable")
        self.lifecycle.deliver_job.side_effect = error

        with self.assertRaises(CronServiceError) as raised:
            self.commands.trigger_job("cron-1")

        self.assertEqual(raised.exception.code, "delivery_failed")
        self.assertIs(raised.exception.__cause__, error)
        self.lifecycle.finalize_claim.assert_called_once_with(
            "cron-1",
            "token-1",
            status="error",
        )

    def test_cleanup_failure_does_not_replace_delivery_error(self) -> None:
        delivery_error = RuntimeError("queue unavailable")
        cleanup_error = OSError("store unavailable")
        self.lifecycle.deliver_job.side_effect = delivery_error
        self.lifecycle.finalize_claim.side_effect = cleanup_error

        with (
            self.assertLogs(
                "scheduler.cron_run_commands",
                level="ERROR",
            ) as logs,
            self.assertRaises(CronServiceError) as raised,
        ):
            self.commands.trigger_job("cron-1")

        self.assertEqual(raised.exception.code, "delivery_failed")
        self.assertIs(raised.exception.__cause__, delivery_error)
        self.assertIn("Failed to finalize manual Cron claim", logs.output[0])

    def test_wake_validates_text_and_uses_main_fallback(self) -> None:
        receipt = self.commands.wake(agent_id="", text=" wake ")

        self.assertEqual(receipt, RunReceipt("cron:wake", 2))
        self.deliver.assert_called_once_with(
            agent_id="main",
            text="wake",
            run_id="cron:wake",
        )
        with self.assertRaises(CronServiceError) as raised:
            self.commands.wake(agent_id="main", text="   ")
        self.assertEqual(raised.exception.code, "invalid_payload")


if __name__ == "__main__":
    unittest.main()
