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
from scheduler.cron_job_catalog import CronJobCatalog
from scheduler.cron_schedule import schedule_state
from scheduler.cron_types import (
    CronPayload,
    CronSchedule,
    CronStore,
)


class CronJobCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("/tmp/cron-job-catalog.json")
        self.store = CronStore()
        self.transaction_depth = 0
        self.transaction_entries = 0
        self.save_store = Mock(side_effect=self._assert_in_transaction)
        self.ensure_enabled = Mock()
        self.now_ms = 1_000_000
        self.build_schedule = Mock(
            return_value=CronSchedule(
                kind="every",
                every_ms=60_000,
            )
        )
        self.build_payload = Mock(
            return_value=CronPayload(
                kind="systemEvent",
                text="wake",
            )
        )
        self.next_run = Mock(return_value=1_060_000)
        self.catalog = CronJobCatalog(
            transaction=self._transaction,
            save_store=self.save_store,
            ensure_enabled=self.ensure_enabled,
            now_ms=lambda: self.now_ms,
            id_factory=self._next_id,
            build_schedule=self.build_schedule,
            build_payload=self.build_payload,
            next_run=self.next_run,
            schedule_state=schedule_state,
            not_found=lambda job_id: CronServiceError(
                "not_found",
                f"Job {job_id} not found",
            ),
        )

    @contextmanager
    def _transaction(self):
        self.transaction_entries += 1
        self.transaction_depth += 1
        try:
            yield self.store, self.path
        finally:
            self.transaction_depth -= 1

    def _assert_in_transaction(self, *_args) -> None:
        self.assertGreater(self.transaction_depth, 0)

    def _next_id(self) -> str:
        self.assertGreater(self.transaction_depth, 0)
        return "id0000000001"

    def test_create_and_list_return_detached_jobs(self) -> None:
        created = self.catalog.create_job(
            name=" report ",
            description=" daily ",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "wake"},
        )

        created.name = "tampered"
        listed = self.catalog.list_jobs(agent_id="main")
        listed[0].name = "listed mutation"
        found = self.catalog.find_job(created.id)
        found.name = "found mutation"
        current = self.catalog.get_job(created.id)

        self.assertEqual(created.id, "cron-id0000000001")
        self.assertEqual(current.name, "report")
        self.assertEqual(current.description, "daily")
        self.assertEqual(current.next_run_at_ms, 1_060_000)
        self.save_store.assert_called_once_with(self.store, self.path)
        self.build_schedule.assert_called_once_with(
            {"kind": "every", "everyMs": 60_000},
            now_ms=1_000_000,
        )

    def test_update_and_delete_preserve_scope_and_revision(self) -> None:
        created = self.catalog.create_job(
            name="report",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "wake"},
        )
        self.build_schedule.return_value = CronSchedule(
            kind="every",
            every_ms=120_000,
        )
        self.next_run.return_value = 1_120_000

        updated = self.catalog.update_job(
            created.id,
            description="updated",
            schedule={"everyMs": 120_000},
            scope_agent_id="main",
        )

        self.assertEqual(updated.description, "updated")
        self.assertEqual(updated.schedule_revision, 1)
        self.assertEqual(updated.next_run_at_ms, 1_120_000)
        updated.description = "tampered"
        self.assertEqual(
            self.catalog.get_job(created.id).description,
            "updated",
        )
        self.assertEqual(self.catalog.list_jobs(agent_id=""), [])
        self.assertIsNone(
            self.catalog.find_job(created.id, agent_id="   ")
        )
        with self.assertRaises(CronServiceError) as raised:
            self.catalog.delete_job(created.id, agent_id="worker")
        self.assertEqual(raised.exception.code, "not_found")
        self.assertTrue(
            self.catalog.delete_job(created.id, agent_id="main")
        )
        self.assertEqual(self.store.jobs, [])

    def test_revision_changes_only_for_execution_state(self) -> None:
        created = self.catalog.create_job(
            name="report",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "wake"},
        )

        metadata = self.catalog.update_job(
            created.id,
            name="renamed",
            description="updated",
            agent_id="worker",
            payload={"kind": "systemEvent", "text": "changed"},
        )
        disabled = self.catalog.update_job(
            created.id,
            enabled=False,
        )
        one_time = self.catalog.update_job(
            created.id,
            delete_after_run=True,
        )

        self.assertEqual(metadata.schedule_revision, 0)
        self.assertEqual(disabled.schedule_revision, 1)
        self.assertEqual(one_time.schedule_revision, 2)

    def test_disabled_writes_fail_before_time_and_transaction(self) -> None:
        created = self.catalog.create_job(
            name="report",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "wake"},
        )
        self.ensure_enabled.side_effect = CronServiceError(
            "disabled",
            "cron is disabled",
        )
        self.save_store.reset_mock()
        entries_before = self.transaction_entries

        self.assertEqual(len(self.catalog.list_jobs()), 1)
        read_entries = self.transaction_entries
        self.assertEqual(read_entries, entries_before + 1)
        for operation in (
            lambda: self.catalog.create_job(
                name="blocked",
                schedule={"kind": "every", "everyMs": 60_000},
                payload={"kind": "systemEvent", "text": "blocked"},
            ),
            lambda: self.catalog.update_job(created.id, name="blocked"),
            lambda: self.catalog.delete_job(created.id),
        ):
            with self.assertRaises(CronServiceError) as raised:
                operation()
            self.assertEqual(raised.exception.code, "disabled")

        self.assertEqual(self.transaction_entries, read_entries)
        self.save_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
