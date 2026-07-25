from __future__ import annotations

import copy
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.cron_types import CronStore
from scheduler.cron_service import (
    CronService,
    CronServiceError,
)


class _MemoryCronStore:
    def __init__(self) -> None:
        self.store = CronStore()

    def load(self, path: Path) -> CronStore:
        return copy.deepcopy(self.store)

    def save(self, store: CronStore, path: Path) -> None:
        self.store = copy.deepcopy(store)


class CronServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.persistence = _MemoryCronStore()
        self.deliveries: list[dict] = []
        self.now_ms = 1_000_000
        self.ids = iter(["id0000000001", "id0000000002", "id0000000003"])
        self.enabled = True
        self.service = CronService(
            load_store=self.persistence.load,
            save_store=self.persistence.save,
            resolve_store_path=lambda: Path("/tmp/cron-test.json"),
            is_enabled=lambda: self.enabled,
            deliver=lambda **kwargs: self.deliveries.append(kwargs) or 1,
            now_ms=lambda: self.now_ms,
            id_factory=lambda: next(self.ids),
            default_timezone=lambda: "UTC",
        )

    def test_crud_returns_detached_jobs_and_preserves_partial_schedule(self) -> None:
        created = self.service.create_job(
            name="daily report",
            description="initial",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "report"},
        )

        created.name = "tampered"
        current = self.service.get_job(created.id)
        self.assertEqual(current.name, "daily report")

        updated = self.service.update_job(
            created.id,
            description="updated",
            schedule={"everyMs": 120_000},
        )

        self.assertEqual(updated.description, "updated")
        self.assertEqual(updated.schedule.kind, "every")
        self.assertEqual(updated.schedule.every_ms, 120_000)
        self.assertEqual(updated.payload.text, "report")
        self.assertTrue(self.service.delete_job(created.id))
        self.assertEqual(self.service.list_jobs(), [])

    def test_crud_methods_delegate_to_injected_catalog(self) -> None:
        catalog = Mock()
        expected = SimpleNamespace(id="cron-1")
        catalog.create_job.return_value = expected
        service = CronService(
            load_store=self.persistence.load,
            save_store=self.persistence.save,
            resolve_store_path=lambda: Path("/tmp/cron-test.json"),
            is_enabled=lambda: True,
            deliver=lambda **_kwargs: 1,
            job_catalog=catalog,
        )

        result = service.create_job(
            name="report",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "report"},
        )

        self.assertIs(result, expected)
        catalog.create_job.assert_called_once_with(
            name="report",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "report"},
            description="",
            enabled=True,
            delete_after_run=False,
            id_prefix="cron",
        )
        catalog.list_jobs.return_value = [expected]
        catalog.find_job.return_value = expected
        catalog.get_job.return_value = expected
        catalog.update_job.return_value = expected
        catalog.delete_job.return_value = True

        self.assertEqual(service.list_jobs(agent_id="main"), [expected])
        self.assertIs(service.find_job("cron-1", agent_id="main"), expected)
        self.assertIs(service.get_job("cron-1", agent_id="main"), expected)
        self.assertIs(
            service.update_job("cron-1", name="renamed"),
            expected,
        )
        self.assertTrue(service.delete_job("cron-1", agent_id="main"))
        catalog.list_jobs.assert_called_once_with(agent_id="main")
        catalog.find_job.assert_called_once_with(
            "cron-1",
            agent_id="main",
        )
        catalog.get_job.assert_called_once_with(
            "cron-1",
            agent_id="main",
        )
        catalog.update_job.assert_called_once_with(
            "cron-1",
            name="renamed",
            description=None,
            agent_id=None,
            enabled=None,
            delete_after_run=None,
            schedule=None,
            payload=None,
            scope_agent_id=None,
        )
        catalog.delete_job.assert_called_once_with(
            "cron-1",
            agent_id="main",
        )

    def test_catalog_keeps_dynamic_next_run_entrypoint(self) -> None:
        with patch(
            "scheduler.cron_service.compute_next_run",
            return_value=1_500_000,
        ) as next_run:
            job = self._create_job("main", "dynamic")

        self.assertEqual(job.next_run_at_ms, 1_500_000)
        next_run.assert_called_once_with(
            ANY,
            self.now_ms,
            None,
        )

    def test_manual_run_methods_delegate_to_injected_commands(self) -> None:
        commands = Mock()
        trigger_receipt = SimpleNamespace(
            job_id="cron-1",
            queue_position=2,
        )
        wake_receipt = SimpleNamespace(
            job_id="cron:wake",
            queue_position=3,
        )
        commands.trigger_job.return_value = trigger_receipt
        commands.wake.return_value = wake_receipt
        service = CronService(
            load_store=self.persistence.load,
            save_store=self.persistence.save,
            resolve_store_path=lambda: Path("/tmp/cron-test.json"),
            is_enabled=lambda: True,
            deliver=lambda **_kwargs: 1,
            run_commands=commands,
        )

        self.assertIs(
            service.trigger_job("cron-1", agent_id="main"),
            trigger_receipt,
        )
        self.assertIs(
            service.wake(agent_id="main", text="wake"),
            wake_receipt,
        )
        commands.trigger_job.assert_called_once_with(
            "cron-1",
            agent_id="main",
        )
        commands.wake.assert_called_once_with(
            agent_id="main",
            text="wake",
        )

    def test_invalid_schedule_and_empty_payload_are_rejected(self) -> None:
        with self.assertRaises(CronServiceError) as schedule_error:
            self.service.create_job(
                name="invalid",
                agent_id="main",
                schedule={"kind": "every", "everyMs": 0},
                payload={"kind": "systemEvent", "text": "wake"},
            )
        self.assertEqual(schedule_error.exception.code, "invalid_schedule")

        with self.assertRaises(CronServiceError) as payload_error:
            self.service.create_job(
                name="invalid",
                agent_id="main",
                schedule={"kind": "cron", "expr": "0 8 * * *"},
                payload={"kind": "systemEvent", "text": ""},
            )
        self.assertEqual(payload_error.exception.code, "invalid_payload")

    def test_agent_scope_prevents_tool_from_mutating_other_jobs(self) -> None:
        main_job = self._create_job("main", "main task")
        worker_job = self._create_job("worker", "worker task")

        scoped = self.service.list_jobs(agent_id="main")

        self.assertEqual([job.id for job in scoped], [main_job.id])
        with self.assertRaises(CronServiceError) as raised:
            self.service.delete_job(
                worker_job.id,
                agent_id="main",
            )
        self.assertEqual(raised.exception.code, "not_found")

    def test_concurrent_creates_do_not_lose_jobs(self) -> None:
        barrier = threading.Barrier(3)
        errors: list[Exception] = []

        def create(name: str) -> None:
            barrier.wait()
            try:
                self.service.create_job(
                    name=name,
                    agent_id="main",
                    schedule={
                        "kind": "every",
                        "everyMs": 60_000,
                    },
                    payload={
                        "kind": "systemEvent",
                        "text": name,
                    },
                )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=create, args=("one",)),
            threading.Thread(target=create, args=("two",)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=1)

        self.assertEqual(errors, [])
        self.assertEqual(
            sorted(job.name for job in self.service.list_jobs()),
            ["one", "two"],
        )

    def test_manual_and_due_triggers_use_same_delivery(self) -> None:
        recurring = self._create_job("main", "recurring")
        one_time = self.service.create_job(
            name="one time",
            agent_id="main",
            delete_after_run=True,
            schedule={
                "kind": "at",
                "at": "1970-01-01T00:16:41+00:00",
            },
            payload={
                "kind": "systemEvent",
                "text": "one time",
            },
        )

        self.service.trigger_job(recurring.id)
        processed = self.service.process_due_jobs(
            now_ms=1_001_000
        )

        self.assertEqual(processed.fired, 1)
        self.assertEqual(len(self.deliveries), 2)
        self.assertEqual(
            {item["run_id"] for item in self.deliveries},
            {recurring.id, one_time.id},
        )
        current = self.service.get_job(one_time.id)
        self.assertEqual(current.last_run_status, "running")
        one_time_delivery = next(
            item
            for item in self.deliveries
            if item["run_id"] == one_time.id
        )
        one_time_delivery["on_success"]()
        self.assertIsNone(
            self.service.find_job(one_time.id)
        )

    def test_execution_failure_retries_one_time_job_after_queueing(self) -> None:
        job = self.service.create_job(
            name="one time failure",
            agent_id="main",
            schedule={
                "kind": "at",
                "at": "1970-01-01T00:16:41+00:00",
            },
            payload={"kind": "systemEvent", "text": "retry me"},
        )

        result = self.service.process_due_jobs(now_ms=1_001_000)
        self.deliveries[0]["on_failure"]()

        current = self.service.get_job(job.id)
        self.assertEqual(result.fired, 1)
        self.assertEqual(current.last_run_status, "error")
        self.assertEqual(current.next_run_at_ms, 1_061_000)

    def test_recovered_work_finalizes_only_its_bound_claim(self) -> None:
        def deliver(**kwargs):
            kwargs["on_record_created"](
                SimpleNamespace(id="work-1")
            )
            self.deliveries.append(kwargs)
            return 1

        self.service._deliver = deliver
        job = self.service.create_job(
            name="recoverable one time",
            agent_id="main",
            delete_after_run=True,
            schedule={
                "kind": "at",
                "at": "1970-01-01T00:16:41+00:00",
            },
            payload={"kind": "systemEvent", "text": "recover me"},
        )

        self.service.process_due_jobs(now_ms=1_001_000)

        current = self.service.get_job(job.id)
        self.assertEqual(current.active_run_work_id, "work-1")
        self.assertIsNone(
            self.service.recovery_callbacks(job.id, "stale-work"),
        )
        callbacks = self.service.recovery_callbacks(job.id, "work-1")
        callbacks["on_success"]()
        self.assertIsNone(self.service.find_job(job.id))

    def test_reconcile_finalizes_terminal_bound_work_after_restart(self) -> None:
        def deliver(**kwargs):
            kwargs["on_record_created"](
                SimpleNamespace(id="work-done")
            )
            return 1

        self.service._deliver = deliver
        job = self.service.create_job(
            name="reconcile one time",
            agent_id="main",
            delete_after_run=True,
            schedule={
                "kind": "at",
                "at": "1970-01-01T00:16:41+00:00",
            },
            payload={"kind": "systemEvent", "text": "done"},
        )
        self.service.process_due_jobs(now_ms=1_001_000)

        reconciled = self.service.reconcile_active_work(
            lambda work_id: SimpleNamespace(
                id=work_id,
                status="done",
            )
        )

        self.assertEqual(reconciled, 1)
        self.assertIsNone(self.service.find_job(job.id))

    def test_disabled_service_allows_reads_but_rejects_mutations(self) -> None:
        self._create_job("main", "existing")
        self.enabled = False

        self.assertEqual(len(self.service.list_jobs()), 1)
        with self.assertRaises(CronServiceError) as raised:
            self._create_job("main", "blocked")
        self.assertEqual(raised.exception.code, "disabled")

    def test_delivery_callback_cannot_be_overwritten_by_old_snapshot(self) -> None:
        job = self._create_job("main", "original")

        def deliver(**kwargs):
            self.service.update_job(
                job.id,
                name="updated during delivery",
            )
            return 1

        self.service._deliver = deliver

        self.service.trigger_job(job.id)

        self.assertEqual(
            self.service.get_job(job.id).name,
            "updated during delivery",
        )

    def test_stale_due_claim_retries_missed_recurring_run(self) -> None:
        job = self.service.create_job(
            name="hourly",
            agent_id="main",
            schedule={"kind": "every", "everyMs": 3_600_000},
            payload={"kind": "systemEvent", "text": "hourly"},
        )

        self.service._deliver = lambda **kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt()
        )
        with self.assertRaises(KeyboardInterrupt):
            self.service.process_due_jobs(now_ms=4_600_000)

        self.service._deliver = (
            lambda **kwargs: self.deliveries.append(kwargs) or 1
        )
        recovered = self.service.process_due_jobs(now_ms=4_900_001)

        self.assertEqual(recovered.fired, 1)
        self.assertEqual(len(self.deliveries), 1)
        self.assertEqual(self.deliveries[0]["run_id"], job.id)

    def test_delivery_does_not_apply_old_completion_to_new_schedule(self) -> None:
        job = self._create_job("main", "recurring")

        def deliver(**kwargs):
            self.service.update_job(
                job.id,
                delete_after_run=True,
                schedule={
                    "kind": "at",
                    "at": "1970-01-01T00:18:40+00:00",
                },
            )
            return 1

        self.service._deliver = deliver

        result = self.service.process_due_jobs(now_ms=1_060_000)

        current = self.service.get_job(job.id)
        self.assertEqual(result.fired, 1)
        self.assertTrue(current.enabled)
        self.assertEqual(current.schedule.kind, "at")
        self.assertEqual(current.next_run_at_ms, 1_120_000)

    def test_overdue_one_time_failure_retries_after_attempt(self) -> None:
        job = self.service.create_job(
            name="overdue",
            agent_id="main",
            schedule={
                "kind": "at",
                "at": "1970-01-01T00:16:41+00:00",
            },
            payload={"kind": "systemEvent", "text": "overdue"},
        )
        self.service._deliver = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("queue unavailable")
        )

        result = self.service.process_due_jobs(now_ms=1_100_000)

        current = self.service.get_job(job.id)
        self.assertEqual(result.failed, 1)
        self.assertEqual(current.next_run_at_ms, 1_160_000)

    def test_stale_manual_claim_is_cleared_for_disabled_job(self) -> None:
        job = self.service.create_job(
            name="disabled",
            agent_id="main",
            enabled=False,
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": "disabled"},
        )
        self.service._deliver = lambda **kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt()
        )
        with self.assertRaises(KeyboardInterrupt):
            self.service.trigger_job(job.id)

        self.service.process_due_jobs(now_ms=1_300_001)
        self.service._deliver = lambda **kwargs: 1

        receipt = self.service.trigger_job(job.id)

        self.assertEqual(receipt.job_id, job.id)

    def test_services_for_same_path_share_transaction_lock(self) -> None:
        class ObservedStore(_MemoryCronStore):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.guard = threading.Lock()
                inner_self.active_loads = 0
                inner_self.max_active_loads = 0

            def load(inner_self, path: Path) -> CronStore:
                with inner_self.guard:
                    inner_self.active_loads += 1
                    inner_self.max_active_loads = max(
                        inner_self.max_active_loads,
                        inner_self.active_loads,
                    )
                time.sleep(0.03)
                result = super().load(path)
                with inner_self.guard:
                    inner_self.active_loads -= 1
                return result

        persistence = ObservedStore()

        def make_service(identifier: str) -> CronService:
            return CronService(
                load_store=persistence.load,
                save_store=persistence.save,
                resolve_store_path=lambda: Path(
                    "/tmp/shared-cron-test.json"
                ),
                is_enabled=lambda: True,
                deliver=lambda **kwargs: 1,
                now_ms=lambda: self.now_ms,
                id_factory=lambda: identifier,
                default_timezone=lambda: "UTC",
            )

        services = [make_service("service-one"), make_service("service-two")]
        threads = [
            threading.Thread(
                target=service.create_job,
                kwargs={
                    "name": f"job-{index}",
                    "agent_id": "main",
                    "schedule": {
                        "kind": "every",
                        "everyMs": 60_000,
                    },
                    "payload": {
                        "kind": "systemEvent",
                        "text": f"job-{index}",
                    },
                },
            )
            for index, service in enumerate(services)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(persistence.max_active_loads, 1)
        self.assertEqual(len(persistence.store.jobs), 2)

    def test_missing_timezone_is_persisted_as_utc(self) -> None:
        at_job = self.service.create_job(
            name="at",
            agent_id="main",
            schedule={
                "kind": "at",
                "at": "1970-01-01T00:16:41",
            },
            payload={"kind": "systemEvent", "text": "at"},
        )
        cron_job = self.service.create_job(
            name="cron",
            agent_id="main",
            schedule={"kind": "cron", "expr": "0 8 * * *"},
            payload={"kind": "systemEvent", "text": "cron"},
        )

        self.assertEqual(at_job.schedule.tz, "UTC")
        self.assertEqual(cron_job.schedule.tz, "UTC")

    def _create_job(self, agent_id: str, name: str):
        return self.service.create_job(
            name=name,
            agent_id=agent_id,
            schedule={"kind": "every", "everyMs": 60_000},
            payload={"kind": "systemEvent", "text": name},
        )


if __name__ == "__main__":
    unittest.main()
