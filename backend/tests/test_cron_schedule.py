from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.cron_errors import CronServiceError
from scheduler.cron_schedule import (
    build_payload,
    build_schedule,
    compute_next_run,
    schedule_state,
)
from scheduler.cron_types import CronJob, CronSchedule


class CronScheduleTests(unittest.TestCase):
    def test_build_schedule_preserves_partial_update_fields(self) -> None:
        current = CronSchedule(kind="every", every_ms=60_000)

        schedule = build_schedule(
            {"everyMs": 120_000},
            current=current,
            now_ms=1_000_000,
            default_timezone=lambda: "UTC",
        )

        self.assertEqual(schedule.kind, "every")
        self.assertEqual(schedule.every_ms, 120_000)

    def test_build_schedule_and_payload_preserve_validation_errors(self) -> None:
        with self.assertRaises(CronServiceError) as schedule_error:
            build_schedule(
                {"kind": "cron", "expr": "0 8 * * *", "tz": "Mars/Base"},
                now_ms=1_000_000,
                default_timezone=lambda: "UTC",
            )
        self.assertEqual(schedule_error.exception.code, "invalid_schedule")

        with self.assertRaises(CronServiceError) as payload_error:
            build_payload({"kind": "systemEvent", "text": ""})
        self.assertEqual(payload_error.exception.code, "invalid_payload")

    def test_compute_and_state_helpers_keep_existing_contract(self) -> None:
        job = CronJob(
            id="cron-1",
            name="interval",
            created_at_ms=1_000_000,
            schedule=CronSchedule(kind="every", every_ms=60_000),
        )

        self.assertEqual(compute_next_run(job, 1_000_000), 1_060_000)
        self.assertEqual(
            schedule_state(job),
            (True, False, "every", None, 60_000, None, None),
        )

    def test_cron_service_reexports_compatibility_types_and_helpers(self) -> None:
        from scheduler.cron_service import (
            CronServiceError as ServiceError,
            compute_next_run as service_compute_next_run,
        )

        self.assertIs(ServiceError, CronServiceError)
        self.assertIs(service_compute_next_run, compute_next_run)


if __name__ == "__main__":
    unittest.main()
