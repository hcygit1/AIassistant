from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scheduler.cron_store import (
    CronStoreError,
    load_cron_store,
    save_cron_store,
)
from scheduler.cron_types import CronJob, CronStore


class CronStoreTests(unittest.TestCase):
    def test_active_work_claim_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            save_cron_store(
                CronStore(
                    jobs=[
                        CronJob(
                            id="cron-active",
                            name="active",
                            active_run_token="token-1",
                            active_run_work_id="work-1",
                        )
                    ]
                ),
                path,
            )

            restored = load_cron_store(path)

        self.assertEqual(
            restored.jobs[0].active_run_work_id,
            "work-1",
        )

    def test_failed_save_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            original = CronStore(
                jobs=[CronJob(id="cron-old", name="old")]
            )
            save_cron_store(original, path)

            with (
                patch(
                    "scheduler.cron_store.json.dump",
                    side_effect=RuntimeError("serialize failed"),
                ),
                self.assertRaises(RuntimeError),
            ):
                save_cron_store(
                    CronStore(
                        jobs=[
                            CronJob(id="cron-new", name="new")
                        ]
                    ),
                    path,
                )

            restored = load_cron_store(path)
            self.assertEqual(
                [job.id for job in restored.jobs],
                ["cron-old"],
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_corrupt_store_raises_instead_of_becoming_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text("{broken", encoding="utf-8")

            with self.assertRaises(CronStoreError):
                load_cron_store(path)

    def test_partially_corrupt_jobs_are_not_silently_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text(
                '{"version": 1, "jobs": '
                '[{"id": "cron-ok", "name": "ok"}, '
                '{"name": "missing id"}]}',
                encoding="utf-8",
            )

            with self.assertRaises(CronStoreError):
                load_cron_store(path)


if __name__ == "__main__":
    unittest.main()
